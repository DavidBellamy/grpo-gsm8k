from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def bootstrap_ci(
    scores: list[float], n_bootstrap: int = 1000, alpha: float = 0.05
) -> tuple[float, float]:
    """Bootstrap confidence intervals - fewer assumptions than t-distribution."""
    bootstrap_means = []
    n = len(scores)

    for _ in range(n_bootstrap):
        # Resample with replacement
        resampled = np.random.choice(scores, size=n, replace=True)
        bootstrap_means.append(np.mean(resampled))

    # Percentile method
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return lower, upper


def permutation_test(
    scores_a: list[float], scores_b: list[float], n_permutations: int = 1000
) -> float:
    """Non-parametric permutation test for difference in means."""
    observed_diff = np.mean(scores_b) - np.mean(scores_a)
    combined = scores_a + scores_b
    n_a = len(scores_a)

    diffs = []
    for _ in range(n_permutations):
        shuffled = np.random.permutation(combined)
        fake_a = shuffled[:n_a]
        fake_b = shuffled[n_a:]
        diffs.append(np.mean(fake_b) - np.mean(fake_a))

    p_value = np.mean(np.abs(diffs) >= np.abs(observed_diff))
    return p_value


def analyze_eval_variance(
    scores: list[float], baseline_scores: list[float] | None = None, alpha: float = 0.05
) -> dict[str, Any]:
    """Analyze evaluation variance with bootstrap CIs and optional comparison."""
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    ci_lower, ci_upper = bootstrap_ci(scores, alpha=alpha)

    result = {
        "mean": mean_score,
        "std": std_score,
        "n_runs": len(scores),
        "ci_95": (float(ci_lower), float(ci_upper)),
        "scores": scores,
    }

    if baseline_scores is not None:
        baseline_mean = float(np.mean(baseline_scores))
        baseline_ci = bootstrap_ci(baseline_scores, alpha=alpha)
        p_value = permutation_test(baseline_scores, scores)

        result.update(
            {
                "baseline_mean": baseline_mean,
                "baseline_ci_95": (float(baseline_ci[0]), float(baseline_ci[1])),
                "difference": mean_score - baseline_mean,
                "p_value": p_value,
                "significant": p_value < alpha,
            }
        )

        logger.info(f"Model mean: {mean_score:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
        logger.info(f"Baseline mean: {baseline_mean:.3f}")
        logger.info(f"Difference: {mean_score - baseline_mean:.3f} (p={p_value:.3f})")
        logger.info(f"Significant: {p_value < alpha}")
    else:
        logger.info(f"Mean: {mean_score:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
        logger.info(f"Std: {std_score:.3f} over {len(scores)} runs")

    return result
