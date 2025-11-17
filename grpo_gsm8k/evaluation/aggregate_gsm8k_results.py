import json
import logging
from pathlib import Path
from typing import Any

import wandb

from grpo_gsm8k.evaluation.eval import (
    _percentile,
    compute_bootstrap_ci_binary,
    compute_bootstrap_ci_percentile,
    log_gsm8k_to_wandb,
)

logger = logging.getLogger(__name__)


def run_aggregate(cfg: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(cfg["output_dir"])
    shard_files = sorted(output_dir.glob("results_shard*_of_*.json"))
    if not shard_files:
        raise FileNotFoundError(f"No shard files found in {output_dir}")

    logger.info(f"[aggregate] found {len(shard_files)} shard files under {output_dir}")

    all_results: list[dict[str, Any]] = []
    for f in shard_files:
        with f.open() as fh:
            shard = json.load(fh)
        all_results.extend(shard["gsm8k_results"])

    # Compute aggregated GSM8K metrics (same logic as in run_gsm8k_eval)
    rewards = [int(r["reward"]) for r in all_results]
    completion_lens = [int(r["completion_len"]) for r in all_results]
    truncated_flags = [1 if r["is_truncated"] else 0 for r in all_results]

    stats = compute_bootstrap_ci_binary(
        rewards, n_boot=cfg["gsm8k_bootstrap_samples"], alpha=cfg["gsm8k_ci_alpha"]
    )
    pass_at_1 = stats["mean"]
    trunc_rate = sum(truncated_flags) / len(truncated_flags)

    comp_p50 = _percentile(completion_lens, 0.5)
    comp_p95 = _percentile(completion_lens, 0.95)

    p50_ci = compute_bootstrap_ci_percentile(
        [float(x) for x in completion_lens],
        q=0.5,
        n_boot=cfg["gsm8k_bootstrap_samples"],
        alpha=cfg["gsm8k_ci_alpha"],
    )
    p95_ci = compute_bootstrap_ci_percentile(
        [float(x) for x in completion_lens],
        q=0.95,
        n_boot=cfg["gsm8k_bootstrap_samples"],
        alpha=cfg["gsm8k_ci_alpha"],
    )
    trunc_ci = compute_bootstrap_ci_binary(
        truncated_flags, n_boot=cfg["gsm8k_bootstrap_samples"], alpha=cfg["gsm8k_ci_alpha"]
    )

    # Error rates (exclude truncations from denominator)
    non_trunc = [r for r in all_results if not r["is_truncated"]]
    non_trunc_count = len(non_trunc)

    if non_trunc_count:
        fmt_flags = [
            1 if ((int(r["reward"]) == 0) and (not r["formatting_ok"])) else 0 for r in non_trunc
        ]
        logic_flags = [
            1 if ((int(r["reward"]) == 0) and r["formatting_ok"]) else 0 for r in non_trunc
        ]

        fmt_err_rate = sum(fmt_flags) / non_trunc_count
        logic_err_rate = sum(logic_flags) / non_trunc_count

        fmt_ci = compute_bootstrap_ci_binary(
            fmt_flags,
            n_boot=cfg["gsm8k_bootstrap_samples"],
            alpha=cfg["gsm8k_ci_alpha"],
        )
        logic_ci = compute_bootstrap_ci_binary(
            logic_flags,
            n_boot=cfg["gsm8k_bootstrap_samples"],
            alpha=cfg["gsm8k_ci_alpha"],
        )
    else:
        fmt_err_rate = 0.0
        logic_err_rate = 0.0
        fmt_ci = {"ci_lower": 0.0, "ci_upper": 0.0}
        logic_ci = {"ci_lower": 0.0, "ci_upper": 0.0}

    all_results_dict: dict[str, Any] = {
        "gsm8k_pass@1": pass_at_1,
        "gsm8k_n_examples": len(all_results),
        "gsm8k_results": all_results,
        "gsm8k_pass@1_ci_lower": stats["ci_lower"],
        "gsm8k_pass@1_ci_upper": stats["ci_upper"],
        "gsm8k_completion_len_p50": comp_p50,
        "gsm8k_completion_len_p95": comp_p95,
        "gsm8k_truncation_rate": trunc_rate,
        "gsm8k_format_error_rate": fmt_err_rate,
        "gsm8k_logic_error_rate": logic_err_rate,
        "gsm8k_completion_len_p50_ci_lower": p50_ci["ci_lower"],
        "gsm8k_completion_len_p50_ci_upper": p50_ci["ci_upper"],
        "gsm8k_completion_len_p95_ci_lower": p95_ci["ci_lower"],
        "gsm8k_completion_len_p95_ci_upper": p95_ci["ci_upper"],
        "gsm8k_truncation_rate_ci_lower": trunc_ci["ci_lower"],
        "gsm8k_truncation_rate_ci_upper": trunc_ci["ci_upper"],
        "gsm8k_format_error_rate_ci_lower": fmt_ci["ci_lower"],
        "gsm8k_format_error_rate_ci_upper": fmt_ci["ci_upper"],
        "gsm8k_logic_error_rate_ci_lower": logic_ci["ci_lower"],
        "gsm8k_logic_error_rate_ci_upper": logic_ci["ci_upper"],
    }

    logger.info(
        f"[aggregate] aggregated_n={len(all_results)} "
        f"bootstrap_samples={cfg['gsm8k_bootstrap_samples']} alpha={cfg['gsm8k_ci_alpha']}"
    )

    # Save merged results
    merged_results_file = output_dir / "results.json"
    with merged_results_file.open("w", encoding="utf-8") as f_out:
        json.dump(all_results_dict, f_out, indent=2)

    log_gsm8k_to_wandb(
        gsm8k_results=all_results_dict,
        model_path=cfg["model_path"],
        output_path=output_dir,
    )

    artifact = wandb.Artifact("eval", type="evaluation")
    artifact.add_file(str(merged_results_file))
    wandb.run.log_artifact(artifact)

    return all_results_dict
