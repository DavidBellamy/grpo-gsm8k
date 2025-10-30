import math
import statistics as stats
from collections.abc import Callable

import torch


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float = 1e-8,
    normalize_by_std: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    assert len(rollout_responses) == len(repeated_ground_truths)
    n = len(rollout_responses)
    assert group_size > 0 and n % group_size == 0

    # Score
    raw_list: list[float] = []
    fmt_list: list[float] = []
    ans_list: list[float] = []
    for gt, resp in zip(repeated_ground_truths, rollout_responses):
        out = reward_fn(gt, resp)
        raw_list.append(float(out["reward"]))
        fmt_list.append(float(out["format_reward"]))
        ans_list.append(float(out["answer_reward"]))

    # Group-normalize
    adv_list = [0.0] * n
    num_groups = n // group_size
    for g in range(num_groups):
        start = g * group_size
        end = start + group_size
        grp = raw_list[start:end]
        mu = stats.mean(grp)
        if normalize_by_std:
            sd = stats.pstdev(grp)
            denom = sd if sd > advantage_eps else advantage_eps
            adv_list[start:end] = [(x - mu) / denom for x in grp]
        else:
            adv_list[start:end] = [x - mu for x in grp]

    # Metadata
    def safe_mean(xs: list[float]) -> float:
        return stats.mean(xs) if xs else math.nan

    def safe_pstd(xs: list[float]) -> float:
        return stats.pstdev(xs) if xs else math.nan

    metadata: dict[str, float] = {
        "reward_mean": safe_mean(raw_list),
        "reward_std": safe_pstd(raw_list),
        "reward_min": min(raw_list) if raw_list else math.nan,
        "reward_max": max(raw_list) if raw_list else math.nan,
        "group_count": float(num_groups),
        "group_size": float(group_size),
        "format_reward_mean": safe_mean(fmt_list),
        "format_reward_std": safe_pstd(fmt_list),
        "answer_reward_mean": safe_mean(ans_list),
        "answer_reward_std": safe_pstd(ans_list),
    }

    advantages = torch.tensor(adv_list, dtype=torch.float32)
    raw_rewards = torch.tensor(raw_list, dtype=torch.float32)
    return advantages, raw_rewards, metadata
