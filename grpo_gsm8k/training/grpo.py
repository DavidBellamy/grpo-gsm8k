import math
import statistics as stats
from collections.abc import Callable
from typing import Literal

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


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the policy-gradient loss at every token, where raw_rewards_or_advantages is either
    the raw reward or an already-normalized advantage.
    """
    return -raw_rewards_or_advantages * policy_log_probs


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    importance_wts = torch.exp(policy_log_probs - old_log_probs)
    clipped_wts = torch.clamp(importance_wts, 1.0 - cliprange, 1.0 + cliprange)

    lhs = importance_wts * advantages
    rhs = clipped_wts * advantages
    metadata = {
        "importance_wts": importance_wts.detach(),
        "clipped_wts": clipped_wts.detach(),
        "was_clipped": (importance_wts.ne(clipped_wts)).detach().to(torch.int),
        "min_is_rhs": (rhs < lhs).detach().to(torch.int),
    }

    return -torch.min(lhs, rhs), metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    metadata: dict[str, torch.Tensor] = {}

    assert loss_type in ["no_baseline", "reinforce_with_baseline", "grpo_clip"]

    if loss_type == "no_baseline":
        if raw_rewards is None:
            raise ValueError("raw_rewards required for loss_type='no_baseline'")
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
    elif loss_type == "reinforce_with_baseline":
        if advantages is None:
            raise ValueError("advantages required for loss_type='reinforce_with_baseline'")
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
    else:  # grpo_clip
        if advantages is None or old_log_probs is None or cliprange is None:
            raise ValueError("advantages, old_log_probs, cliprange required for 'grpo_clip'")
        loss, metadata = compute_grpo_clip_loss(
            advantages, policy_log_probs, old_log_probs, cliprange
        )

    return loss, metadata


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    wt_sum = (tensor * mask).sum(dim=dim)
    count = mask.sum(dim=dim)
    assert torch.all(count > 0)
    return wt_sum / count
