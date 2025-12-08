import math

import pytest
import torch

from grpo_gsm8k.training.policy_gradient import (
    compute_group_normalized_rewards,
)


def _stub_reward(gt: str, resp: str) -> dict[str, float]:  # noqa: ARG001
    # Reward equals numeric value parsed from resp; format/answer rewards split
    val = float(resp)
    return {
        "reward": val,
        "format_reward": val * 0.1,
        "answer_reward": val * 0.9,
    }


def test_group_normalization_by_std() -> None:
    # Two groups of size 3 each; explicit numbers
    group_size = 3
    rollout_responses = ["0", "1", "2", "10", "10", "10"]
    gts = ["_"] * len(rollout_responses)

    adv, raw, fmt, ans = compute_group_normalized_rewards(
        _stub_reward, rollout_responses, gts, group_size, advantage_eps=1e-6, normalize_by_std=True
    )

    assert adv.shape == torch.Size([6])
    assert raw.tolist() == [0.0, 1.0, 2.0, 10.0, 10.0, 10.0]
    assert fmt.tolist() == pytest.approx([0.0, 0.1, 0.2, 1.0, 1.0, 1.0], abs=1e-6)
    assert ans.tolist() == pytest.approx([0.0, 0.9, 1.8, 9.0, 9.0, 9.0], abs=1e-6)

    # First group: mean=1, std=sqrt(((1^2)+(0^2)+(1^2))/3)=sqrt(2/3)
    sd1 = math.sqrt(2.0 / 3.0)
    expected1 = [(-1.0) / sd1, 0.0, (1.0) / sd1]
    for x, y in zip(adv[:3].tolist(), expected1):
        assert abs(x - y) < 1e-5

    # Second group all equal -> std=0 -> denom=advantage_eps, so (x-mu)/eps == 0
    for x in adv[3:].tolist():
        assert abs(x - 0.0) < 1e-8


def test_group_normalization_by_mean_only() -> None:
    group_size = 2
    rollout_responses = ["5", "7", "1", "-1"]
    gts = ["_"] * len(rollout_responses)

    adv, raw, fmt, ans = compute_group_normalized_rewards(
        _stub_reward, rollout_responses, gts, group_size, normalize_by_std=False
    )

    # Check mean subtraction per group
    # Group1: mu=6 -> [-1, +1]; Group2: mu=0 -> [1, -1]
    assert adv.tolist() == [-1.0, 1.0, 1.0, -1.0]
    assert raw.tolist() == [5.0, 7.0, 1.0, -1.0]
    assert fmt.tolist() == pytest.approx([0.5, 0.7, 0.1, -0.1], abs=1e-6)
    assert ans.tolist() == pytest.approx([4.5, 6.3, 0.9, -0.9], abs=1e-6)
