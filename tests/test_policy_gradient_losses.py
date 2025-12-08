import pytest
import torch

from grpo_gsm8k.training.policy_gradient import (
    compute_grpo_clip_loss,
    compute_naive_policy_gradient_loss,
    compute_policy_gradient_loss,
)


def test_compute_naive_policy_gradient_loss_shapes_and_values() -> None:
    rewards = torch.tensor([[1.0, 0.5], [-1.0, 2.0]])
    logp = torch.tensor([[0.2, -0.4], [0.0, 0.3]])
    loss = compute_naive_policy_gradient_loss(rewards, logp)
    expected = -rewards * logp
    assert loss.shape == logp.shape
    assert torch.allclose(loss, expected)


def test_compute_grpo_clip_loss_clipping_and_metadata() -> None:
    advantages = torch.tensor([[1.0, 1.0]])
    old_logp = torch.tensor([[0.0, 0.0]])
    # policy_logp creates weights: exp(delta)
    policy_logp = torch.tensor([[0.3, -0.25]])  # weights ~ [1.3499, 0.7788]
    cliprange = 0.2

    loss, meta = compute_grpo_clip_loss(advantages, policy_logp, old_logp, cliprange)
    # Compute expected
    w = torch.exp(policy_logp - old_logp)
    cw = torch.clamp(w, 1.0 - cliprange, 1.0 + cliprange)
    lhs = w * advantages
    rhs = cw * advantages
    expected = -torch.min(lhs, rhs)

    assert torch.allclose(loss, expected)
    assert "importance_wts" in meta and "clipped_wts" in meta
    assert meta["importance_wts"].shape == w.shape
    assert meta["was_clipped"].dtype == torch.int
    # First weight clipped to 1.2? 1.3499 -> 1.2; second to 0.8? 0.7788 -> remains 0.8 lower bound
    assert torch.allclose(meta["clipped_wts"], cw)
    assert torch.equal(meta["was_clipped"], (w.ne(cw)).to(torch.int))


def test_compute_policy_gradient_loss_dispatch_and_errors() -> None:
    logp = torch.tensor([[0.1, 0.2]])
    rewards = torch.tensor([[1.0]])  # broadcastable if needed
    adv = torch.tensor([[0.5]])
    old_logp = torch.tensor([[0.1, 0.2]])

    # no_baseline
    loss, meta = compute_policy_gradient_loss(logp, "no_baseline", raw_rewards=rewards)
    assert meta == {}
    assert torch.allclose(loss, -rewards * logp)

    # reinforce_with_baseline
    loss2, meta2 = compute_policy_gradient_loss(logp, "reinforce_with_baseline", advantages=adv)
    assert meta2 == {}
    assert torch.allclose(loss2, -adv * logp)

    # grpo_clip
    loss3, meta3 = compute_policy_gradient_loss(
        logp, "grpo_clip", advantages=adv, old_log_probs=old_logp, cliprange=0.2
    )
    assert "importance_wts" in meta3

    # error paths
    with pytest.raises(ValueError):
        compute_policy_gradient_loss(logp, "no_baseline")
    with pytest.raises(ValueError):
        compute_policy_gradient_loss(logp, "reinforce_with_baseline")
    with pytest.raises(ValueError):
        compute_policy_gradient_loss(logp, "grpo_clip", advantages=adv, old_log_probs=old_logp)
