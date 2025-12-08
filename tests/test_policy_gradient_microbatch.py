import torch

from grpo_gsm8k.training.policy_gradient import policy_gradient_microbatch_train_step


def test_microbatch_reinforce_with_baseline_episode_drop_and_metadata() -> None:
    # Two examples, T=4; one has zero-length response_mask → should be dropped
    policy_logp = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], requires_grad=True)
    response_mask = torch.tensor([[0, 0, 0, 0], [0, 1, 1, 0]])
    advantages = torch.tensor([[1.0], [2.0]])

    loss, meta = policy_gradient_microbatch_train_step(
        policy_logp,
        response_mask,
        episodes_per_update=4,
        loss_type="reinforce_with_baseline",
        advantages=advantages,
    )

    assert loss.ndim == 0
    assert int(meta["episodes_used"].item()) == 1  # one row contributed
    assert int(meta["episodes_dropped_zero_len"].item()) == 1
    assert int(meta["tokens"].item()) == int(response_mask[1].sum().item())
    assert "mean_neg_logp" in meta and "mean_seq_len" in meta
    assert torch.allclose(meta["mean_advantage"], torch.tensor(2.0))

    # Grad should exist on input
    assert policy_logp.grad is not None


def test_microbatch_no_baseline_scaling_by_episodes_per_update() -> None:
    # Two valid episodes; episodes_per_update larger to test rescaling in caller
    policy_logp = torch.tensor([[0.1, 0.0], [0.2, -0.2]], requires_grad=True)
    response_mask = torch.tensor([[0, 1], [1, 1]])
    raw_rewards = torch.tensor([[1.0], [2.0]])

    loss, meta = policy_gradient_microbatch_train_step(
        policy_logp,
        response_mask,
        episodes_per_update=8,
        loss_type="no_baseline",
        raw_rewards=raw_rewards,
    )
    assert loss.ndim == 0
    # Check mean_neg_logp computed only over masked tokens
    masked = response_mask.bool()
    mean_neg = -(policy_logp[masked]).mean()
    assert torch.allclose(meta["mean_neg_logp"], mean_neg)


def test_microbatch_grpo_clip_metadata_flags_present() -> None:
    policy_logp = torch.tensor([[0.2, -0.2]], requires_grad=True)
    response_mask = torch.tensor([[1, 1]])
    advantages = torch.tensor([[1.0]])
    old_logp = torch.tensor([[0.0, 0.0]])

    loss, meta = policy_gradient_microbatch_train_step(
        policy_logp,
        response_mask,
        episodes_per_update=2,
        loss_type="grpo_clip",
        advantages=advantages,
        old_log_probs=old_logp,
        cliprange=0.2,
    )
    assert loss.ndim == 0
    for k in ["importance_wts", "clipped_wts", "was_clipped", "min_is_rhs"]:
        assert k in meta
