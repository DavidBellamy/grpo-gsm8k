import pytest
import torch

from grpo_gsm8k.training.sft import sft_microbatch_train_step


def test_microbatch_step_gradients_and_loss_values() -> None:
    # Setup a small example
    policy_log_probs: torch.Tensor = torch.tensor(
        [
            [-0.1, -1.0, -0.2],
            [-0.3, -2.0, -0.4],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    # Include positions (0,0), (0,2), (1,1), (1,2)
    response_mask: torch.Tensor = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )

    # Run step (performs backward internally)
    loss: torch.Tensor
    loss, _ = sft_microbatch_train_step(
        policy_log_probs,
        response_mask,
    )

    # Expected values
    included_sum: float = (
        policy_log_probs.detach()[0, 0]
        + policy_log_probs.detach()[0, 2]
        + policy_log_probs.detach()[1, 1]
        + policy_log_probs.detach()[1, 2]
    ).item()
    expected_loss: float = -included_sum

    # Check returned loss
    assert loss.shape == ()
    assert pytest.approx(loss.item(), rel=1e-6, abs=1e-6) == expected_loss

    # Gradients: d(loss)/d(x) = -mask
    expected_grad: torch.Tensor = -response_mask
    assert torch.allclose(policy_log_probs.grad, expected_grad, atol=1e-7)


def test_microbatch_step_gradients_comprehensive() -> None:
    policy_log_probs = torch.tensor(
        [[-0.1, -1.0, -0.2], [-0.3, -2.0, -0.4]],
        dtype=torch.float32,
        requires_grad=True,
    )
    response_mask = torch.tensor(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
        dtype=torch.float32,
    )

    loss, _ = sft_microbatch_train_step(policy_log_probs, response_mask)
    expected_grad = -response_mask
    assert torch.allclose(policy_log_probs.grad, expected_grad, atol=1e-7)

    # Test 2: All zeros mask (should have zero gradients)
    policy_log_probs.grad = None  # Reset gradients
    zero_mask = torch.zeros_like(response_mask)
    loss, _ = sft_microbatch_train_step(policy_log_probs, zero_mask)
    assert torch.allclose(policy_log_probs.grad, torch.zeros_like(policy_log_probs))

    # Test 3: All ones mask (should have -1 everywhere)
    policy_log_probs.grad = None
    ones_mask = torch.ones_like(response_mask)
    loss, _ = sft_microbatch_train_step(policy_log_probs, ones_mask)
    assert torch.allclose(policy_log_probs.grad, -ones_mask)


def test_metadata_contents_and_types() -> None:
    x: torch.Tensor = torch.tensor([[0.0, -1.0], [-2.0, -3.0]], requires_grad=True)
    mask: torch.Tensor = torch.tensor([[True, False], [True, True]])  # boolean mask
    loss, meta = sft_microbatch_train_step(x, mask)

    # Token count and means
    token_count: float = mask.to(x.dtype).sum().item()
    included_sum: float = (x.detach() * mask.to(x.dtype)).sum().item()
    mean_lp: float = included_sum / token_count
    mean_nll: float = -mean_lp

    assert isinstance(meta, dict)
    assert set(meta.keys()) == {
        "nll_unscaled",
        "loss",
        "token_count",
        "avg_response_token_logprob",
        "avg_response_token_nll",
    }

    assert meta["nll_unscaled"].shape == ()
    assert meta["loss"].shape == ()
    assert meta["token_count"].shape == ()
    assert pytest.approx(meta["nll_unscaled"].item(), rel=1e-6) == -included_sum
    assert pytest.approx(meta["loss"].item(), rel=1e-6) == (-included_sum)
    assert pytest.approx(meta["token_count"].item(), rel=1e-6) == token_count
    assert pytest.approx(meta["avg_response_token_logprob"].item(), rel=1e-6) == mean_lp
    assert pytest.approx(meta["avg_response_token_nll"].item(), rel=1e-6) == mean_nll

    # Returned loss is already used for backward in the function, but should still require grad.
    assert loss.requires_grad


def test_errors_on_wrong_mask_shape() -> None:
    x: torch.Tensor = torch.randn(2, 3, requires_grad=True)
    mask_wrong: torch.Tensor = torch.ones(2, 2)
    with pytest.raises(ValueError):
        _ = sft_microbatch_train_step(x, mask_wrong)
