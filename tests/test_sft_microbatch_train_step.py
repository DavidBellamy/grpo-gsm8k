import pytest
import torch

from grpo_gsm8k.sft import sft_microbatch_train_step


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
    normalize_constant: float = 2.0
    grad_accum_steps: int = 4

    # Run step (performs backward internally)
    loss: torch.Tensor
    metadata: dict[str, torch.Tensor]
    loss, metadata = sft_microbatch_train_step(
        policy_log_probs,
        response_mask,
        gradient_accumulation_steps=grad_accum_steps,
        normalize_constant=normalize_constant,
    )

    # Expected values
    included_sum: float = (
        policy_log_probs.detach()[0, 0]
        + policy_log_probs.detach()[0, 2]
        + policy_log_probs.detach()[1, 1]
        + policy_log_probs.detach()[1, 2]
    ).item()
    expected_nll: float = -included_sum / normalize_constant
    expected_loss: float = expected_nll / grad_accum_steps

    # Check returned loss
    assert loss.shape == ()
    assert pytest.approx(loss.item(), rel=1e-6, abs=1e-6) == expected_loss

    # Gradients: d(loss)/d(x) = -mask / (normalize_constant * grad_accum_steps)
    expected_grad: torch.Tensor = -response_mask / (normalize_constant * grad_accum_steps)
    assert torch.allclose(policy_log_probs.grad, expected_grad, atol=1e-7)


def test_metadata_contents_and_types() -> None:
    x: torch.Tensor = torch.tensor([[0.0, -1.0], [-2.0, -3.0]], requires_grad=True)
    mask: torch.Tensor = torch.tensor([[True, False], [True, True]])  # boolean mask
    ga: int = 2
    C: float = 1.0

    loss, meta = sft_microbatch_train_step(x, mask, ga, normalize_constant=C)

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
        "mean_log_prob_response",
        "mean_nll_response",
    }

    assert meta["nll_unscaled"].shape == ()
    assert meta["loss"].shape == ()
    assert meta["token_count"].shape == ()
    assert pytest.approx(meta["nll_unscaled"].item(), rel=1e-6) == -included_sum / C
    assert pytest.approx(meta["loss"].item(), rel=1e-6) == (-included_sum / C) / ga
    assert pytest.approx(meta["token_count"].item(), rel=1e-6) == token_count
    assert pytest.approx(meta["mean_log_prob_response"].item(), rel=1e-6) == mean_lp
    assert pytest.approx(meta["mean_nll_response"].item(), rel=1e-6) == mean_nll

    # Returned loss is already used for backward in the function, but should still require grad.
    assert loss.requires_grad


def test_errors_on_bad_inputs() -> None:
    x: torch.Tensor = torch.randn(2, 3, requires_grad=True)
    mask_wrong: torch.Tensor = torch.ones(2, 2)
    with pytest.raises(ValueError):
        _ = sft_microbatch_train_step(x, mask_wrong, gradient_accumulation_steps=1)

    mask_ok: torch.Tensor = torch.ones_like(x)
    with pytest.raises(ValueError):
        _ = sft_microbatch_train_step(x, mask_ok, gradient_accumulation_steps=0)
