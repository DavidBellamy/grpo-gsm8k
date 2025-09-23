import pytest
import torch

from grpo_gsm8k.masked_normalize import masked_normalize


def test_masked_normalize_global_sum_scalar() -> None:
    tensor: torch.Tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    mask: torch.Tensor = torch.tensor([[1, 0], [0, 1]])  # int mask OK
    normalize_constant: float = 2.0

    out: torch.Tensor = masked_normalize(tensor, mask, normalize_constant, dim=None)

    # Sum of masked elements: 1 + 4 = 5; divide by 2 -> 2.5
    assert out.shape == ()
    assert torch.allclose(out, torch.tensor(2.5))


def test_masked_normalize_sum_along_dim() -> None:
    tensor: torch.Tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mask: torch.Tensor = torch.tensor([[1, 0, 1], [0, 1, 1]])
    normalize_constant: float = 2.0

    out: torch.Tensor = masked_normalize(tensor, mask, normalize_constant, dim=1)
    # Row 0: (1 + 3) / 2 = 2.0; Row 1: (5 + 6) / 2 = 5.5
    expected: torch.Tensor = torch.tensor([2.0, 5.5])
    assert out.shape == (2,)
    assert torch.allclose(out, expected)


def test_bool_mask_and_gradients() -> None:
    tensor: torch.Tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    mask: torch.Tensor = torch.tensor([[True, False], [True, True]])
    normalize_constant: float = 3.0

    out: torch.Tensor = masked_normalize(tensor, mask, normalize_constant, dim=None)
    out.backward()  # d/dx of sum(mask * x)/C -> mask/C

    expected_grad: torch.Tensor = mask.to(dtype=tensor.dtype) / normalize_constant
    assert torch.allclose(tensor.grad, expected_grad)


def test_dim_reduction_shape_and_values_3d() -> None:
    B, T, K = 2, 3, 4
    tensor: torch.Tensor = torch.arange(B * T * K, dtype=torch.float32).reshape(B, T, K)
    # Only keep last two elements along K
    mask: torch.Tensor = torch.zeros_like(tensor)
    mask[..., 2:] = 1.0
    const: float = 2.0

    out: torch.Tensor = masked_normalize(tensor, mask, const, dim=2)
    # Expect shape (B, T) and value = average of last two elements along K
    expected: torch.Tensor = tensor[..., 2:].mean(dim=2)
    assert out.shape == (B, T)
    assert torch.allclose(out, expected)


def test_raises_on_shape_mismatch() -> None:
    tensor: torch.Tensor = torch.randn(2, 3)
    mask: torch.Tensor = torch.ones(2, 2)  # wrong shape
    with pytest.raises(ValueError):
        _ = masked_normalize(tensor, mask, 1.0, dim=1)
