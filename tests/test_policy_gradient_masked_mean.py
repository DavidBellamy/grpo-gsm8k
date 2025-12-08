import pytest
import torch

from grpo_gsm8k.training.policy_gradient import masked_mean


def test_masked_mean_rowwise() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0], [10.0, 0.0, -10.0]])
    m = torch.tensor([[1, 0, 1], [0, 1, 1]])
    out = masked_mean(x, m, dim=1)
    # Row1: (1+3)/2 = 2; Row2: (0-10)/2 = -5
    assert torch.allclose(out, torch.tensor([2.0, -5.0]))


def test_masked_mean_elementwise_and_global() -> None:
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    m = torch.tensor([[1, 1], [0, 1]])
    # No dim => sums entire tensor
    out = masked_mean(x, m, dim=None)
    # (1+2+4) / 3 = 7/3
    assert torch.allclose(out, torch.tensor(7.0 / 3.0))


def test_masked_mean_zero_count_assertion() -> None:
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    m = torch.tensor([[0, 0], [1, 0]])
    with pytest.raises(AssertionError):
        masked_mean(x, m, dim=1)
