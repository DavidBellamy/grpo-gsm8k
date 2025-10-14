import torch

from grpo_gsm8k.core.get_response_log_probs import get_response_log_probs
from grpo_gsm8k.core.per_token_entropy import compute_entropy


class ModelOut:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits: torch.Tensor = logits


class DummyModel(torch.nn.Module):
    """
    Minimal causal-LM-like stub. Given input_ids (B, T), returns logits (B, T, V)
    where the preferred class at position (b, t) is input_ids[b, t] % V.
    """

    def __init__(self, vocab_size: int, hi: float = 5.0, lo: float = -5.0) -> None:
        super().__init__()
        self.vocab_size: int = vocab_size
        self.hi: float = hi
        self.lo: float = lo

    def forward(self, input_ids: torch.Tensor) -> ModelOut:
        B: int
        T: int
        B, T = input_ids.shape
        V: int = self.vocab_size
        logits: torch.Tensor = torch.full(
            (B, T, V), self.lo, dtype=torch.float32, device=input_ids.device
        )
        top_idx: torch.Tensor = (input_ids % V).unsqueeze(-1).long()  # (B, T, 1)
        logits.scatter_(-1, top_idx, self.hi)
        return ModelOut(logits)


def test_matches_manual_gather_and_entropy() -> None:
    torch.manual_seed(0)
    B: int = 2
    T: int = 3
    V: int = 7
    model: DummyModel = DummyModel(V)

    input_ids: torch.Tensor = torch.randint(0, 100, (B, T), dtype=torch.long)
    labels: torch.Tensor = torch.randint(0, V, (B, T), dtype=torch.long)

    res: dict[str, torch.Tensor] = get_response_log_probs(
        model, input_ids, labels, return_token_entropy=True
    )

    # Manual expectations
    logits: torch.Tensor = model(input_ids).logits
    all_log_probs: torch.Tensor = torch.log_softmax(logits, dim=-1)
    expected_log_probs: torch.Tensor = all_log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    expected_entropy: torch.Tensor = compute_entropy(logits)

    assert res["log_probs"].shape == (B, T)
    assert torch.allclose(res["log_probs"], expected_log_probs, atol=1e-6)

    assert "token_entropy" in res
    assert res["token_entropy"].shape == (B, T)
    assert torch.allclose(res["token_entropy"], expected_entropy, atol=1e-6)


def test_no_entropy_flag_and_no_grad() -> None:
    B: int = 1
    T: int = 4
    V: int = 5
    model: DummyModel = DummyModel(V)

    input_ids: torch.Tensor = torch.arange(B * T, dtype=torch.long).reshape(B, T)
    labels: torch.Tensor = torch.randint(0, V, (B, T), dtype=torch.long)

    res: dict[str, torch.Tensor] = get_response_log_probs(
        model, input_ids, labels, return_token_entropy=False
    )

    assert set(res.keys()) == {"log_probs"}
    assert res["log_probs"].shape == (B, T)
    # Should be computed under torch.no_grad()
    assert res["log_probs"].requires_grad is False
