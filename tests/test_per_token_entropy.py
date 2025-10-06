import torch

from grpo_gsm8k.core.per_token_entropy import compute_entropy


def test_entropy_uniform_logits() -> None:
    bsz: int = 3
    seq: int = 5
    vocab: int = 7
    logits: torch.Tensor = torch.zeros(bsz, seq, vocab)
    ent: torch.Tensor = compute_entropy(logits)
    expected: torch.Tensor = torch.full(
        (bsz, seq), fill_value=torch.log(torch.tensor(vocab, dtype=logits.dtype))
    )
    assert ent.shape == (bsz, seq)
    assert torch.allclose(ent, expected, atol=1e-6)


def test_entropy_one_hot_like_logits() -> None:
    # Very confident predictions -> entropy ~ 0
    bsz: int = 2
    seq: int = 3
    vocab: int = 4
    logits: torch.Tensor = torch.full((bsz, seq, vocab), -1000.0)
    logits[..., 2] = 1000.0
    ent: torch.Tensor = compute_entropy(logits)
    assert torch.all(ent >= 0)
    assert torch.allclose(ent, torch.zeros(bsz, seq), atol=1e-5)


def test_entropy_invariant_to_additive_constant() -> None:
    bsz: int = 2
    seq: int = 4
    vocab: int = 5
    base: torch.Tensor = torch.randn(bsz, seq, vocab)
    c: torch.Tensor = torch.randn(bsz, seq, 1)  # broadcast additive constant per position
    ent1: torch.Tensor = compute_entropy(base)
    ent2: torch.Tensor = compute_entropy(base + c)
    assert torch.allclose(ent1, ent2, atol=1e-6)


def test_entropy_matches_manual_computation() -> None:
    # Compare against manual softmax-based entropy
    torch.manual_seed(0)
    bsz: int = 2
    seq: int = 2
    vocab: int = 3
    logits: torch.Tensor = torch.randn(bsz, seq, vocab)
    log_probs: torch.Tensor = torch.log_softmax(logits, dim=-1)
    probs: torch.Tensor = log_probs.exp()
    expected: torch.Tensor = -(probs * log_probs).sum(dim=-1)
    ent: torch.Tensor = compute_entropy(logits)
    assert torch.allclose(ent, expected, atol=1e-6)
