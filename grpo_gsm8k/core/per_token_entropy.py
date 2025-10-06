import torch


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute per-token entropy over the vocabulary dimension.

    Args:
        logits: Tensor of shape (batch_size, seq_len, vocab_size)
                containing unnormalized logits.

    Returns:
        Tensor of shape (batch_size, seq_len) with entropy values.
    """
    # log_softmax for numerical stability
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy
