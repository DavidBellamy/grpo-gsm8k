import torch
from transformers import PreTrainedModel

from grpo_gsm8k.per_token_entropy import compute_entropy


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Compute per-token conditional log-probabilities log p(x_t | x_<t) for the given labels,
    and optionally the per-token entropy of the model's next-token distribution.

    Args:
        model: HuggingFace causal LM used for scoring.
        input_ids: Tensor of shape (batch_size, sequence_length), typically ids[:-1].
        labels: Tensor of shape (batch_size, sequence_length), typically ids[1:].
        return_token_entropy: If True, also return per-token entropy over the vocab.

    Returns:
        {
            "log_probs": (batch_size, sequence_length),
            "token_entropy": (batch_size, sequence_length)  # only if requested
        }
    """
    # Inference mode for efficiency/numerical parity with evaluation.
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits  # (B, T, V)

        log_probs_all = torch.log_softmax(logits, dim=-1)  # (B, T, V)
        # Gather the log-prob of the actual next token at each position.
        log_probs = log_probs_all.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # (B, T)

        result: dict[str, torch.Tensor] = {"log_probs": log_probs}

        if return_token_entropy:
            result["token_entropy"] = compute_entropy(logits)  # (B, T)

    return result
