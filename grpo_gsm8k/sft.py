from unittest.mock import patch

import torch
from transformers import PreTrainedModel
from vllm import LLM
from vllm.model_executor import set_random_seed as vllm_set_random_seed

from grpo_gsm8k.masked_normalize import masked_normalize


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass for a single microbatch.

    Args:
        policy_log_probs: (batch_size, sequence_length) per-token log-probabilities
            of the correct next token under the current SFT policy.
        response_mask: (batch_size, sequence_length) tensor with 1 for response tokens,
            0 for prompt/padding.
        gradient_accumulation_steps: number of microbatches per optimizer step;
            loss is scaled by 1 / gradient_accumulation_steps before backward().
        normalize_constant: constant by which to divide the masked sum (e.g., to turn
            a sum into an average). Default 1.0.

    Returns:
        loss: scalar tensor, scaled for gradient accumulation.
        metadata: dict of tensors with stats that may be useful for logging.
    """
    if policy_log_probs.shape != response_mask.shape:
        raise ValueError(
            f"response_mask shape {response_mask.shape} must match "
            f"policy_log_probs shape {policy_log_probs.shape}"
        )
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be a positive integer")

    # Cross-entropy over response tokens: -sum(log p) normalized as requested
    # Use masked_normalize to respect the mask and normalize by normalize_constant.
    nll: torch.Tensor = -masked_normalize(
        policy_log_probs, response_mask, normalize_constant, dim=None
    )  # scalar

    # Scale for gradient accumulation and backprop
    loss: torch.Tensor = nll / float(gradient_accumulation_steps)
    loss.backward()

    # Prepare metadata for logging (detached)
    mask_f = response_mask.to(policy_log_probs.dtype)
    token_count: torch.Tensor = mask_f.sum().detach()
    mean_log_prob: torch.Tensor = (
        (policy_log_probs * mask_f).sum() / token_count.clamp_min(1.0)
    ).detach()
    mean_nll: torch.Tensor = (-mean_log_prob).detach()

    metadata: dict[str, torch.Tensor] = {
        "nll_unscaled": nll.detach(),
        "loss": loss.detach(),
        "token_count": token_count,
        "mean_log_prob_response": mean_log_prob,
        "mean_nll_response": mean_nll,
    }

    return loss, metadata


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85) -> None:
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy
    """
    vllm_set_random_seed(seed)

    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM) -> None:
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())
