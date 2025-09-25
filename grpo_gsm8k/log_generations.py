from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from grpo_gsm8k.per_token_entropy import compute_entropy

RewardFn = Callable[[str | None], dict[str, Any]]


def _model_device(model: PreTrainedModel) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def log_generations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    references: list[str] | None = None,
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 1.0,
    reward_fn: RewardFn | None = None,
) -> dict[str, Any]:
    """
    Generate responses for prompts and compute simple logging stats.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Matching tokenizer.
        prompts: List of prompt strings.
        references: Optional list of reference/ground-truth answers (same length as prompts).
        max_new_tokens: Max tokens to generate for each prompt.
        temperature: Sampling temperature. If 0.0, greedy.
        top_p: Nucleus sampling parameter (only used if sampling).
        reward_fn: Optional callback (prompt, response, reference) -> dict with reward info.
                   If present, we include its result per example and use `is_correct` if provided
                   to compute separate average response lengths for correct/incorrect subsets.

    Returns:
        dict with:
            - prompts: list[str]
            - responses: list[str]
            - references: list[str] (empty strings if not provided)
            - rewards: list[dict] (empty dicts if reward_fn is None)
            - avg_token_entropy: float (mean over all response tokens)
            - avg_response_length: float (mean number of generated tokens)
            - avg_response_length_correct: float | None
            - avg_response_length_incorrect: float | None
            - per_example: list[dict] with fields:
                prompt, response, reference, length, mean_token_entropy, reward (dict)
    """
    if references is not None and len(references) != len(prompts):
        raise ValueError("references must be None or have the same length as prompts")

    device = _model_device(model)
    model.eval()

    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)
    input_lens: list[int] = attention_mask.sum(dim=1).tolist()

    with torch.no_grad():
        generate_kwargs: dict[str, Any] = dict(
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
        )
        # Enable sampling only if temperature > 0
        if temperature and temperature > 0.0:
            generate_kwargs.update(
                dict(do_sample=True, temperature=float(temperature), top_p=float(top_p))
            )
        out = model.generate(input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs)

    sequences: torch.Tensor = out.sequences  # (B, prompt+gen)
    batch_size: int = sequences.size(0)

    # Decode generated tokens (exclude the prompt)
    responses: list[str] = []
    resp_token_ids: list[list[int]] = []
    for i in range(batch_size):
        start = int(input_lens[i])
        gen_ids_i = sequences[i, start:].tolist()
        resp_token_ids.append(gen_ids_i)
        responses.append(tokenizer.decode(gen_ids_i, skip_special_tokens=True))

    # Compute per-token entropy from generation scores.
    # out.scores is a list of length T_gen with tensors of shape (B, V).
    if len(out.scores) > 0:
        logits_bt = torch.stack(out.scores, dim=1)  # (B, T_gen, V)
        ent_bt = compute_entropy(logits_bt).to(torch.float32)  # (B, T_gen)
    else:
        ent_bt = torch.zeros((batch_size, 0), dtype=torch.float32, device=device)

    # Determine actual generated length per example:
    # count non-pad tokens in the generated slice, but never exceed the
    # number of score steps we have (handles padding artifacts).
    scores_T = ent_bt.size(1)
    pad_id = getattr(tokenizer, "pad_token_id", 0)
    lengths: list[int] = []
    for ids in resp_token_ids:
        if scores_T == 0:
            lengths.append(0)
        else:
            if pad_id is None:
                pad_id = 0
            L = sum(1 for x in ids[:scores_T] if x != pad_id)
            # If tokenizer has no pad token and zeros are meaningful, fall back to scores_T
            if L == 0 and scores_T > 0 and pad_id == 0:
                L = min(len(ids), scores_T)
            lengths.append(L)

    # Mask only the actually generated tokens per example for averaging entropies
    max_gen = ent_bt.size(1)
    mask_bt = torch.zeros((batch_size, max_gen), device=device, dtype=ent_bt.dtype)
    for i, L in enumerate(lengths):
        if L > 0:
            mask_bt[i, : min(L, max_gen)] = 1.0

    denom = mask_bt.sum().clamp_min(1.0)
    avg_token_entropy: float = ((ent_bt * mask_bt).sum() / denom).item()

    # Average response length in tokens
    avg_response_length: float = (
        torch.tensor(lengths, dtype=torch.float32).mean().item() if lengths else 0.0
    )

    references_filled: list[str] = references if references is not None else [""] * batch_size
    rewards: list[dict[str, Any]] = []
    is_correct_flags: list[bool] = []
    for p, r, ref in zip(prompts, responses, references_filled):
        if reward_fn is not None:
            reward_info = reward_fn(r or None)
        else:
            reward_info = {}
            if references is not None:
                reward_info["is_correct"] = r.strip() == ref.strip()
        rewards.append(reward_info)
        is_correct_flags.append(bool(reward_info.get("is_correct", False)))

    # Length stats by correctness if any correctness is available
    if any(is_correct_flags) or (
        references is not None and any(ref != "" for ref in references_filled)
    ):
        lengths_tensor = torch.tensor(lengths, dtype=torch.float32)
        correct_mask = torch.tensor(is_correct_flags, dtype=torch.bool)
        avg_len_correct = lengths_tensor[correct_mask].mean().item() if correct_mask.any() else None
        avg_len_incorrect = (
            lengths_tensor[~correct_mask].mean().item() if (~correct_mask).any() else None
        )
    else:
        avg_len_correct = None
        avg_len_incorrect = None

    per_example: list[dict[str, Any]] = []
    for i in range(batch_size):
        L = lengths[i]
        ent_mean_i = ent_bt[i, : min(L, max_gen)].mean().item() if L > 0 and max_gen > 0 else 0.0
        per_example.append(
            {
                "prompt": prompts[i],
                "response": responses[i],
                "reference": references_filled[i],
                "length": L,
                "mean_token_entropy": ent_mean_i,
                "reward": rewards[i],
            }
        )

    return {
        "prompts": prompts,
        "responses": responses,
        "references": references_filled,
        "rewards": rewards,
        "avg_token_entropy": avg_token_entropy,
        "avg_response_length": avg_response_length,
        "avg_response_length_correct": avg_len_correct,
        "avg_response_length_incorrect": avg_len_incorrect,
        "per_example": per_example,
    }
