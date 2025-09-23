from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

from grpo_gsm8k.get_response_log_probs import get_response_log_probs
from grpo_gsm8k.log_generations import log_generations
from grpo_gsm8k.masked_normalize import masked_normalize
from grpo_gsm8k.per_token_entropy import compute_entropy
from grpo_gsm8k.prompts import render_batch
from grpo_gsm8k.tokenize import tokenize_prompt_and_output

logger = logging.getLogger(__name__)


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


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85) -> LLM:
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


# -----------------------------
# Simple SFT utilities
# -----------------------------


def _ensure_pad_token(tokenizer: PreTrainedTokenizer) -> None:
    if tokenizer.pad_token_id is None:
        # Fall back to EOS or create a new pad token id
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})


def _load_jsonl_pairs(path: str | Path) -> list[dict[str, str]]:
    """
    Load a JSONL file with records containing keys "prompt" and "response".
    Returns a list of dicts with those two keys as strings.
    """
    recs: list[dict[str, str]] = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt = obj.get("prompt", "")
            response = obj.get("response", "")
            if not isinstance(prompt, str):
                prompt = str(prompt)
            if not isinstance(response, str):
                response = str(response)
            recs.append({"prompt": prompt, "response": response})
    return recs


def _build_qwen_chat_prompts(tokenizer: PreTrainedTokenizer, prompts_raw: list[str]) -> list[str]:
    """
    Build Qwen-style chat prompts from raw questions using prompts.render_batch.
    """
    # Pass add_generation_prompt as a positional arg for compatibility with tests
    return render_batch(tokenizer, prompts_raw, True)


def _resolve_resume_path(path: str | Path) -> Path:
    """
    If `path` is a checkpoint root containing step subdirs, pick the latest.
    Otherwise, return `path` as-is.
    """
    p = Path(path)
    # If it already looks like a model dir (e.g., has config.json), use it directly
    if (p / "config.json").exists():
        return p
    # Prefer step_XXXX subdirectories
    step_dirs: list[tuple[int, Path]] = []
    if p.is_dir():
        for child in p.iterdir():
            if child.is_dir() and child.name.startswith("step_"):
                try:
                    step_num = int(child.name.split("step_")[-1])
                    step_dirs.append((step_num, child))
                except ValueError:
                    continue
    if step_dirs:
        step_dirs.sort(key=lambda x: x[0], reverse=True)
        return step_dirs[0][1]
    return p


@torch.no_grad()
def _sample_eval(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    questions: list[str],
    *,
    max_new_tokens: int = 128,
) -> dict[str, Any]:
    """
    Generate responses for quick sanity checks during training. Keeps it lightweight.
    """
    model.eval()
    chat_prompts = _build_qwen_chat_prompts(tokenizer, questions)
    enc = tokenizer(chat_prompts, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"].to(next(model.parameters()).device)
    attn = enc.get("attention_mask", torch.ones_like(input_ids)).to(input_ids.device)
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_scores=True,
    )
    seqs = out.sequences
    lens = attn.sum(-1).tolist()
    responses: list[str] = []
    for i in range(seqs.size(0)):
        gen_ids = seqs[i, int(lens[i]) :].tolist()
        responses.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
    # Entropy over generated tokens
    if len(out.scores) > 0:
        logits_bt = torch.stack(out.scores, dim=1)
        ent_bt = compute_entropy(logits_bt)
        avg_ent = ent_bt.mean().item()
    else:
        avg_ent = 0.0
    return {"prompts": questions, "responses": responses, "avg_token_entropy": avg_ent}


def _vllm_generate(
    llm: LLM,
    prompts: list[str],
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> dict[str, Any]:
    """
    Generate with vLLM for a list of prompts. Returns responses and simple stats.
    """
    samp = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=float(temperature),
        top_p=float(top_p),
    )
    outs = llm.generate(prompts, samp)
    responses: list[str] = []
    lengths: list[int] = []
    for out in outs:
        # vLLM returns a list of candidates; take the first
        if not out.outputs:
            responses.append("")
            lengths.append(0)
            continue
        text = out.outputs[0].text or ""
        responses.append(text)
        # tokens field may be available; approximate length from token ids if present
        token_ids = getattr(out.outputs[0], "token_ids", None)
        lengths.append(len(token_ids) if token_ids is not None else len(text.split()))
    avg_len = float(torch.tensor(lengths, dtype=torch.float32).mean().item()) if lengths else 0.0
    return {"responses": responses, "avg_response_length": avg_len}


def train_sft_on_r1_pairs(
    data_path: str | Path,
    *,
    model_id: str = "Qwen/Qwen2.5-Math-1.5B-Instruct",
    device: str | torch.device | None = None,
    microbatch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    num_epochs: int = 1,
    max_steps: int | None = None,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.0,
    max_total_tokens: int | None = 2048,
    log_every: int = 10,
    eval_every: int = 200,
    eval_examples: int = 4,
    dtype: torch.dtype | None = None,
    # vLLM eval on a second GPU
    do_vllm_eval: bool = True,
    vllm_device: str | None = "cuda:1",
    vllm_gpu_memory_utilization: float = 0.85,
    # Optional W&B logging callback: wb_log(step, metrics)
    wb_log: Callable[[int, dict[str, float]], None] | None = None,
    # Optional: checkpointing and eval callback
    checkpoint_dir: str | Path | None = None,
    checkpoint_every: int | None = None,
    on_eval: Callable[[int, dict[str, Any]], None] | None = None,
    # Optional: resume from a previous checkpoint directory
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    """
    Run a simple SFT loop on a JSONL dataset with keys {"prompt","response"},
    building Qwen-style chat prompts for inputs. Uses gradient accumulation via
    sft_microbatch_train_step and logs mean response-token entropy.

    Returns a dict of final metrics and basic run info.
    """
    if isinstance(device, str):
        device_t = torch.device(device)
    elif isinstance(device, torch.device):
        device_t = device
    else:
        device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve resume path if provided
    load_path: str | Path = model_id
    if resume_from is not None:
        ckpt_path = _resolve_resume_path(resume_from)
        logger.info("Resuming from checkpoint at %s", ckpt_path)
        load_path = ckpt_path

    logger.info("Loading model from %s", load_path)
    tok = AutoTokenizer.from_pretrained(load_path, use_fast=True)
    _ensure_pad_token(tok)
    # Ensure decoder-only models use left padding for correct generation
    tok.padding_side = "left"

    if dtype is None:
        dtype = (
            torch.bfloat16
            if (device_t.type == "cuda" and torch.cuda.is_bf16_supported())
            else torch.float32
        )
    model = AutoModelForCausalLM.from_pretrained(load_path, torch_dtype=dtype)
    # Keep model configs in sync with tokenizer pad token
    if tok.pad_token_id is not None:
        model.config.pad_token_id = tok.pad_token_id
        gen_cfg = getattr(model, "generation_config", None)
        if gen_cfg is not None:
            gen_cfg.pad_token_id = tok.pad_token_id
    model.to(device_t)
    model.train()

    # Optional vLLM instance for eval on separate GPU
    llm: LLM | None = None
    if do_vllm_eval and vllm_device is not None:
        try:
            # Positional args for compatibility with monkeypatched fakes in tests
            llm = init_vllm(model_id, vllm_device, 42, vllm_gpu_memory_utilization)
            logger.info("Initialized vLLM on %s for eval", vllm_device)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to init vLLM: %s", e)
            llm = None

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer.zero_grad(set_to_none=True)

    # Load and lightly validate data
    records = _load_jsonl_pairs(data_path)
    if len(records) == 0:
        raise ValueError(f"No records found in {data_path}")

    prompts_raw = [r["prompt"] for r in records]
    responses = [r["response"] for r in records]

    # Training loop
    step = 0
    total_tokens = 0
    total_loss = 0.0
    total_entropy = 0.0
    total_entropy_count = 0.0

    for epoch in range(num_epochs):
        logger.info("Epoch %d starting (%d examples)", epoch + 1, len(records))
        # Simple sequential pass (shuffle outside if needed)
        for start in range(0, len(records), microbatch_size):
            end = min(start + microbatch_size, len(records))
            if start >= end:
                break

            mb_prompts_raw = prompts_raw[start:end]
            mb_responses = responses[start:end]

            # Build Qwen chat prompts
            mb_chat_prompts = _build_qwen_chat_prompts(tok, mb_prompts_raw)

            # Tokenize prompt+response separately to build shifted inputs and response mask
            tok_out = tokenize_prompt_and_output(mb_chat_prompts, mb_responses, tok)
            input_ids = tok_out["input_ids"]
            labels = tok_out["labels"]
            response_mask = tok_out["response_mask"].to(torch.long)

            # Optional truncation to control context length
            if max_total_tokens is not None and input_ids.size(1) > max_total_tokens:
                input_ids = input_ids[:, :max_total_tokens]
                labels = labels[:, :max_total_tokens]
                response_mask = response_mask[:, :max_total_tokens]

            input_ids = input_ids.to(device_t)
            labels = labels.to(device_t)
            response_mask = response_mask.to(device_t)

            # Build attention mask: non-pad tokens are 1
            pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id or 0
            attention_mask = (input_ids != pad_id).long()

            # Use positional args to avoid keyword name mismatches in tests' fakes
            outputs = model(input_ids, attention_mask)
            logits = outputs.logits  # (B, T, V)

            # Per-token log-probabilities for the actual next token
            log_probs_all = torch.log_softmax(logits, dim=-1)
            policy_log_probs = log_probs_all.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

            # Response-token entropy for logging
            ent_bt = compute_entropy(logits).detach()
            resp_token_count = response_mask.sum().detach().clamp_min(1).item()
            mean_resp_entropy = (
                masked_normalize(ent_bt, response_mask, normalize_constant=float(resp_token_count))
                .detach()
                .item()
            )

            # Use number of response tokens as normalization constant (average NLL/token)
            normalize_constant = float(resp_token_count)

            loss, meta = sft_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=gradient_accumulation_steps,
                normalize_constant=normalize_constant,
            )

            total_loss += loss.detach().item()
            total_tokens += resp_token_count
            total_entropy += mean_resp_entropy * resp_token_count
            total_entropy_count += resp_token_count

            step += 1
            # Optimizer step on accumulation boundary
            if step % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if step % log_every == 0:
                avg_lp = meta["mean_log_prob_response"].item()
                avg_nll = meta["mean_nll_response"].item()
                logger.info(
                    (
                        "step=%d epoch=%d mb=[%d:%d] mean_lp=%.4f mean_nll=%.4f "
                        "mean_ent=%.4f tokens=%d"
                    ),
                    step,
                    epoch + 1,
                    start,
                    end,
                    avg_lp,
                    avg_nll,
                    mean_resp_entropy,
                    int(resp_token_count),
                )
                if wb_log is not None:
                    wb_log(
                        step,
                        {
                            "train/mean_log_prob_response": float(avg_lp),
                            "train/mean_nll_response": float(avg_nll),
                            "train/mean_entropy_response": float(mean_resp_entropy),
                            "train/response_tokens": float(resp_token_count),
                        },
                    )

            if eval_every and step % eval_every == 0:
                # Sample a small subset for eval
                sample_qs = prompts_raw[: max(1, min(eval_examples, len(prompts_raw)))]
                chat_prompts_eval = _build_qwen_chat_prompts(tok, sample_qs)

                # 1) HF generation + entropy via log_generations
                gen_log = log_generations(
                    model,
                    tok,
                    chat_prompts_eval,
                    references=None,
                    max_new_tokens=128,
                    temperature=0.0,
                    top_p=1.0,
                    reward_fn=None,
                )
                logger.info(
                    "eval (HF): avg_token_entropy=%.4f avg_length=%.2f example=%s",
                    gen_log["avg_token_entropy"],
                    gen_log["avg_response_length"],
                    (gen_log["responses"][0] if gen_log["responses"] else ""),
                )
                if on_eval is not None:
                    try:
                        on_eval(step, gen_log)
                    except Exception as e:  # noqa: BLE001
                        logger.warning("on_eval callback failed: %s", e)
                if wb_log is not None:
                    wb_log(
                        step,
                        {
                            "eval/avg_token_entropy": float(gen_log["avg_token_entropy"]),
                            "eval/avg_response_length": float(gen_log["avg_response_length"]),
                        },
                    )

                # 2) Quick score-only eval using get_response_log_probs on a few pairs
                val_pairs = records[: max(1, min(eval_examples, len(records)))]
                val_prompts = _build_qwen_chat_prompts(tok, [r["prompt"] for r in val_pairs])
                val_outputs = [r["response"] for r in val_pairs]
                tok_val = tokenize_prompt_and_output(val_prompts, val_outputs, tok)
                with torch.no_grad():
                    # Positional args for compatibility with monkeypatched fakes in tests
                    scores = get_response_log_probs(
                        model,
                        tok_val["input_ids"].to(device_t),
                        tok_val["labels"].to(device_t),
                        True,
                    )
                mask_val = tok_val["response_mask"].to(torch.float32).to(device_t)
                lp_mean = (
                    masked_normalize(
                        scores["log_probs"],
                        mask_val,
                        normalize_constant=float(mask_val.sum().clamp_min(1)),
                    )
                    .detach()
                    .item()
                )
                ent_mean = (
                    masked_normalize(
                        scores["token_entropy"],
                        mask_val,
                        normalize_constant=float(mask_val.sum().clamp_min(1)),
                    )
                    .detach()
                    .item()
                )
                logger.info("eval (score): mean_log_prob=%.4f mean_entropy=%.4f", lp_mean, ent_mean)
                if wb_log is not None:
                    wb_log(
                        step,
                        {
                            "eval/mean_log_prob": float(lp_mean),
                            "eval/mean_entropy": float(ent_mean),
                        },
                    )

                # 3) vLLM generation on the other GPU, sync weights just-in-time
                if llm is not None:
                    try:
                        load_policy_into_vllm_instance(model, llm)
                        vllm_out = _vllm_generate(
                            llm,
                            chat_prompts_eval,
                            max_new_tokens=128,
                            temperature=0.0,
                            top_p=1.0,
                        )
                        logger.info(
                            "eval (vLLM): avg_length=%.2f example=%s",
                            vllm_out["avg_response_length"],
                            (vllm_out["responses"][0] if vllm_out["responses"] else ""),
                        )
                        if wb_log is not None:
                            wb_log(
                                step,
                                {
                                    "eval_vllm/avg_response_length": float(
                                        vllm_out["avg_response_length"]
                                    ),
                                },
                            )
                    except Exception as e:  # noqa: BLE001
                        logger.warning("vLLM eval failed: %s", e)

            # Checkpointing
            if checkpoint_dir is not None and checkpoint_every is not None:
                if checkpoint_every > 0 and step % checkpoint_every == 0:
                    try:
                        ckpt_root = Path(checkpoint_dir)
                        ckpt_path = ckpt_root / f"step_{step}"
                        ckpt_path.mkdir(parents=True, exist_ok=True)
                        model.save_pretrained(ckpt_path)
                        tok.save_pretrained(ckpt_path)
                        logger.info("Saved checkpoint to %s", ckpt_path)
                    except Exception as e:  # noqa: BLE001
                        logger.warning("Failed to save checkpoint: %s", e)

            if max_steps is not None and step >= max_steps:
                break

        if max_steps is not None and step >= max_steps:
            break

    # Final stats
    avg_nll_per_token = total_loss * gradient_accumulation_steps / max(1.0, total_tokens)
    avg_entropy = total_entropy / max(1.0, total_entropy_count)
    final = {
        "steps": step,
        "total_response_tokens": int(total_tokens),
        "avg_nll_per_token": float(avg_nll_per_token),
        "avg_response_token_entropy": float(avg_entropy),
        "model_id": model_id,
        "device": str(device_t),
        "dtype": str(dtype).replace("torch.", ""),
    }
    if wb_log is not None:
        wb_log(
            step,
            {f"final/{k}": float(v) for k, v in final.items() if isinstance(v, int | float)},
        )
    return final
