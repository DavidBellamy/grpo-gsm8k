from __future__ import annotations

import json
import logging
import os
import shutil
import time
from collections.abc import Callable
from multiprocessing import Queue, get_context
from multiprocessing.context import SpawnProcess
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from torch.nn.utils import clip_grad_norm_
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from vllm import LLM, SamplingParams

from grpo_gsm8k.get_response_log_probs import get_response_log_probs
from grpo_gsm8k.masked_normalize import masked_normalize
from grpo_gsm8k.per_token_entropy import compute_entropy
from grpo_gsm8k.prompts import render_batch
from grpo_gsm8k.tokenize import tokenize_prompt_and_output

logger = logging.getLogger(__name__)


# -----------------------------
# Training primitives
# -----------------------------


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if policy_log_probs.shape != response_mask.shape:
        raise ValueError(
            f"response_mask shape {response_mask.shape} must match "
            f"policy_log_probs shape {policy_log_probs.shape}"
        )
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be a positive integer")

    nll: torch.Tensor = -masked_normalize(
        policy_log_probs, response_mask, normalize_constant, dim=None
    )
    loss: torch.Tensor = nll / float(gradient_accumulation_steps)
    loss.backward()

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


# -----------------------------
# Lightweight checkpoint for vLLM hot-reload
# -----------------------------


def save_policy_checkpoint_for_vllm(
    policy: PreTrainedModel,
    step: int,
    *,
    out_root: str | Path = "/dev/shm",
    base: str = "qwen15b_step",
    dtype: torch.dtype = torch.bfloat16,
) -> Path:
    """
    Write a weights-only HF-style folder:
      /dev/shm/{base}_{step}/
        - config.json
        - pytorch_model.safetensors
        - READY (marker)
    """
    out_root = Path(out_root)
    tmp_dir = out_root / f"{base}_{step}.tmp"
    final_dir = out_root / f"{base}_{step}"
    ready = final_dir / "READY"

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    if final_dir.exists():
        logger.warning("Overwriting existing %s", final_dir)
        shutil.rmtree(final_dir)

    tmp_dir.mkdir(parents=True, exist_ok=True)

    # config.json
    cfg_json = policy.config.to_json_string()
    (tmp_dir / "config.json").write_text(cfg_json, encoding="utf-8")

    # safetensors weights (cast + move to cpu)
    state: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for k, v in policy.state_dict().items():
            state[k] = v.detach().to(dtype).to("cpu", non_blocking=True).contiguous()
    save_file(state, str(tmp_dir / "pytorch_model.safetensors"))

    # Atomic publish
    try:
        os.sync()
    except Exception:
        pass
    os.rename(tmp_dir, final_dir)
    ready.touch()
    logger.info("Wrote vLLM checkpoint: %s", final_dir)
    return final_dir


# -----------------------------
# vLLM persistent worker (GPU1)
# -----------------------------


def _vllm_hot_reload_from_dir(llm: LLM, ckpt_dir: str | Path) -> None:
    """Reload weights in-place from a prepared directory (safetensors)."""
    eng = getattr(llm, "engine", None) or getattr(llm, "llm_engine", None)
    if eng is None:
        raise RuntimeError("Could not find LLM engine on the LLM object.")

    # Best-effort pause
    try:
        eng.engine_core.sleep()
    except Exception:
        pass

    # Update config + reload
    llm.collective_rpc(
        "update_config",
        args=(
            {
                "model_config": {"model": str(ckpt_dir)},
                "load_config": {"load_format": "safetensors"},
            },
        ),
    )
    llm.collective_rpc("reload_weights")

    # Wake engine
    try:
        eng.engine_core.wake_up()
    except Exception:
        pass


def _vllm_worker_persistent(
    jobs_q: Queue[dict[str, Any] | None],
    results_q: Queue[dict[str, Any]],
    *,
    base_model_id: str,
    gpu_id: str = "1",
    gpu_memory_utilization: float = 0.85,
    dtype: str = "bfloat16",
) -> None:
    """
    Persistent vLLM engine pinned to a single GPU. Blocks on jobs_q.get() until sentinel None.
    Job schema:
      {"ckpt_dir": str, "prompts": list[str], "step": int,
       "max_new_tokens": int, "temperature": float, "top_p": float}
    Result schema:
      {"step": int, "responses": list[str], "avg_response_length": float, "ts": float}
    """
    # Isolate the target GPU in this process; then it's cuda:0 inside the worker.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Create engine ONCE from base model
    llm = LLM(
        model=base_model_id,
        dtype=dtype,
        enable_prefix_caching=True,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    last_loaded_path: str | None = None

    while True:
        job = jobs_q.get()  # blocking; empty queue does NOT shutdown
        if job is None:
            # Clean shutdown
            try:
                if hasattr(llm, "shutdown"):
                    llm.shutdown()
            except Exception:
                pass
            break

        ckpt_dir = str(job["ckpt_dir"])
        prompts = list(job["prompts"])
        step = int(job.get("step", -1))
        max_new_tokens = int(job.get("max_new_tokens", 128))
        temperature = float(job.get("temperature", 0.0))
        top_p = float(job.get("top_p", 1.0))

        # Hot-reload only when the checkpoint path changes
        if last_loaded_path != ckpt_dir:
            try:
                _vllm_hot_reload_from_dir(llm, ckpt_dir)
                last_loaded_path = ckpt_dir
            except Exception as e:
                results_q.put(
                    {
                        "step": step,
                        "error": f"hot_reload_failed: {e}",
                        "ts": time.time(),
                    }
                )
                continue

        # Generate
        samp = SamplingParams(max_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
        outs = llm.generate(prompts, samp)

        responses: list[str] = []
        lengths: list[int] = []
        for out in outs:
            if not out.outputs:
                responses.append("")
                lengths.append(0)
                continue
            text = out.outputs[0].text or ""
            responses.append(text)
            token_ids = getattr(out.outputs[0], "token_ids", None)
            lengths.append(len(token_ids) if token_ids is not None else len(text.split()))
        avg_len = sum(lengths) / max(len(lengths), 1)

        results_q.put(
            {
                "step": step,
                "responses": responses,
                "avg_response_length": float(avg_len),
                "ts": time.time(),
            }
        )


# -----------------------------
# Simple SFT utilities
# -----------------------------


def _ensure_pad_token(tokenizer: PreTrainedTokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})


def _load_jsonl_pairs(path: str | Path) -> list[dict[str, str]]:
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
    # Pass add_generation_prompt as a positional arg for compatibility with tests
    return render_batch(tokenizer, prompts_raw, True)


def _resolve_resume_path(path: str | Path) -> Path:
    p = Path(path)
    if (p / "config.json").exists():
        return p
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


# -----------------------------
# Main training loop
# -----------------------------


def train_sft_on_r1_pairs(
    data_path: str | Path,
    *,
    model_id: str = "Qwen/Qwen2.5-Math-1.5B",
    device: str | torch.device = "cuda:0",
    vllm_device: str | None = "cuda:1",
    vllm_gpu_memory_utilization: float = 0.85,
    microbatch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    num_epochs: int = 1,
    max_steps: int | None = None,
    learning_rate: float = 1e-5,
    adamw_beta1: float = 0.9,
    adamw_beta2: float = 0.95,
    adamw_eps: float = 1e-8,
    weight_decay: float = 0.0,
    max_grad_norm: float | None = 1.0,
    max_total_tokens: int | None = 2048,
    log_every: int = 10,
    eval_every: int = 200,
    eval_examples: int = 4,
    teacher_eval_examples: int = 4,
    teacher_eval_every: int | None = None,
    # Generation params for vLLM worker
    gen_max_new_tokens: int = 2048,
    gen_temperature: float = 0.0,
    gen_top_p: float = 1.0,
    model_dtype: torch.dtype | None = None,
    wb_log: Callable[[int, dict[str, float]], None] | None = None,
    checkpoint_dir: str | Path | None = None,
    checkpoint_every: int | None = None,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    """
    Train SFT on JSONL with {"prompt","response"}.
    - GPU0 (trainer): training + tiny teacher-forced metrics (no generation).
    - GPU1 (worker): persistent vLLM engine; async generation with hot-reloaded weights.
    """
    # Resolve device
    if isinstance(device, str):
        device_t = torch.device(device)
    elif isinstance(device, torch.device):
        device_t = device
    else:
        raise ValueError("device must be a string (e.g., 'cuda:0') or a torch.device instance.")

    # Resolve resume path
    load_path: str | Path = model_id
    if resume_from is not None:
        ckpt_path = _resolve_resume_path(resume_from)
        logger.info("Resuming from checkpoint at %s", ckpt_path)
        load_path = ckpt_path

    logger.info("Loading model from %s", load_path)
    tok = AutoTokenizer.from_pretrained(load_path, use_fast=True)
    _ensure_pad_token(tok)
    tok.padding_side = "left"

    if model_dtype is None:
        if device_t.type == "cuda":
            if torch.cuda.is_bf16_supported():
                model_dtype = torch.bfloat16
            else:
                logger.warning("CUDA device does not support bfloat16; falling back to float16.")
                model_dtype = torch.float16
        else:
            model_dtype = torch.float32  # CPU/debug

    model = AutoModelForCausalLM.from_pretrained(load_path, torch_dtype=model_dtype)
    if tok.pad_token_id is not None:
        model.config.pad_token_id = tok.pad_token_id
        gen_cfg = getattr(model, "generation_config", None)
        if gen_cfg is not None:
            gen_cfg.pad_token_id = tok.pad_token_id
    model.to(device_t)
    model.train()

    # Start persistent vLLM worker on GPU1 (if requested)
    jobs_q: Queue | None = None
    results_q: Queue | None = None
    vllm_proc: SpawnProcess | None = None

    if vllm_device is not None:
        # Use a dedicated 'spawn' context for queues and process (safer for CUDA + vLLM)
        ctx = get_context("spawn")
        jobs_q = ctx.Queue(maxsize=16)
        results_q = ctx.Queue(maxsize=16)
        gpu_id = vllm_device.replace("cuda:", "")  # e.g., "1"

        vllm_proc = ctx.Process(
            target=_vllm_worker_persistent,
            args=(jobs_q, results_q),
            kwargs=dict(
                base_model_id=model_id,
                gpu_id=gpu_id,
                gpu_memory_utilization=float(vllm_gpu_memory_utilization),
                dtype="bfloat16",
            ),
            daemon=False,  # IMPORTANT: allow vLLM to spawn children
        )
        if vllm_proc is not None:
            vllm_proc.start()
            logger.info("Started persistent vLLM worker on GPU %s (pid=%s)", gpu_id, vllm_proc.pid)
    else:
        logger.warning("vllm_device is None; async generation is disabled.")

    # Optimizer (no-decay for bias/LayerNorm/embeddings)
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        n = name.lower()
        if (
            param.ndim == 1
            or name.endswith(".bias")
            or "layernorm" in n
            or "layer_norm" in n
            or "embed" in n
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(adamw_beta1, adamw_beta2),
        eps=adamw_eps,
    )
    optimizer.zero_grad(set_to_none=True)

    # Data
    records = _load_jsonl_pairs(data_path)
    if len(records) == 0:
        raise ValueError(f"No records found in {data_path}")

    prompts_raw = [r["prompt"] for r in records]
    responses = [r["response"] for r in records]

    # Loop state
    step = 0
    total_tokens = 0
    total_nonpad_tokens = 0
    total_loss = 0.0
    total_entropy = 0.0
    total_entropy_count = 0.0
    start_time = time.perf_counter()

    if teacher_eval_every is None:
        teacher_eval_every = eval_every
    outstanding_jobs = 0
    for epoch in range(num_epochs):
        logger.info("Epoch %d starting (%d examples)", epoch + 1, len(records))
        for start in range(0, len(records), microbatch_size):
            end = min(start + microbatch_size, len(records))
            if start >= end:
                break

            mb_prompts_raw = prompts_raw[start:end]
            mb_responses = responses[start:end]

            # Build chat prompts
            mb_chat_prompts = _build_qwen_chat_prompts(tok, mb_prompts_raw)

            # Tokenize prompt+response to build labels and response mask
            tok_out = tokenize_prompt_and_output(mb_chat_prompts, mb_responses, tok)
            input_ids = tok_out["input_ids"]
            labels = tok_out["labels"]
            response_mask = tok_out["response_mask"].to(torch.long)

            # Truncate if needed
            if max_total_tokens is not None and input_ids.size(1) > max_total_tokens:
                input_ids = input_ids[:, :max_total_tokens]
                labels = labels[:, :max_total_tokens]
                response_mask = response_mask[:, :max_total_tokens]

            input_ids = input_ids.to(device_t)
            labels = labels.to(device_t)
            response_mask = response_mask.to(device_t)

            # Attention mask
            pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id or 0
            attention_mask = (input_ids != pad_id).long()
            nonpad_token_count = int(attention_mask.sum().item())
            total_nonpad_tokens += nonpad_token_count

            # Forward
            outputs = model(input_ids, attention_mask)
            logits = outputs.logits  # (B, T, V)

            # Per-token log-probabilities for the actual next token
            log_probs_all = torch.log_softmax(logits, dim=-1)
            policy_log_probs = log_probs_all.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

            # Response-token entropy (for logging)
            ent_bt = compute_entropy(logits).detach()
            resp_token_count = response_mask.sum().detach().clamp_min(1).item()
            mean_resp_entropy = (
                masked_normalize(ent_bt, response_mask, normalize_constant=float(resp_token_count))
                .detach()
                .item()
            )

            # Loss normalized by #response tokens
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
                if max_grad_norm is not None and max_grad_norm > 0:
                    clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # Periodic logs
            if step % log_every == 0:
                elapsed = time.perf_counter() - start_time
                tps_response = total_tokens / max(elapsed, 1e-9)
                tps_nonpad = total_nonpad_tokens / max(elapsed, 1e-9)
                avg_lp = meta["mean_log_prob_response"].item()
                avg_nll = meta["mean_nll_response"].item()
                logger.info(
                    (
                        "step=%d epoch=%d mb=[%d:%d] mean_lp=%.4f mean_nll=%.4f "
                        "mean_ent=%.4f resp_tokens=%d tps_resp=%.1f tps_nonpad=%.1f"
                    ),
                    step,
                    epoch + 1,
                    start,
                    end,
                    avg_lp,
                    avg_nll,
                    mean_resp_entropy,
                    int(resp_token_count),
                    tps_response,
                    tps_nonpad,
                )
                if wb_log is not None:
                    wb_log(
                        step,
                        {
                            "perf/tokens_per_sec_response": float(tps_response),
                            "perf/tokens_per_sec_nonpad": float(tps_nonpad),
                            "train/mean_log_prob_response": float(avg_lp),
                            "train/mean_nll_response": float(avg_nll),
                            "train/mean_entropy_response": float(mean_resp_entropy),
                            "train/response_tokens": float(resp_token_count),
                        },
                    )

            # --- Tiny teacher-forced eval on GPU0 ---
            if teacher_eval_every and step % teacher_eval_every == 0:
                k = max(1, min(int(teacher_eval_examples), len(records)))
                val_pairs = records[:k]
                val_prompts = _build_qwen_chat_prompts(tok, [r["prompt"] for r in val_pairs])
                val_outputs = [r["response"] for r in val_pairs]
                tok_val = tokenize_prompt_and_output(val_prompts, val_outputs, tok)

                with torch.inference_mode():
                    scores = get_response_log_probs(
                        model,
                        tok_val["input_ids"].to(device_t),
                        tok_val["labels"].to(device_t),
                        True,
                    )
                mask_val = tok_val["response_mask"].to(torch.float32).to(device_t)
                denom = float(mask_val.sum().clamp_min(1))
                lp_mean = (scores["log_probs"] * mask_val).sum().item() / denom
                ent_mean = (scores["token_entropy"] * mask_val).sum().item() / denom

                logger.info(
                    "teacher (GPU0): mean_log_prob=%.4f mean_entropy=%.4f", lp_mean, ent_mean
                )
                if wb_log is not None:
                    wb_log(
                        step,
                        {
                            "teacher/mean_log_prob": float(lp_mean),
                            "teacher/mean_entropy": float(ent_mean),
                        },
                    )

            # --- Enqueue async vLLM generation on GPU1 (persistent worker) ---
            if eval_every and step % eval_every == 0 and jobs_q is not None:
                kgen = max(1, min(int(eval_examples), len(prompts_raw)))
                prompts_eval = _build_qwen_chat_prompts(tok, prompts_raw[:kgen])

                # Save lightweight checkpoint and enqueue a job
                try:
                    ckpt_dir = save_policy_checkpoint_for_vllm(
                        model, step, out_root="/dev/shm", base="qwen15b_step", dtype=torch.bfloat16
                    )
                except Exception as e:
                    logger.warning("Failed to save vLLM ckpt for async eval: %s", e)
                    ckpt_dir = None

                if ckpt_dir is not None:
                    try:
                        jobs_q.put_nowait(
                            {
                                "ckpt_dir": str(ckpt_dir),
                                "prompts": prompts_eval,
                                "step": int(step),
                                "max_new_tokens": int(gen_max_new_tokens),
                                "temperature": float(gen_temperature),
                                "top_p": float(gen_top_p),
                            }
                        )
                        outstanding_jobs += 1
                    except Exception as e:
                        logger.warning("Failed to enqueue async eval job: %s", e)

            # --- Opportunistic, non-blocking result drain ---
            if results_q is not None:
                drained = 0
                while True:
                    try:
                        result = results_q.get_nowait()
                    except Exception:
                        break
                    drained += 1
                    if "error" in result:
                        logger.warning("async vLLM error (GPU1): %s", result["error"])
                        continue
                    outstanding_jobs = max(0, outstanding_jobs - 1)
                    avg_len = float(result.get("avg_response_length", 0.0))
                    example = (result.get("responses", [""]) or [""])[0]
                    logger.info(
                        "async vLLM (GPU1): step=%s avg_len=%.2f example=%s",
                        result.get("step", "?"),
                        avg_len,
                        example,
                    )
                    if wb_log is not None:
                        wb_log(step, {"eval/avg_response_length": avg_len})

            # Checkpointing (optional)
            if checkpoint_dir is not None and checkpoint_every is not None:
                if checkpoint_every > 0 and step % checkpoint_every == 0:
                    try:
                        ckpt_root = Path(checkpoint_dir)
                        ckpt_path = ckpt_root / f"step_{step}"
                        ckpt_path.mkdir(parents=True, exist_ok=True)
                        model.save_pretrained(ckpt_path)
                        tok.save_pretrained(ckpt_path)
                        logger.info("Saved checkpoint to %s", ckpt_path)
                    except Exception as e:
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
        "model_dtype": str(model_dtype).replace("torch.", ""),
    }

    # --- Graceful shutdown of async eval worker & queues ---
    await_async_eval_at_end = True  # True = wait briefly for last job
    try:
        # If requested, wait for outstanding eval jobs to finish (bounded time)
        if await_async_eval_at_end and jobs_q is not None and results_q is not None:
            import queue

            deadline = time.time() + 600  # up to 10 minutes; tune as you like
            # Keep draining until all jobs come back or we hit the deadline
            while outstanding_jobs > 0 and time.time() < deadline:
                try:
                    result = results_q.get(timeout=2.0)
                except queue.Empty:
                    continue
                if "error" in result:
                    logger.warning("async vLLM error (GPU1 on shutdown): %s", result["error"])
                else:
                    outstanding_jobs = max(0, outstanding_jobs - 1)
                    avg_len = float(result.get("avg_response_length", 0.0))
                    logger.info(
                        "async vLLM (GPU1 final): step=%s avg_len=%.2f",
                        result.get("step", "?"),
                        avg_len,
                    )
                    if wb_log is not None:
                        wb_log(step, {"eval/avg_response_length": avg_len})
            # Tell the worker to exit once we're done waiting
            try:
                jobs_q.put_nowait(None)
            except Exception:
                pass

        # Join worker with a generous timeout; only hard-kill if it still won't exit
        if vllm_proc is not None:
            vllm_proc.join(timeout=60.0)
            if vllm_proc.is_alive():
                logger.warning("vLLM worker still alive after timeout; terminating...")
                vllm_proc.terminate()
                vllm_proc.join(timeout=10.0)

        # Close queues and cancel feeder threads to avoid atexit hangs
        if results_q is not None:
            try:
                results_q.close()
                results_q.cancel_join_thread()
            except Exception:
                pass
        if jobs_q is not None:
            try:
                jobs_q.close()
                jobs_q.cancel_join_thread()
            except Exception:
                pass
    except Exception:
        pass

    return final
