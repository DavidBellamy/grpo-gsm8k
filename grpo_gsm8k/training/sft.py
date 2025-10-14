from __future__ import annotations

import glob
import json
import logging
import math
import os
import queue
import shutil
import statistics
import time
from multiprocessing import Queue, get_context
from multiprocessing.context import SpawnProcess
from pathlib import Path
from typing import Any

import torch
import wandb
from safetensors.torch import save_file
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from vllm import LLM, SamplingParams

from grpo_gsm8k.core.masked_normalize import masked_normalize
from grpo_gsm8k.core.per_token_entropy import compute_entropy
from grpo_gsm8k.data.data_loader import BucketMicrobatcher, PTShardStream
from grpo_gsm8k.evaluation.reward_fn import extract_answer_colon, extract_boxed, normalize_number

logger = logging.getLogger(__name__)


# -----------------------------
# Training primitives
# -----------------------------


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if policy_log_probs.shape != response_mask.shape:
        raise ValueError(
            f"response_mask shape {response_mask.shape} must match "
            f"policy_log_probs shape {policy_log_probs.shape}"
        )

    # NLL as masked sum (loss-bearing token summation)
    nll: torch.Tensor = -masked_normalize(policy_log_probs, response_mask)
    loss: torch.Tensor = nll
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
        "avg_response_token_logprob": mean_log_prob,
        "avg_response_token_nll": mean_nll,
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
      {
        "ckpt_dir": str,
        "prompts": list[str],
        "answers": list[str],          # ground-truth
        "step": int,
        "max_new_tokens": int,
        "temperature": float,
        "top_p": float,
        "gold_strs": list[str],     # original gold answer strings (for logging)
      }
    Result schema:
      {
        "step": int,
        "responses": list[str],
        "answers": list[str],
        "gold_strs": list[str],
        "prompts": list[str],
        "avg_response_length": float,
        "lengths": list[int],
        "truncated": list[int],
        "stop_reasons": list[str],
        "avg_token_logprobs": list[float],
        "perplexities": list[float],
        "rep3_ratios": list[float],
        "pred_answers": list[str],
        "correct": list[int],
        "accuracy": float,
        "truncation_rate": float,
        "length_p50": float,
        "length_p95": float,
        "toks_per_sec": float,
        "gen_token_total": int,
        "ts": float,
      }
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
        answers = list(job.get("answers", []))
        step = int(job.get("step", -1))
        max_new_tokens = int(job.get("max_new_tokens", 2048))
        temperature = float(job.get("temperature", 0.0))
        top_p = float(job.get("top_p", 1.0))
        gold_strs = list(job.get("gold_strs", []))

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

        # Generate with logprobs
        samp = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            logprobs=1,  # request only the chosen token's logprob
            prompt_logprobs=0,
        )
        t0 = time.time()
        outs = llm.generate(prompts, samp)
        t1 = time.time()

        responses: list[str] = []
        lengths: list[int] = []
        trunc_flags: list[int] = []
        stop_reasons: list[str] = []
        avg_token_logprobs: list[float] = []
        perplexities: list[float] = []
        rep3_ratios: list[float] = []
        pred_answers: list[str] = []
        correct_flags: list[int] = []
        gen_token_counts: list[int] = []

        for out in outs:
            if not out.outputs:
                responses.append("")
                lengths.append(0)
                trunc_flags.append(0)
                stop_reasons.append("none")
                avg_token_logprobs.append(float("nan"))
                perplexities.append(float("nan"))
                rep3_ratios.append(1.0)
                pred_answers.append("")
                correct_flags.append(0)
                gen_token_counts.append(0)
                continue

            choice = out.outputs[0]
            text = choice.text or ""
            responses.append(text)
            token_ids = getattr(choice, "token_ids", None)
            gen_len = len(token_ids) if token_ids is not None else len(text.split())
            lengths.append(gen_len)
            gen_token_counts.append(len(token_ids) if token_ids is not None else 0)

            # stop reason / truncation
            fr = getattr(choice, "finish_reason", None) or ""
            stop_reasons.append(fr)
            trunc_flags.append(1 if fr == "length" else 0)

            # avg token logprob & perplexity
            lp = None
            try:
                vals: list[float] = []
                for step_dict in choice.logprobs:  # list[dict[token_id, Logprob]]
                    first = next(iter(step_dict.values()))
                    vals.append(float(first.logprob))  # chosen token's logprob
                if vals:
                    lp = sum(vals) / len(vals)
            except Exception:
                lp = None
            avg_token_logprobs.append(lp if lp is not None else float("nan"))
            perplexities.append(float(math.exp(-lp)) if lp is not None else float("nan"))

            # repetition (3-gram duplication ratio)
            rep3 = _ngram_repetition_ratio(token_ids or [], n=3)
            rep3_ratios.append(float(rep3))

            # extract predicted answer and correctness
            pa = _normalize_num_str(_extract_final_answer(text))
            pred_answers.append(pa)
            if answers and len(answers) >= len(pred_answers):
                gt = _normalize_num_str(answers[len(pred_answers) - 1])
                correct_flags.append(1 if (gt and pa and gt == pa) else 0)
            else:
                correct_flags.append(0)

        avg_len = sum(lengths) / max(len(lengths), 1)
        tot_gen_toks = sum(gen_token_counts)
        toks_per_sec = (tot_gen_toks / max(t1 - t0, 1e-6)) if tot_gen_toks else 0.0

        # Percentiles
        if lengths:
            lengths_sorted = sorted(lengths)
            p50: float = statistics.median(lengths_sorted)
            p95: float = lengths_sorted[int(0.95 * (len(lengths_sorted) - 1))]
        else:
            p50 = 0.0
            p95 = 0.0

        results_q.put(
            {
                "step": step,
                "responses": responses,
                "answers": answers,
                "gold_strs": gold_strs,
                "prompts": prompts,
                "avg_response_length": float(avg_len),
                # per-example
                "lengths": lengths,
                "truncated": trunc_flags,
                "stop_reasons": stop_reasons,
                "avg_token_logprobs": avg_token_logprobs,
                "perplexities": perplexities,
                "rep3_ratios": rep3_ratios,
                "pred_answers": pred_answers,
                "correct": correct_flags,
                # aggregates
                "accuracy": float(sum(correct_flags) / max(len(correct_flags), 1)),
                "truncation_rate": float(sum(trunc_flags) / max(len(trunc_flags), 1)),
                "length_p50": float(p50),
                "length_p95": float(p95),
                "toks_per_sec": float(toks_per_sec),
                "gen_token_total": int(tot_gen_toks),
                "ts": time.time(),
            }
        )


# -----------------------------
# SFT utilities
# -----------------------------


def count_examples(train_data_path: str | Path) -> int:
    paths: list[str] = []
    if os.path.isfile(train_data_path):
        paths = [str(train_data_path)]
    else:
        paths = glob.glob(str(train_data_path))
    if not paths:
        raise FileNotFoundError(f"No shards match {train_data_path}")
    total = 0
    for p in paths:
        x = torch.load(p, map_location="cpu")
        # trust your writer's meta, else fallback to tensor length
        total += int(x["meta"].get("count", x["input_ids"].shape[0]))
    return total


def count_response_tokens(train_data_path: str | Path) -> int:
    paths: list[str] = []
    if os.path.isfile(train_data_path):
        paths = [str(train_data_path)]
    else:
        paths = glob.glob(str(train_data_path))
    if not paths:
        raise FileNotFoundError(f"No shards match {train_data_path}")
    total = 0
    for p in paths:
        x = torch.load(p, map_location="cpu")
        total += int(x["response_mask"].long().sum().item())
    return total


def pad_collate(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    # Max length within this microbatch
    L = max(int(s.get("len", len(s["input_ids"]))) for s in batch)
    B = len(batch)

    input_ids = torch.full((B, L), pad_id, dtype=torch.long)
    labels = torch.full((B, L), -100, dtype=torch.long)
    response = torch.zeros((B, L), dtype=torch.long)

    for i, s in enumerate(batch):
        x = s["input_ids"]
        y = s["labels"]
        r = s["response_mask"]
        # Convert to tensors if needed
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.long)
        if not torch.is_tensor(y):
            y = torch.tensor(y, dtype=torch.long)
        if not torch.is_tensor(r):
            r = torch.tensor(r, dtype=torch.long)

        n = min(int(x.shape[0]), L)  # safety; microbatcher already truncates
        input_ids[i, :n] = x[:n]
        labels[i, :n] = y[:n]
        response[i, :n] = r[:n]

    attention_mask = (input_ids != pad_id).long()

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "response_mask": response,
    }


def _load_prerendered_eval_set(path: str | Path) -> tuple[list[str], list[str], list[str]]:
    """
    Strict loader for pre-rendered eval sets (JSONL) for vLLM eval.
    Requires each row to have:
      - "prompt": chat-rendered string
      - "gold": original gold answer string (for logging)
      - "gold_num": normalized numeric gold as string (for exact match)
    No on-the-fly rendering or parsing is performed here.
    """
    p = Path(path)
    prompts: list[str] = []
    gold_strs: list[str] = []
    gold_nums: list[str] = []
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "prompt" not in obj or "gold" not in obj or "gold_num" not in obj:
                raise ValueError(
                    f"Eval row {i} missing required keys. Expected 'prompt','gold','gold_num'."
                )
            prompt = str(obj["prompt"])
            gold = str(obj["gold"])
            gold_num_raw = obj["gold_num"]
            if gold_num_raw is None:
                raise ValueError(f"Eval row {i} has null gold_num")
            gold_num = str(gold_num_raw)
            if gold_num == "":
                raise ValueError(f"Eval row {i} has empty gold_num (no on-the-fly parsing allowed)")
            prompts.append(prompt)
            gold_strs.append(gold)
            gold_nums.append(gold_num)
    if not prompts:
        raise ValueError(f"No eval examples loaded from {path}")
    return prompts, gold_strs, gold_nums


def _ensure_pad_token(tokenizer: PreTrainedTokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})


def _resolve_resume_path(root: str | Path) -> Path:
    p = Path(root)
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
        latest = step_dirs[0][1]
        # Check if the latest directory contains the expected files
        if (latest / "config.json").exists() and (latest / "pytorch_model.safetensors").exists():
            return latest
    return p


# -----------------------------
# Answer extraction helpers
# -----------------------------


def _extract_final_answer(text: str) -> str | None:
    """
    Try boxed then 'ANSWER:' style extraction; fall back to None.
    """
    return extract_boxed(text) or extract_answer_colon(text)


def _normalize_num_str(s: str | None) -> str:
    """
    Normalize numeric substrings (remove commas, keep canonical numeric token).
    Returns '' when no parseable number found (keeps length aligned).
    """
    if s is None:
        return ""
    n = normalize_number(s)
    return n if n is not None else ""


def _ngram_repetition_ratio(token_ids: list[int], n: int = 3) -> float:
    """
    Fraction of duplicated n-grams among all n-grams.
    Returns 0.0 when fewer than n tokens.
    """
    if n <= 0 or len(token_ids) < n:
        return 0.0
    total = len(token_ids) - n + 1
    seen: dict[tuple[int, ...], int] = {}
    dup = 0
    for i in range(total):
        ng = tuple(token_ids[i : i + n])
        c = seen.get(ng, 0)
        if c == 1:
            dup += 1  # count second occurrence as duplication start
        elif c > 1:
            dup += 1
        seen[ng] = c + 1
    return dup / total if total > 0 else 0.0


# -----------------------------
# Main training loop
# -----------------------------


def train_sft_on_r1_pairs(
    train_data_path: str | Path,
    eval_data_path: str | Path | None = None,
    *,
    model_id: str = "Qwen/Qwen2.5-Math-1.5B",
    device: str | torch.device = "cuda:0",
    vllm_device: str | torch.device = "cuda:1",
    vllm_gpu_memory_utilization: float = 0.85,
    microbatch_size: int = 2,
    num_epochs: int = 1,
    min_tokens_per_update: int = 4096,
    max_update_steps: int | None = None,
    learning_rate: float = 1e-5,
    adamw_beta1: float = 0.9,
    adamw_beta2: float = 0.95,
    adamw_eps: float = 1e-8,
    weight_decay: float = 0.0,
    max_grad_norm: float | None = 1.0,
    max_total_tokens: int | None = 2048,
    eval_every: int = 4,
    eval_examples: int | None = None,
    # Generation params for vLLM worker
    gen_max_new_tokens: int = 2048,
    gen_temperature: float = 0.0,
    gen_top_p: float = 1.0,
    model_dtype: torch.dtype | None = None,
    checkpoint_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
) -> None:
    """
    Train SFT on JSONL with {"prompt","response"}.
    - GPU0 (trainer): training loop with AdamW; microbatching with BucketMicrobatcher.
    - GPU1 (worker): persistent vLLM engine; async generation with hot-reloaded weights.

    Updates are performed once at least `min_tokens_per_update` loss-bearing tokens
    (response tokens) have been processed. Gradients are accumulated as a sum across microbatches
    and divided by the accumulated response-token count immediately before optimizer.step().
    All logging, async eval, and checkpoint triggers are update-based.

    Checkpointing: if `checkpoint_dir` is provided, a full HF checkpoint is saved at every
    `eval_every` update (aligned with async eval). No separate checkpoint cadence exists.
    """
    # Decouple steps across train and eval so async work doesn't break monotonicity
    wandb.define_metric("steps/train_step")
    wandb.define_metric("train/*", step_metric="steps/train_step")
    wandb.define_metric("steps/eval_step")
    wandb.define_metric("eval/*", step_metric="steps/eval_step")

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
    model.config.pad_token_id = tok.pad_token_id
    model.config.use_cache = False
    gc = getattr(model, "generation_config", None)
    if gc is not None:
        gc.pad_token_id = tok.pad_token_id  # make generate() consistent
    model.to(device_t)
    model.train()

    # Load pre-rendered eval set
    eval_prompts_chat: list[str] = []
    eval_gold_strs: list[str] = []
    eval_gold_nums: list[str] = []
    if eval_every:
        if eval_data_path is None:
            raise ValueError(
                "eval_every=%d but eval_data_path is not provided.",
                eval_every,
            )
        else:
            eval_prompts_chat, eval_gold_strs, eval_gold_nums = _load_prerendered_eval_set(
                eval_data_path
            )
            logger.info(
                "Loaded pre-rendered eval set for vLLM: n=%d from %s",
                len(eval_prompts_chat),
                eval_data_path,
            )

    # Start persistent vLLM worker on separate GPU
    jobs_q: Queue | None = None
    results_q: Queue | None = None
    vllm_proc: SpawnProcess | None = None

    # Use a dedicated 'spawn' context for queues and process (safer for CUDA + vLLM)
    ctx = get_context("spawn")
    jobs_q = ctx.Queue(maxsize=64)
    results_q = ctx.Queue(maxsize=64)
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
        raise RuntimeError("Failed to start vLLM worker process.")

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

    # Load training data stats
    num_train_examples = count_examples(train_data_path)
    total_resp_tokens_dataset = count_response_tokens(train_data_path)

    # --- LR scheduler: warmup for min(200, 0.025*total_steps); cosine to 10% base ---
    if max_update_steps is not None:
        total_opt_steps = int(max(1, max_update_steps))
    else:
        # Estimate total updates from response token budget
        expected_updates = math.ceil(
            (total_resp_tokens_dataset * max(1, num_epochs)) / max(1, int(min_tokens_per_update))
        )
        total_opt_steps = int(max(1, expected_updates))

    warmup_steps = int(min(200, math.floor(0.025 * total_opt_steps)))
    min_lr_scale = 0.10  # 10% of base

    def _lr_lambda(step_idx: int) -> float:
        # step_idx is 0-based number of calls to scheduler.step() - 1
        s = step_idx + 1  # make it 1-based
        if warmup_steps > 0 and s <= warmup_steps:
            return float(s) / float(max(1, warmup_steps))
        if s >= total_opt_steps:
            return float(min_lr_scale)
        span = max(1, total_opt_steps - warmup_steps)
        progress = float(s - warmup_steps) / float(span)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(min_lr_scale + (1.0 - min_lr_scale) * cosine)

    scheduler = LambdaLR(optimizer, lr_lambda=_lr_lambda)
    logger.info(
        "LR scheduler: base_lr=%.2e total_steps=%d warmup_steps=%d final_scale=%.2f",
        learning_rate,
        total_opt_steps,
        warmup_steps,
        min_lr_scale,
    )

    # Loop state
    update_step = 0  # optimizer update count
    total_tokens = 0
    total_nonpad_tokens = 0
    start_time = time.perf_counter()

    # --- Single persistent W&B table that we update once per eval step ---
    eval_table_cols: list[str] = [
        "eval_step",
        "prompt",
        "gold_reasoning",
        "model_response",
        "model_answer",
        "correct",
        "response_length",
        "truncated",
        "stop_reason",
    ]
    eval_table = wandb.Table(columns=eval_table_cols, log_mode="INCREMENTAL")

    # Main epoch loop
    outstanding_jobs = 0
    for epoch in range(num_epochs):
        logger.info("Epoch %d starting (%d examples)", epoch + 1, num_train_examples)

        stream = PTShardStream(str(train_data_path), shuffle_files=False)
        mb = BucketMicrobatcher(
            microbatch_size, max_seq_len=max_total_tokens if max_total_tokens is not None else 2048
        )

        # Accumulate unnormalized grads until reaching token threshold
        tokens_since_update = 0
        sum_nll_since_update = 0.0
        sum_entropy_since_update = 0.0
        nonpad_tokens_since_update = 0
        total_elements_since_update = 0  # total tokens (pad + non-pad) across microbatches

        for sample in stream:
            maybe = mb.add(sample)
            if not maybe:
                continue

            batch = pad_collate(maybe, tok.pad_token_id)
            input_ids = batch["input_ids"].to(device_t)
            labels = batch["labels"].to(device_t)
            attention_mask = batch["attention_mask"].to(device_t)
            response_mask = batch["response_mask"].to(device_t)

            debug_checks = bool(int(os.environ.get("DEBUG_CHECKS", "0")))  # run with DEBUG_CHECKS=1
            if debug_checks:
                # 1) No loss-bearing token should have labels == -100
                assert not torch.any(
                    response_mask.bool() & labels.eq(-100)
                ), "response_mask=1 but labels=-100 → labeling bug"

                # 2) Padded positions (attention_mask==0) should be ignored in labels
                assert torch.all(
                    labels.masked_select(attention_mask.eq(0)) == -100
                ), "padded tokens should have labels=-100"

            # Forward
            outputs = model(input_ids, attention_mask)
            logits = outputs.logits  # (B, T, V)

            # Per-token log-probabilities for the actual next token
            log_probs_all = torch.log_softmax(logits, dim=-1)
            safe_labels = torch.where(
                labels.eq(-100), torch.zeros_like(labels), labels
            )  # or pad_id
            policy_log_probs = log_probs_all.gather(
                dim=-1, index=safe_labels.unsqueeze(-1)
            ).squeeze(-1)

            # Response-token entropy (for logging)
            ent_bt = compute_entropy(logits).detach()
            resp_token_count_t = response_mask.sum().detach()
            resp_token_count = int(resp_token_count_t.clamp_min(1).item())
            mean_resp_entropy = (
                masked_normalize(ent_bt, response_mask, normalize_constant=float(resp_token_count))
                .detach()
                .item()
            )

            # Accumulate loss as SUM over response tokens for this microbatch
            # (normalize_constant=1.0 ensures a pure sum; grads will be normalized at update)
            _, meta = sft_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
            )

            # Running aggregates
            nonpad_token_count = int(attention_mask.sum().item())
            total_nonpad_tokens += nonpad_token_count
            nonpad_tokens_since_update += nonpad_token_count
            total_elements_since_update += int(attention_mask.numel())
            total_tokens += resp_token_count
            sum_entropy_since_update += mean_resp_entropy * resp_token_count
            sum_nll_since_update += float(meta["nll_unscaled"].item())
            tokens_since_update += resp_token_count

            if tokens_since_update >= min_tokens_per_update:
                # Opportunistic, non-blocking result drain during accumulation
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
                        logger.info(
                            "async vLLM (GPU1): step=%s avg_len=%.2f",
                            result.get("step", "?"),
                            avg_len,
                        )

                        result_step = int(result.get("step", update_step))
                        metrics = {
                            "steps/eval_step": int(result_step),
                            "eval/avg_response_length": float(
                                result.get("avg_response_length", 0.0)
                            ),
                            "eval/accuracy": float(result.get("accuracy", 0.0)),
                            "eval/truncation_rate": float(result.get("truncation_rate", 0.0)),
                            "eval/length_p50": float(result.get("length_p50", 0.0)),
                            "eval/length_p95": float(result.get("length_p95", 0.0)),
                            "eval/toks_per_sec": float(result.get("toks_per_sec", 0.0)),
                        }

                        lps = [
                            x
                            for x in result.get("avg_token_logprobs", [])
                            if x == x and not math.isinf(x)
                        ]
                        ppls = [
                            x
                            for x in result.get("perplexities", [])
                            if x == x and not math.isinf(x)
                        ]
                        reps = [
                            x for x in result.get("rep3_ratios", []) if x == x and not math.isinf(x)
                        ]
                        if lps:
                            metrics["eval/mean_gen_token_logprob"] = float(sum(lps) / len(lps))
                        if ppls:
                            metrics["eval/mean_gen_perplexity"] = float(sum(ppls) / len(ppls))
                        if reps:
                            metrics["eval/mean_rep3_ratio"] = float(sum(reps) / len(reps))
                        wandb.log(metrics)

                        # Add eval examples to persistent table
                        L = len(result.get("responses", []))
                        for j in range(L):
                            try:
                                eval_table.add_data(
                                    result.get("step", update_step),
                                    (result.get("prompts", []) or [""])[j],
                                    (result.get("gold_strs", []) or [""])[j],
                                    (result.get("responses", []) or [""])[j],
                                    (result.get("pred_answers", []) or [""])[j],
                                    (result.get("correct", []) or [0])[j],
                                    (result.get("lengths", []) or [0])[j],
                                    (result.get("truncated", []) or [0])[j],
                                    (result.get("stop_reasons", []) or [""])[j],
                                )
                            except Exception as e:
                                logger.warning("Failed to add eval row j=%d: %s", j, e)
                        if wandb.run is not None:
                            wandb.run.log({"eval/examples": eval_table})  # type: ignore[dict-item]

                # Early stop if update limit reached (will flush below if needed)
                if max_update_steps is not None and update_step >= max_update_steps:
                    break

                denom = float(tokens_since_update)
                # Normalize accumulated gradients by total loss-bearing tokens
                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.div_(denom)

                # Compute global grad L2 norm (pre-clip)
                global_grad_l2_preclip = 0.0
                with torch.no_grad():
                    sq_sum = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            # cast to float32 for numerical stability
                            sq_sum += p.grad.detach().float().pow(2).sum().item()
                    global_grad_l2_preclip = float(sq_sum**0.5)

                # Clip (if enabled) and compute post-clip norm
                global_grad_l2_postclip = global_grad_l2_preclip
                if max_grad_norm is not None and max_grad_norm > 0:
                    clip_grad_norm_(model.parameters(), max_grad_norm)
                    with torch.no_grad():
                        sq_sum_post = 0.0
                        for p in model.parameters():
                            if p.grad is not None:
                                sq_sum_post += p.grad.detach().float().pow(2).sum().item()
                        global_grad_l2_postclip = float(sq_sum_post**0.5)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                update_step += 1
                try:
                    scheduler.step()
                except Exception as e:
                    logger.warning("LR scheduler step failed: %s", e)
                current_lr = float(optimizer.param_groups[0]["lr"])

                # Log train metrics
                elapsed = time.perf_counter() - start_time
                tps_response = total_tokens / max(elapsed, 1e-9)
                tps_nonpad = total_nonpad_tokens / max(elapsed, 1e-9)
                avg_nll_update = sum_nll_since_update / max(tokens_since_update, 1.0)
                avg_lp_update = -avg_nll_update
                mean_resp_entropy_update = sum_entropy_since_update / max(tokens_since_update, 1.0)
                mean_ppl_update = float(torch.exp(torch.tensor(-avg_lp_update)).item())

                # Pad percentage per macrobatch (i.e. optimizer step)
                pad_tokens_since_update = max(
                    total_elements_since_update - nonpad_tokens_since_update, 0
                )
                pct_pad_per_macrobatch = (
                    pad_tokens_since_update / max(total_elements_since_update, 1)
                ) * 100.0
                logger.info(
                    (
                        "update=%d epoch=%d toks=%d nonpad_toks=%d "
                        "mean_lp=%.4f mean_nll=%.4f mean_ent=%.4f mean_ppl=%.4f lr=%.2e "
                        "tps_resp=%.1f tps_nonpad=%.1f"
                    ),
                    update_step,
                    epoch + 1,
                    int(tokens_since_update),
                    int(nonpad_tokens_since_update),
                    avg_lp_update,
                    avg_nll_update,
                    mean_resp_entropy_update,
                    mean_ppl_update,
                    current_lr,
                    tps_response,
                    tps_nonpad,
                )

                wandb.log(
                    {
                        "steps/train_step": int(update_step),
                        "train/pct_pad_per_macrobatch": float(pct_pad_per_macrobatch),
                        "train/tokens_per_sec_response": float(tps_response),
                        "train/tokens_per_sec_nonpad": float(tps_nonpad),
                        "train/avg_response_token_logprob": float(avg_lp_update),
                        "train/avg_response_token_nll": float(avg_nll_update),
                        "train/mean_entropy_response": float(mean_resp_entropy_update),
                        "train/mean_perplexity_response": float(mean_ppl_update),
                        "train/response_tokens_update": float(tokens_since_update),
                        "train/global_grad_l2_preclip": float(global_grad_l2_preclip),
                        "train/global_grad_l2_postclip": float(global_grad_l2_postclip),
                        "train/lr": float(current_lr),
                    },
                )

                # --- Async generation enqueue ---
                if eval_every and update_step % eval_every == 0:
                    kgen = -1
                    if eval_examples:
                        kgen = max(1, min(int(eval_examples), len(eval_prompts_chat)))

                    prompts_for_eval = eval_prompts_chat[:kgen]
                    gold_nums_for_eval = eval_gold_nums[:kgen]
                    gold_strs_for_eval = eval_gold_strs[:kgen]

                    # Save lightweight checkpoint and enqueue a job
                    try:
                        ckpt_dir_vllm = save_policy_checkpoint_for_vllm(
                            model,
                            update_step,
                            out_root="/dev/shm",
                            base="qwen15b_step",
                            dtype=torch.bfloat16,
                        )
                        logger.info("Eval trigger: wrote vLLM ckpt to %s", ckpt_dir_vllm)
                    except Exception:
                        logger.exception(
                            "Eval trigger: failed to write vLLM ckpt at step %d", update_step
                        )
                        ckpt_dir_vllm = None

                    # Enqueue only if we have a ckpt AND a worker queue
                    if ckpt_dir_vllm is not None and jobs_q is not None:
                        payload = {
                            "ckpt_dir": str(ckpt_dir_vllm),
                            "prompts": prompts_for_eval,
                            "answers": gold_nums_for_eval,  # numeric golds for exact-match
                            "step": int(update_step),
                            "max_new_tokens": int(gen_max_new_tokens),
                            "temperature": float(gen_temperature),
                            "top_p": float(gen_top_p),
                            "gold_strs": gold_strs_for_eval,
                        }
                        try:
                            jobs_q.put(payload, block=True, timeout=5.0)
                            outstanding_jobs += 1
                            logger.info("Enqueued async eval for step %d (n=%d)", update_step, kgen)
                        except Exception as e:
                            logger.warning("Failed to enqueue async eval: %s", e)

                    # Save full HF checkpoint aligned with eval
                    if checkpoint_dir is not None:
                        try:
                            ckpt_root = Path(checkpoint_dir)
                            hf_ckpt_path = ckpt_root / f"step_{update_step}"
                            hf_ckpt_path.mkdir(parents=True, exist_ok=True)
                            model.save_pretrained(hf_ckpt_path)
                            tok.save_pretrained(hf_ckpt_path)
                            logger.info("Saved full HF checkpoint to %s", hf_ckpt_path)
                        except Exception:
                            logger.warning(
                                "Failed to save full HF checkpoint at step %d", update_step
                            )

                # --- Reset accumulators ---
                tokens_since_update = 0
                sum_nll_since_update = 0.0
                sum_entropy_since_update = 0.0
                nonpad_tokens_since_update = 0
                total_elements_since_update = 0

            # Stop condition in update units
            if max_update_steps is not None and update_step >= max_update_steps:
                break

        if max_update_steps is not None and update_step >= max_update_steps:
            break

    # --- Graceful shutdown of async eval worker & queues ---
    await_async_eval_at_end = True  # True = wait briefly for last job
    try:
        # If requested, wait for outstanding eval jobs to finish (bounded time)
        if await_async_eval_at_end and jobs_q is not None and results_q is not None:
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

                    result_step = int(result.get("step", update_step))
                    metrics = {
                        "steps/eval_step": int(result_step),
                        "eval/avg_response_length": float(result.get("avg_response_length", 0.0)),
                        "eval/accuracy": float(result.get("accuracy", 0.0)),
                        "eval/truncation_rate": float(result.get("truncation_rate", 0.0)),
                        "eval/length_p50": float(result.get("length_p50", 0.0)),
                        "eval/length_p95": float(result.get("length_p95", 0.0)),
                        "eval/toks_per_sec": float(result.get("toks_per_sec", 0.0)),
                    }

                    lps = [
                        x
                        for x in result.get("avg_token_logprobs", [])
                        if x == x and not math.isinf(x)
                    ]
                    ppls = [
                        x for x in result.get("perplexities", []) if x == x and not math.isinf(x)
                    ]
                    reps = [
                        x for x in result.get("rep3_ratios", []) if x == x and not math.isinf(x)
                    ]
                    if lps:
                        metrics["eval/mean_gen_token_logprob"] = float(sum(lps) / len(lps))
                    if ppls:
                        metrics["eval/mean_gen_perplexity"] = float(sum(ppls) / len(ppls))
                    if reps:
                        metrics["eval/mean_rep3_ratio"] = float(sum(reps) / len(reps))
                    wandb.log(metrics)

                    # Add final eval examples to persistent table
                    L = len(result.get("responses", []))
                    for j in range(L):
                        try:
                            eval_table.add_data(
                                result.get("step", update_step),
                                (result.get("prompts", []) or [""])[j],
                                (result.get("gold_strs", []) or [""])[j],
                                (result.get("responses", []) or [""])[j],
                                (result.get("pred_answers", []) or [""])[j],
                                (result.get("correct", []) or [0])[j],
                                (result.get("lengths", []) or [0])[j],
                                (result.get("truncated", []) or [0])[j],
                                (result.get("stop_reasons", []) or [""])[j],
                            )
                        except Exception as e:
                            logging.warning("Failed to add eval row j=%d: %s", j, e)
                    if wandb.run is not None:
                        wandb.run.log({"eval/examples": eval_table})  # type: ignore[dict-item]

            # Tell the worker to exit once we're done waiting
            try:
                jobs_q.put_nowait(None)
            except Exception:
                pass

        # Join worker with a generous timeout; only hard-kill if it still won't exit
        if vllm_proc is not None:
            vllm_proc.join(timeout=60.0)
            if vllm_proc.is_alive():
                logger.warning("vllm worker still alive after timeout; terminating...")
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
