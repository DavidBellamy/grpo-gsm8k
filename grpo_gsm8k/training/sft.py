from __future__ import annotations

import glob
import logging
import math
import os
import queue
import time
from multiprocessing import Queue, get_context
from multiprocessing.context import SpawnProcess
from pathlib import Path

import torch
import wandb
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from grpo_gsm8k.core.masked_normalize import masked_normalize
from grpo_gsm8k.core.per_token_entropy import compute_entropy
from grpo_gsm8k.data.data_loader import BucketMicrobatcher, PTShardStream
from grpo_gsm8k.training.utils import (
    ensure_pad_token,
    load_templated_gsm8k,
    resolve_resume_path,
    sanitize_wandb_component,
    save_policy_checkpoint_for_vllm,
    vllm_worker_persistent,
)

logger = logging.getLogger(__name__)


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


# Main training loop
def train_sft_on_r1_pairs(
    train_data_path: str | Path,
    val_data_path: str | Path | None = None,
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
    sanitized_model = sanitize_wandb_component(model_id)
    train_title = f"{sanitized_model}-sft-train"
    wandb.define_metric(train_title + "/*", step_metric="steps/train_step")
    wandb.define_metric("steps/val_step")
    val_title = f"{sanitized_model}-sft-val"
    wandb.define_metric(val_title + "/*", step_metric="steps/val_step")

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
        ckpt_path = resolve_resume_path(resume_from)
        logger.info("Resuming from checkpoint at %s", ckpt_path)
        load_path = ckpt_path

    logger.info("Loading model from %s", load_path)
    tok = AutoTokenizer.from_pretrained(load_path, use_fast=True)
    ensure_pad_token(tok)
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
        if val_data_path is None:
            raise ValueError(
                "eval_every=%d but val_data_path is not provided.",
                eval_every,
            )
        else:
            eval_prompts_chat, eval_gold_strs, eval_gold_nums = load_templated_gsm8k(val_data_path)
            logger.info(
                "Loaded pre-rendered eval set for vLLM: n=%d from %s",
                len(eval_prompts_chat),
                val_data_path,
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
        target=vllm_worker_persistent,
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
        "val_step",
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
                            "steps/val_step": int(result_step),
                            f"{val_title}/avg_response_length": float(
                                result.get("avg_response_length", 0.0)
                            ),
                            f"{val_title}/accuracy": float(result.get("accuracy", 0.0)),
                            f"{val_title}/truncation_rate": float(
                                result.get("truncation_rate", 0.0)
                            ),
                            f"{val_title}/length_p50": float(result.get("length_p50", 0.0)),
                            f"{val_title}/length_p95": float(result.get("length_p95", 0.0)),
                            f"{val_title}/toks_per_sec": float(result.get("toks_per_sec", 0.0)),
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
                            metrics[f"{val_title}/mean_gen_token_logprob"] = float(
                                sum(lps) / len(lps)
                            )
                        if ppls:
                            metrics[f"{val_title}/mean_gen_perplexity"] = float(
                                sum(ppls) / len(ppls)
                            )
                        if reps:
                            metrics[f"{val_title}/mean_rep3_ratio"] = float(sum(reps) / len(reps))
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
                            wandb.run.log({f"{val_title}/examples": eval_table})  # type: ignore[dict-item]

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
                        f"{train_title}/pct_pad_per_macrobatch": float(pct_pad_per_macrobatch),
                        f"{train_title}/tokens_per_sec_response": float(tps_response),
                        f"{train_title}/tokens_per_sec_nonpad": float(tps_nonpad),
                        f"{train_title}/avg_response_token_logprob": float(avg_lp_update),
                        f"{train_title}/avg_response_token_nll": float(avg_nll_update),
                        f"{train_title}/mean_entropy_response": float(mean_resp_entropy_update),
                        f"{train_title}/mean_perplexity_response": float(mean_ppl_update),
                        f"{train_title}/response_tokens_update": float(tokens_since_update),
                        f"{train_title}/global_grad_l2_preclip": float(global_grad_l2_preclip),
                        f"{train_title}/global_grad_l2_postclip": float(global_grad_l2_postclip),
                        f"{train_title}/lr": float(current_lr),
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
                            logger=logger,
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
                        "steps/val_step": int(result_step),
                        f"{val_title}/avg_response_length": float(
                            result.get("avg_response_length", 0.0)
                        ),
                        f"{val_title}/accuracy": float(result.get("accuracy", 0.0)),
                        f"{val_title}/truncation_rate": float(result.get("truncation_rate", 0.0)),
                        f"{val_title}/length_p50": float(result.get("length_p50", 0.0)),
                        f"{val_title}/length_p95": float(result.get("length_p95", 0.0)),
                        f"{val_title}/toks_per_sec": float(result.get("toks_per_sec", 0.0)),
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
                        metrics[f"{val_title}/mean_gen_token_logprob"] = float(sum(lps) / len(lps))
                    if ppls:
                        metrics[f"{val_title}/mean_gen_perplexity"] = float(sum(ppls) / len(ppls))
                    if reps:
                        metrics[f"{val_title}/mean_rep3_ratio"] = float(sum(reps) / len(reps))
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
                        wandb.run.log({f"{val_title}/examples": eval_table})  # type: ignore[dict-item]

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
