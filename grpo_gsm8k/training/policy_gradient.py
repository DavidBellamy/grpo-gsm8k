import logging
import math
import queue as pyqueue
import statistics as stats
import time
from collections.abc import Callable
from multiprocessing import Queue, get_context
from multiprocessing.context import SpawnProcess
from pathlib import Path
from typing import Any, Literal

import torch
import wandb
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from grpo_gsm8k.training.utils import (
    RunningStats,
    default_reward,
    ensure_pad_token,
    load_templated_gsm8k,
    resolve_resume_path,
    sanitize_wandb_component,
    save_policy_checkpoint_for_vllm,
    vllm_worker_persistent,
)

logger = logging.getLogger(__name__)


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float = 1e-8,
    normalize_by_std: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert len(rollout_responses) == len(repeated_ground_truths)
    n = len(rollout_responses)
    assert group_size > 0 and n % group_size == 0

    # Score
    raw_list: list[float] = []
    fmt_list: list[float] = []
    ans_list: list[float] = []
    for gt, resp in zip(repeated_ground_truths, rollout_responses):
        out = reward_fn(gt, resp)
        raw_list.append(float(out["reward"]))
        fmt_list.append(float(out["format_reward"]))
        ans_list.append(float(out["answer_reward"]))

    # Group-normalize
    adv_list = [0.0] * n
    num_groups = n // group_size
    for g in range(num_groups):
        start = g * group_size
        end = start + group_size
        grp = raw_list[start:end]
        mu = stats.mean(grp)
        if normalize_by_std:
            sd = stats.pstdev(grp)
            denom = sd if sd > advantage_eps else advantage_eps
            adv_list[start:end] = [(x - mu) / denom for x in grp]
        else:
            adv_list[start:end] = [x - mu for x in grp]

    advantages = torch.tensor(adv_list, dtype=torch.float32)
    raw_rewards = torch.tensor(raw_list, dtype=torch.float32)
    fmt_rewards = torch.tensor(fmt_list, dtype=torch.float32)
    ans_rewards = torch.tensor(ans_list, dtype=torch.float32)
    return advantages, raw_rewards, fmt_rewards, ans_rewards


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the policy-gradient loss at every token, where raw_rewards_or_advantages is either
    the raw reward or an already-normalized advantage.
    """
    return -raw_rewards_or_advantages * policy_log_probs


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    importance_wts = torch.exp(policy_log_probs - old_log_probs)
    clipped_wts = torch.clamp(importance_wts, 1.0 - cliprange, 1.0 + cliprange)

    lhs = importance_wts * advantages
    rhs = clipped_wts * advantages
    metadata = {
        "importance_wts": importance_wts.detach(),
        "clipped_wts": clipped_wts.detach(),
        "was_clipped": (importance_wts.ne(clipped_wts)).detach().to(torch.int),
        "min_is_rhs": (rhs < lhs).detach().to(torch.int),
    }

    return -torch.min(lhs, rhs), metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    metadata: dict[str, torch.Tensor] = {}

    assert loss_type in ["no_baseline", "reinforce_with_baseline", "grpo_clip"]

    if loss_type == "no_baseline":
        if raw_rewards is None:
            raise ValueError("raw_rewards required for loss_type='no_baseline'")
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
    elif loss_type == "reinforce_with_baseline":
        if advantages is None:
            raise ValueError("advantages required for loss_type='reinforce_with_baseline'")
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
    else:  # grpo_clip
        if advantages is None or old_log_probs is None or cliprange is None:
            raise ValueError("advantages, old_log_probs, cliprange required for 'grpo_clip'")
        loss, metadata = compute_grpo_clip_loss(
            advantages, policy_log_probs, old_log_probs, cliprange
        )

    return loss, metadata


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    wt_sum = (tensor * mask).sum(dim=dim)
    count = mask.sum(dim=dim)
    assert torch.all(count > 0)
    return wt_sum / count


def policy_gradient_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    episodes_per_update: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    assert policy_log_probs.ndim == 2 and response_mask.ndim == 2, "Expect (B, T) tensors"
    B, T = policy_log_probs.shape
    assert response_mask.shape == (B, T), "mask shape must match log_probs"

    per_token_loss, extrameta = compute_policy_gradient_loss(
        policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange
    )

    # Weight each episode equally in the loss (independent of length)
    loss_per_example = masked_mean(per_token_loss, response_mask, dim=1)
    loss_sum = loss_per_example.sum()
    scaled_loss = loss_sum / max(1, episodes_per_update)
    scaled_loss.backward()

    with torch.no_grad():
        token_count = response_mask.sum()
        mean_neg_logp = -(policy_log_probs * response_mask).sum() / token_count.clamp_min(1.0)
        meta: dict[str, torch.Tensor] = {
            "tokens": token_count.detach(),
            "mean_neg_logp": mean_neg_logp.detach(),
            "mean_seq_len": (response_mask.sum(dim=1).float().mean()).detach(),
        }
        if advantages is not None:
            meta["mean_advantage"] = advantages.mean().detach()
            meta["std_advantage"] = advantages.float().std(unbiased=False).detach()
        meta.update(
            {k: v if isinstance(v, torch.Tensor) else torch.tensor(v) for k, v in extrameta.items()}
        )
    return scaled_loss, meta


def train_policy_gradient(
    train_data_path: str | Path,  # JSONL with {"prompt","gold","gold_num"}
    val_data_path: str | Path | None = None,  # same pre-rendered format, for async eval
    *,
    model_id: str = "Qwen/Qwen2.5-Math-1.5B",
    device: str | torch.device = "cuda:0",
    vllm_device: str | torch.device = "cuda:1",
    vllm_gpu_memory_utilization: float = 0.85,
    vllm_prompts_per_batch: int = 64,
    # Rollout / grouping
    group_size: int = 8,  # K samples per prompt (GRPO group)
    episodes_per_update: int = 256,  # target episodes before each optimizer.step()
    soft_token_cap_per_update: int | None = 1_000_000,
    trainer_episodes_per_mb: int = 8,
    max_new_tokens: int = 2048,  # generation length cap (action tokens)
    temperature: float = 1.0,
    top_p: float = 1.0,
    # Optimizer / schedule
    learning_rate: float = 1e-5,
    adamw_beta1: float = 0.9,
    adamw_beta2: float = 0.95,
    adamw_eps: float = 1e-8,
    weight_decay: float = 0.0,
    max_grad_norm: float | None = 1.0,
    total_update_steps: int = 200,
    # Policy gradient loss type
    loss_type: Literal[
        "no_baseline", "reinforce_with_baseline", "grpo_clip"
    ] = "reinforce_with_baseline",
    cliprange: float = 0.2,
    normalize_adv_by_std: bool = True,
    advantage_eps: float = 1e-6,
    # Misc
    eval_every: int = 20,
    eval_examples: int | None = None,
    checkpoint_dir: str | Path | None = None,  # full HF ckpt aligned to evals
    model_dtype: torch.dtype | None = None,
    resume_from: str | Path | None = None,
    # Reward fn: maps (gold_str, response_text) -> {"reward","format_reward","answer_reward"}
    reward_fn: Callable[[str, str], dict[str, float]] | None = None,
    entropy_topk: int | None = 32,
) -> None:
    """
    Policy gradient-style training with:
      - on-policy rollouts from a persistent vLLM worker (GPU1)
      - For GRPO, group-normalized rewards over K samples per prompt
      - example-weighted policy-gradient reduction
      - optimizer step after accumulating a target number of episodes
    """
    assert episodes_per_update % group_size == 0, "group_size must divide episodes_per_update."

    reward: Callable[[str, str], dict[str, float]]
    if reward_fn is None:
        reward = default_reward
    else:
        reward = reward_fn

    wandb.define_metric("steps/train_step")
    wandb.define_metric("steps/val_step")
    sanitized_model = sanitize_wandb_component(model_id)
    train_title = f"{sanitized_model}-policygradient-{loss_type}-train"
    val_title = f"{sanitized_model}-policygradient-{loss_type}-val"
    wandb.define_metric(train_title + "/*", step_metric="steps/train_step")
    wandb.define_metric(val_title + "/*", step_metric="steps/val_step")

    if isinstance(device, str):
        device_t = torch.device(device)
    elif isinstance(device, torch.device):
        device_t = device
    else:
        raise ValueError("device must be a string (e.g., 'cuda:0') or a torch.device")

    load_path: str | Path = model_id
    if resume_from is not None:
        load_path = resolve_resume_path(resume_from)
        logger.info("Resuming from %s", load_path)

    tok = AutoTokenizer.from_pretrained(load_path, use_fast=True)
    ensure_pad_token(tok)
    tok.padding_side = "left"

    if model_dtype is None:
        if device_t.type == "cuda":
            model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            model_dtype = torch.float32

    policy = AutoModelForCausalLM.from_pretrained(load_path, torch_dtype=model_dtype)
    policy.config.pad_token_id = tok.pad_token_id
    policy.config.use_cache = False  # forward is only for logprobs; generation is via vLLM
    gc = getattr(policy, "generation_config", None)
    if gc is not None:
        gc.pad_token_id = tok.pad_token_id
    policy.to(device_t)
    policy.train()

    # Async vLLM Worker (GPU1)
    ctx = get_context("spawn")
    jobs_q: Queue = ctx.Queue(maxsize=64)
    results_q: Queue = ctx.Queue(maxsize=64)
    gpu_id = str(vllm_device).replace("cuda:", "")
    vllm_proc: SpawnProcess = ctx.Process(
        target=vllm_worker_persistent,
        args=(jobs_q, results_q),
        kwargs=dict(
            base_model_id=model_id,
            gpu_id=gpu_id,
            gpu_memory_utilization=float(vllm_gpu_memory_utilization),
            dtype="bfloat16",
        ),
        daemon=False,
    )
    vllm_proc.start()
    logger.info("Started vLLM rollout worker on GPU %s (pid=%s)", gpu_id, vllm_proc.pid)

    # Optimizer & Scheduler
    decay_params, no_decay_params = [], []
    for name, p in policy.named_parameters():
        if not p.requires_grad:
            continue
        n = name.lower()
        if (
            p.ndim == 1
            or name.endswith(".bias")
            or "layernorm" in n
            or "layer_norm" in n
            or "embed" in n
        ):
            no_decay_params.append(p)
        else:
            decay_params.append(p)

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

    # Cosine schedule with short warmup
    warmup = max(50, int(0.02 * total_update_steps))
    final_scale = 0.10

    def _lambda(s: int) -> float:
        s = s + 1
        if s <= warmup:
            return s / max(1, warmup)
        if s >= total_update_steps:
            return final_scale
        span = total_update_steps - warmup
        prog = (s - warmup) / max(1, span)
        return final_scale + (1 - final_scale) * 0.5 * (1 + math.cos(math.pi * prog))

    scheduler = LambdaLR(optimizer, lr_lambda=_lambda)

    # Load pre-rendered prompts for rollouts
    train_prompts_chat, _, train_gold_nums = load_templated_gsm8k(train_data_path)

    # Optionally load val set
    eval_prompts_chat: list[str] = []
    eval_gold_strs: list[str] = []
    eval_gold_nums: list[str] = []
    if eval_every and val_data_path is not None:
        eval_prompts_chat, eval_gold_strs, eval_gold_nums = load_templated_gsm8k(val_data_path)
        logger.info("Loaded val set for async eval: n=%d", len(eval_prompts_chat))

    eval_table = wandb.Table(
        columns=[
            "val_step",
            "prompt",
            "gold_reasoning",
            "model_response",
            "model_answer",
            "correct",
            "response_length",
            "truncated",
            "stop_reason",
        ],
        log_mode="INCREMENTAL",
    )

    # Helpers
    def _tokenize_concat_build_mask(
        prompts: list[str], responses: list[str]
    ) -> dict[str, torch.Tensor]:
        """Tokenize [prompt || response]; build response_mask over the response span only."""
        enc_prompt = tok(prompts, add_special_tokens=False, return_tensors=None)
        enc_full = tok(
            [p + r for p, r in zip(prompts, responses)],
            add_special_tokens=False,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc_full["input_ids"]  # (B, T)
        attention_mask = enc_full["attention_mask"]  # (B, T)

        # Compute response spans
        prompt_lens = [len(x) for x in enc_prompt["input_ids"]]  # per-example
        B, T = input_ids.shape
        resp_mask = torch.zeros((B, T), dtype=torch.long)
        for i, pl in enumerate(prompt_lens):
            # The final length after padding is attention_mask[i].sum()
            seq_len = int(attention_mask[i].sum().item())
            # response tokens are positions [pl : seq_len)
            start = min(pl, seq_len)
            resp_mask[i, start:seq_len] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_mask": resp_mask,
        }

    def _rollout_with_vllm(
        step: int, prompts: list[str], golds: list[str], K: int, ckpt_dir_vllm: str | Path
    ) -> dict[str, Any]:
        """Replicate prompts K times; synchronous round-trip with the worker."""
        # Build repeated lists so groups are contiguous
        prom_repeated = [p for p in prompts for _ in range(K)]
        gold_repeated = [g for g in golds for _ in range(K)]
        payload = {
            "ckpt_dir": str(ckpt_dir_vllm),
            "prompts": prom_repeated,
            "answers": gold_repeated,
            "step": int(step),
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "gold_strs": gold_repeated,
        }
        jobs_q.put(payload, block=True)
        # Synchronously wait for this rollout batch
        result = results_q.get(block=True)
        if "error" in result:
            raise RuntimeError(f"vLLM rollout failed: {result['error']}")
        return result

    # Main Loop
    update_step = 0
    start_time = time.perf_counter()
    ptr = 0  # pointer into training prompts
    N = len(train_prompts_chat)
    assert N > 0, "No training prompts"

    while update_step < total_update_steps:
        # send policy weights to vllm worker
        ckpt_dir_vllm = save_policy_checkpoint_for_vllm(
            policy,
            update_step,
            out_root="/dev/shm",
            base="qwen15b_step",
            dtype=torch.bfloat16,
            logger=logger,
        )
        # ---- collect rollouts up to the target update size ----
        ep_since = 0
        toks_since = 0
        sum_entropy = 0.0
        sum_ratio_clipped = 0.0
        sum_ratio = 0.0
        clip_frac_cnt = 0
        adv_stats = RunningStats()
        rew_stats = RunningStats()
        fmt_stats = RunningStats()
        ans_stats = RunningStats()

        while ep_since < episodes_per_update and (
            soft_token_cap_per_update is None or toks_since < soft_token_cap_per_update
        ):
            # Select a chunk of unique prompts; each will be replicated K times in the worker
            episodes_left = episodes_per_update - ep_since
            prompts_left = max(1, episodes_left // group_size)
            take_prompts = min(vllm_prompts_per_batch, prompts_left)

            if ptr >= N:
                ptr = 0  # wrap-around epoching
            end = min(N, ptr + take_prompts)
            prompts_mb = train_prompts_chat[ptr:end]
            golds_mb = train_gold_nums[ptr:end]
            ptr = end
            if not prompts_mb:
                break

            # Rollout via vLLM worker
            result = _rollout_with_vllm(
                update_step, prompts_mb, golds_mb, group_size, ckpt_dir_vllm
            )
            responses = result["responses"]  # length = len(prompts_mb) * group_size
            gold_repeated = result["answers"]

            # Rewards & (group) advantages
            adv, raw, fmt, ans = compute_group_normalized_rewards(
                reward,
                responses,
                gold_repeated,
                group_size,
                advantage_eps=advantage_eps,
                normalize_by_std=normalize_adv_by_std,
            )
            adv_stats.update_batch(adv)
            rew_stats.update_batch(raw)
            fmt_stats.update_batch(fmt)
            ans_stats.update_batch(ans)

            # Tokenize concatenated prompts + responses
            prompts_rep = [p for p in prompts_mb for _ in range(group_size)]
            pack = _tokenize_concat_build_mask(prompts_rep, responses)
            input_ids_full = pack["input_ids"]
            attn_full = pack["attention_mask"]
            resp_mask_full = pack["response_mask"]

            # B is the number of episodes *in one vLLM batch*
            B = input_ids_full.size(0)

            # Make teacher-forcing labels
            next_ids_full = torch.roll(input_ids_full, shifts=-1, dims=1)
            next_ids_full[:, -1] = tok.pad_token_id
            label_mask = resp_mask_full.bool() & attn_full.bool()  # (B, T)
            labels_full = input_ids_full.new_full(input_ids_full.shape, -100)
            labels_full[label_mask] = next_ids_full[label_mask]

            with tqdm(
                total=B,
                desc=f"episodes (upd {update_step})",
                unit="ep",
                leave=False,
            ) as pbar:
                for s in range(0, B, trainer_episodes_per_mb):
                    e = min(B, s + trainer_episodes_per_mb)

                    # slice vLLM rollouts
                    input_ids = input_ids_full[s:e]
                    attn = attn_full[s:e]
                    resp_mask = resp_mask_full[s:e]
                    labels = labels_full[s:e]

                    # move slices to GPU
                    input_ids = input_ids.to(device_t, non_blocking=True)
                    attn = attn.to(device_t, non_blocking=True)
                    resp_mask = resp_mask.to(device_t, non_blocking=True)
                    labels = labels.to(device_t, non_blocking=True)

                    # Per-token NLL without materializing softmax
                    logits = policy(input_ids, attn).logits  # (b,t,V)
                    import torch.nn.functional as F

                    nll = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1),
                        reduction="none",
                        ignore_index=-100,  # ignore prompt tokens
                    ).view_as(labels)

                    policy_logp = -nll  # (b, t)
                    old_logp = policy_logp.detach()  # for proximal updates e.g. GRPO-clip

                    # update tallies
                    toks = int(resp_mask.sum().item())
                    ep_since += policy_logp.size(0)
                    toks_since += toks

                    # Entropy top-k for monitoring (no grad)
                    if entropy_topk and entropy_topk > 0:
                        with torch.no_grad():
                            vals, _ = logits.topk(entropy_topk, dim=-1)
                            lse = vals.logsumexp(dim=-1, keepdim=True)
                            p = (vals - lse).exp()
                            ent = -(p * (vals - lse)).sum(dim=-1)  # (b,t)
                            sum_entropy += (
                                float(masked_mean(ent, resp_mask.to(ent.dtype)).item()) * toks
                            )

                    # choose advantage vs. reward slice for this chunk
                    adv_chunk = None
                    rew_chunk = None
                    if loss_type in ("reinforce_with_baseline", "grpo_clip"):
                        adv_chunk = (
                            adv[s:e].to(device_t, dtype=policy_logp.dtype).unsqueeze(-1)
                        )  # (b,1)
                    if loss_type == "no_baseline":
                        rew_chunk = (
                            raw[s:e].to(device_t, dtype=policy_logp.dtype).unsqueeze(-1)
                        )  # (b,1)

                    # one backward per chunk (grad accumulation)
                    _, meta = policy_gradient_microbatch_train_step(
                        policy_log_probs=policy_logp,  # (b,t)
                        response_mask=resp_mask,  # (b,t)
                        episodes_per_update=episodes_per_update,
                        loss_type=loss_type,
                        raw_rewards=rew_chunk,
                        advantages=adv_chunk,
                        old_log_probs=old_logp,  # (b,t)
                        cliprange=cliprange,
                    )

                    # logging pieces
                    if "importance_wts" in meta:
                        w = meta["importance_wts"]
                        cw = meta.get("clipped_wts", None)
                        sum_ratio += float(w.mean().item())
                        if cw is not None:
                            sum_ratio_clipped += float((w.ne(cw)).float().mean().item())
                            clip_frac_cnt += 1

                    # free big tensors ASAP
                    del logits, nll, policy_logp, old_logp, input_ids, attn, labels, resp_mask

                    pbar.update(e - s)

            del adv, raw, fmt, ans

        # Optimizer Step

        # Rescale gradients by the actual number of episodes
        if 0 < ep_since != episodes_per_update:
            scale = episodes_per_update / ep_since
            for p in policy.parameters():
                if p.grad is not None:
                    p.grad.mul_(scale)

        # Gradient clipping (turn off in config with max_grad_norm = null)
        global_grad_l2_preclip = 0.0
        with torch.no_grad():
            sq = 0.0
            for p in policy.parameters():
                if p.grad is not None:
                    sq += p.grad.detach().float().pow(2).sum().item()
            global_grad_l2_preclip = float(sq**0.5)

        global_grad_l2_postclip = global_grad_l2_preclip
        if max_grad_norm is not None and max_grad_norm > 0:
            clip_grad_norm_(policy.parameters(), max_grad_norm)
            with torch.no_grad():
                sq = 0.0
                for p in policy.parameters():
                    if p.grad is not None:
                        sq += p.grad.detach().float().pow(2).sum().item()
                global_grad_l2_postclip = float(sq**0.5)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update_step += 1
        try:
            scheduler.step()
        except Exception as e:
            logger.warning("LR scheduler step failed: %s", e)
        current_lr = float(optimizer.param_groups[0]["lr"])

        # Logging
        elapsed = time.perf_counter() - start_time
        tps = toks_since / max(elapsed, 1e-9)
        mean_ent_update = (sum_entropy / max(toks_since, 1.0)) if toks_since else 0.0
        a = adv_stats.finalize()
        r = rew_stats.finalize()
        f = fmt_stats.finalize()
        g = ans_stats.finalize()
        metrics = {
            "steps/train_step": int(update_step),
            f"{train_title}/episodes_update": int(ep_since),
            f"{train_title}/tokens_update": int(toks_since),
            f"{train_title}/tokens_per_sec": float(tps),
            f"{train_title}/mean_entropy_response": float(mean_ent_update),
            f"{train_title}/lr": float(current_lr),
            f"{train_title}/global_grad_l2_preclip": float(global_grad_l2_preclip),
            f"{train_title}/global_grad_l2_postclip": float(global_grad_l2_postclip),
            f"{train_title}/adv_mean": a["mean"],
            f"{train_title}/adv_min": a["min"],
            f"{train_title}/adv_max": a["max"],
            f"{train_title}/adv_std": a["std"],
            f"{train_title}/reward_mean": r["mean"],
            f"{train_title}/reward_min": r["min"],
            f"{train_title}/reward_max": r["max"],
            f"{train_title}/reward_std": r["std"],
            f"{train_title}/fmt_reward_mean": f["mean"],
            f"{train_title}/fmt_reward_min": f["min"],
            f"{train_title}/fmt_reward_max": f["max"],
            f"{train_title}/fmt_reward_std": f["std"],
            f"{train_title}/ans_reward_mean": g["mean"],
            f"{train_title}/ans_reward_min": g["min"],
            f"{train_title}/ans_reward_max": g["max"],
            f"{train_title}/ans_reward_std": g["std"],
        }
        if loss_type == "grpo_clip" and clip_frac_cnt > 0:
            metrics[f"{train_title}/wt_mean"] = float(sum_ratio / max(1, clip_frac_cnt))
            metrics[f"{train_title}/wt_clip_frac"] = float(
                sum_ratio_clipped / max(1, clip_frac_cnt)
            )
        wandb.log(metrics)

        # Periodic async eval
        if eval_every and (update_step % eval_every == 0) and val_data_path is not None:
            kgen = -1
            if eval_examples:
                kgen = max(1, min(int(eval_examples), len(eval_prompts_chat)))
            prompts_for_eval = eval_prompts_chat[:kgen]
            gold_nums_for_eval = eval_gold_nums[:kgen]
            gold_strs_for_eval = eval_gold_strs[:kgen]
            try:
                payload = {
                    "ckpt_dir": str(ckpt_dir_vllm),
                    "prompts": prompts_for_eval,
                    "answers": gold_nums_for_eval,
                    "step": int(update_step),
                    "max_new_tokens": int(max_new_tokens),
                    "temperature": 0.0,  # greedy for eval
                    "top_p": 1.0,
                    "gold_strs": gold_strs_for_eval,
                }
                jobs_q.put(payload, block=True, timeout=5.0)

                # Save full HF checkpoint aligned with eval
                if checkpoint_dir is not None:
                    try:
                        ckpt_root = Path(checkpoint_dir)
                        hf_ckpt_path = ckpt_root / f"step_{update_step}"
                        hf_ckpt_path.mkdir(parents=True, exist_ok=True)
                        policy.save_pretrained(hf_ckpt_path)
                        tok.save_pretrained(hf_ckpt_path)
                        logger.info(f"Saved full HF checkpoint to {hf_ckpt_path}")
                    except Exception:
                        logger.warning(f"Failed to save full HF checkpoint at step {update_step}")

                # Non-blocking drain of exactly one eval result
                try:
                    result = results_q.get(timeout=60.0)
                    if "error" in result:
                        logger.warning("async eval error: %s", result["error"])
                    else:
                        metrics = {
                            "steps/val_step": int(result.get("step", update_step)),
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
                        # extra eval stats that may exist
                        lps = [
                            x
                            for x in result.get("avg_token_logprobs", [])
                            if x == x and not math.isinf(x)
                        ]
                        if lps:
                            metrics[f"{val_title}/mean_gen_token_logprob"] = float(
                                sum(lps) / len(lps)
                            )
                        wandb.log(metrics)
                        # table rows
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
                except pyqueue.Empty:
                    logger.info("eval: no result yet (will be logged later)")
                except Exception as e:
                    logger.warning("eval enqueue or drain failed: %s", e)
            except Exception:
                logger.exception("Failed to checkpoint for eval at step %d", update_step)

    # ---------------- graceful shutdown ----------------
    try:
        jobs_q.put_nowait(None)
    except Exception:
        pass
    try:
        vllm_proc.join(timeout=60.0)
        if vllm_proc.is_alive():
            vllm_proc.terminate()
            vllm_proc.join(timeout=10.0)
    except Exception:
        pass
    try:
        results_q.close()
        results_q.cancel_join_thread()
        jobs_q.close()
        jobs_q.cancel_join_thread()
    except Exception:
        pass
