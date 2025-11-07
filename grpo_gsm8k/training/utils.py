from __future__ import annotations

import json
import logging
import math
import os
import shutil
import statistics
import time
from multiprocessing import Queue
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from transformers import PreTrainedModel, PreTrainedTokenizer
from vllm import LLM, SamplingParams

from grpo_gsm8k.evaluation.reward_fn import extract_answer_colon, extract_boxed, normalize_number


def _extract_final_answer(text: str) -> str | None:
    """
    Try boxed then 'ANSWER:' style extraction; fall back to None.
    """
    return extract_boxed(text) or extract_answer_colon(text)


def load_templated_gsm8k(path: str | Path) -> tuple[list[str], list[str], list[str]]:
    """
    Strict loader for pre-rendered val set (JSONL) for vLLM eval.
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
                    f"Val row {i} missing required keys. Expected 'prompt','gold','gold_num'."
                )
            prompt = str(obj["prompt"])
            gold = str(obj["gold"])
            gold_num_raw = obj["gold_num"]
            if gold_num_raw is None:
                raise ValueError(f"Val row {i} has null gold_num")
            gold_num = str(gold_num_raw)
            if gold_num == "":
                raise ValueError(f"Val row {i} has empty gold_num (no on-the-fly parsing allowed)")
            prompts.append(prompt)
            gold_strs.append(gold)
            gold_nums.append(gold_num)
    if not prompts:
        raise ValueError(f"No val examples loaded from {path}")
    return prompts, gold_strs, gold_nums


def _normalize_num_str(s: str | None) -> str:
    """
    Normalize numeric substrings (remove commas, keep canonical numeric token).
    Returns '' when no parseable number found (keeps length aligned).
    """
    if s is None:
        return ""
    n = normalize_number(s)
    return n if n is not None else ""


def default_reward(gt: str, resp: str) -> dict[str, float]:
    # exact-match on normalized number; add a simple format bonus for boxed/ANSWER:
    pred = _normalize_num_str(_extract_final_answer(resp))
    correct = 1.0 if (pred and _normalize_num_str(gt) == pred) else 0.0
    fmt = 1.0 if (extract_boxed(resp) or extract_answer_colon(resp)) else 0.0
    return {
        "reward": 0.9 * correct + 0.1 * fmt,
        "format_reward": fmt,
        "answer_reward": correct,
    }


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


def ensure_pad_token(tokenizer: PreTrainedTokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})


def resolve_resume_path(root: str | Path) -> Path:
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


def sanitize_wandb_component(name: str) -> str:
    """
    W&B treats '/' as a namespace separator. Replace with a lookalike Unicode slash
    to keep a single section while remaining readable.
    """
    return name.replace("/", "∕")


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


def vllm_worker_persistent(
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


def save_policy_checkpoint_for_vllm(
    policy: PreTrainedModel,
    step: int,
    *,
    out_root: str | Path = "/dev/shm",
    base: str = "qwen15b_step",
    dtype: torch.dtype = torch.bfloat16,
    logger: logging.Logger | None = None,
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
        if logger:
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
    if logger:
        logger.info("Wrote vLLM checkpoint: %s", final_dir)
    return final_dir


class RunningStats:
    def __init__(self) -> None:
        self.n: int = 0
        self.mean: float = 0.0
        self.M2: float = 0.0
        self.vmin: float = math.inf
        self.vmax: float = -math.inf

    def update_batch(self, x: torch.Tensor) -> None:
        # x is 1D cpu/gpu tensor; do it on CPU to avoid GPU mem
        t: torch.Tensor = x.detach().float().cpu()

        k: int = int(t.numel())
        if k == 0:
            return

        # per-batch summary stats as Python floats
        batch_min: float = float(t.min().item())
        batch_max: float = float(t.max().item())
        m: float = float(t.mean().item())
        v: float = float(t.var(unbiased=False).item())

        # update running min/max
        self.vmin = min(self.vmin, batch_min)
        self.vmax = max(self.vmax, batch_max)

        # merge batch stats into running stats
        n_total: int = self.n + k
        delta: float = m - self.mean
        self.mean += delta * (k / n_total)

        # parallel variance (Welford/Chan) update
        self.M2 += v * k + (delta * delta) * (self.n * k / n_total)
        self.n = n_total

    def finalize(self) -> dict[str, float]:
        std: float = math.sqrt(self.M2 / max(1, self.n)) if self.n else 0.0
        return {
            "mean": self.mean,
            "std": std,
            "min": self.vmin if self.n else 0.0,
            "max": self.vmax if self.n else 0.0,
        }
