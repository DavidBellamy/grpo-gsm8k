from __future__ import annotations

import csv
import datetime
import json
import logging
import math
import os
import random
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, cast

import wandb
from transformers import AutoTokenizer

from grpo_gsm8k.data.prompts import render_batch
from grpo_gsm8k.evaluation.reward_fn import reward_from_text

logger = logging.getLogger(__name__)


def load_jsonl(p: str) -> list[dict]:
    return [json.loads(line) for line in Path(p).open()]


class VLLMServerManager:
    """Context manager for vLLM server lifecycle."""

    def __init__(
        self,
        model_path: str,
        host: str = "127.0.0.1",
        port: int = 8000,
        dtype: str = "auto",
        served_model_name: str | None = None,
        tp_size: int = 1,
        gpu_mem_util: float = 0.92,
    ):
        self.model_path = model_path
        self.host = host
        self.port = port
        self.dtype = dtype
        self.served_model_name = served_model_name or model_path
        self.tp_size = tp_size
        self.gpu_mem_util = gpu_mem_util
        self.process: subprocess.Popen | None = None

    def __enter__(self) -> VLLMServerManager:
        logger.info(f"Starting vLLM server for {self.model_path}")
        cmd = [
            "vllm",
            "serve",
            self.model_path,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--dtype",
            self.dtype,
            "--served-model-name",
            self.served_model_name,
            "--tensor-parallel-size",
            str(self.tp_size),
            "--gpu-memory-utilization",
            str(self.gpu_mem_util),
            "--enable-prefix-caching",
            "--max-model-len",
            "4096",
        ]

        self.process = subprocess.Popen(
            cmd,
            stdout=None,
            stderr=None,
            text=True,
        )

        # Wait for server to be ready
        import requests

        base_url = f"http://{self.host}:{self.port}"
        for i in range(240):  # Wait up to 240 seconds
            try:
                response = requests.get(f"{base_url}/v1/models", timeout=5)
                if response.status_code == 200:
                    logger.info(f"vLLM server is ready after {i + 1} seconds")
                    return self
            except Exception:
                if i % 10 == 0:  # Log every 10 seconds
                    logger.info(f"Waiting for vLLM server... ({i + 1}s)")
            time.sleep(1)

        # Get server logs for debugging
        if self.process and self.process.poll() is not None:
            stdout, stderr = self.process.communicate()
            logger.error(f"vLLM server startup failed. STDOUT: {stdout}")
            logger.error(f"vLLM server startup failed. STDERR: {stderr}")

        try:
            logger.error("Timed out after 240s; terminating vLLM server")
            self.process.terminate()
            self.process.wait(timeout=10)
        except Exception:
            self.process.kill()

        raise RuntimeError("vLLM server failed to start within 240 seconds")

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self.process:
            logger.info("Stopping vLLM server")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("Force killing vLLM server")
                self.process.kill()
                self.process.wait()


def _percentile(xs: Sequence[float], q: float) -> float:
    """Linear interpolation percentile."""
    if not xs:
        return 0.0
    xs_sorted = sorted(xs)
    k = (len(xs_sorted) - 1) * q
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(xs_sorted[int(k)])
    return float(xs_sorted[f] + (xs_sorted[c] - xs_sorted[f]) * (k - f))


def compute_bootstrap_ci_binary(
    values: list[int] | list[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> dict[str, float]:
    """Bootstrap CI for binary sequences."""
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    mean_obs = sum(values) / n
    mu = sum(means) / len(means)
    var = sum((m - mu) ** 2 for m in means) / (len(means) - 1) if len(means) > 1 else 0.0
    std = math.sqrt(var)
    lower = _percentile(means, alpha / 2)
    upper = _percentile(means, 1 - alpha / 2)
    return {
        "mean": float(mean_obs),
        "std": float(std),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
    }


def compute_bootstrap_ci_percentile(
    values: list[float],
    q: float,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> dict[str, float]:
    """
    Bootstrap CI for a percentile (e.g., p50/p95).
    Returns dict with keys: mean, std, ci_lower, ci_upper.
    """
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
    rng = random.Random(seed)
    stats: list[float] = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        stats.append(_percentile(sample, q))
    mean_obs = _percentile(values, q)
    mu = sum(stats) / len(stats)
    var = sum((x - mu) ** 2 for x in stats) / (len(stats) - 1) if len(stats) > 1 else 0.0
    std = math.sqrt(var)
    lower = _percentile(stats, alpha / 2)
    upper = _percentile(stats, 1 - alpha / 2)
    return {
        "mean": float(mean_obs),
        "std": float(std),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
    }


def _parse_boxed_numeric(text: str) -> tuple[bool, float | None]:
    """
    Return (formatting_ok, value). formatting_ok is True when a \\boxed{...} exists and
    the boxed content can be parsed into a numeric scalar (int/float or simple \\frac{a}{b}).
    """
    # Find all \boxed{...} occurrences; take the last one if multiple
    matches = re.findall(r"\\boxed\s*\{([^}]*)\}", text, flags=re.DOTALL)
    if not matches:
        return False, None

    content = matches[-1]
    # Normalize
    s = content.strip()
    s = s.replace("$", "")
    s = s.replace(",", "")
    s = s.replace("−", "-")  # normalize unicode minus
    s = re.sub(r"\s+", "", s)

    # Try \frac{a}{b}
    m_frac = re.match(r"\\frac\s*\{\s*([-+]?\d+)\s*\}\s*\{\s*([-+]?\d+)\s*\}", s)
    if m_frac:
        try:
            num = int(m_frac.group(1))
            den = int(m_frac.group(2))
            if den == 0:
                return False, None
            return True, num / den
        except Exception:
            return False, None

    # Plain numeric, including scientific notation
    if re.match(r"^[-+]?\d+(?:\.\d+)?(?:e[+-]?\d+)?$", s, flags=re.IGNORECASE):
        try:
            return True, float(s)
        except Exception:
            return False, None

    # Not parseable as numeric
    return False, None


def _extract_gold_numeric_str(answer_text: str) -> str:
    """
    Return the text after '####', stripped and lightly normalized.
    """
    m = re.search(r"####\s*(.+)", answer_text)
    if not m:
        return ""
    s = m.group(1).strip()
    # Light normalization (remove dollar signs and commas)
    s = s.replace("$", "").replace(",", "")
    return s


def run_gsm8k_eval(
    model_path: str,
    eval_path: str,
    limit: int | None,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    k_shot: int,
    server_host: str = "127.0.0.1",
    server_port: int = 8000,
    tokenizer_path: str | None = None,
    bootstrap_samples: int = 1000,
    ci_alpha: float = 0.05,
    shard_id: int = 0,
    num_shards: int = 1,
) -> dict[str, Any]:
    """Run GSM8K evaluation using vLLM server."""
    logger.info("Running GSM8K evaluation")

    import requests

    # Use tokenizer_path if provided, otherwise fall back to model_path
    tok_path = tokenizer_path or model_path

    # Load tokenizer for prompt rendering
    tok = AutoTokenizer.from_pretrained(tok_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # Load eval set
    data = load_jsonl(eval_path)
    logger.info(f"Loaded {len(data)} examples from {eval_path}")

    if limit:
        data = data[:limit]
        logger.info(f"Limited to {len(data)} examples")

    # Shard data across workers
    total_n = len(data)
    if num_shards > 1:
        if not (0 <= shard_id < num_shards):
            raise ValueError(f"shard_id {shard_id} must be in [0, {num_shards})")
        data = [ex for i, ex in enumerate(data) if i % num_shards == shard_id]
        logger.info(
            f"[gsm8k] shard_id={shard_id}/{num_shards} "
            f"total_before={total_n} total_after_shard={len(data)}"
        )

    questions = [d["question"] for d in data]

    # For k-shot, load examples from training set
    few_shot_examples = None
    if k_shot > 0:
        train_data = load_jsonl("artifacts/r1_sft_pairs.jsonl")
        few_shot_examples = train_data[:k_shot]
        logger.info(f"Using {k_shot}-shot with {len(few_shot_examples)} examples")

    prompts = render_batch(
        tok, questions, add_generation_prompt=True, few_shot_examples=few_shot_examples
    )

    # Use server for generation
    base_url = f"http://{server_host}:{server_port}/v1/completions"

    results = []
    truncated_count = 0  # track how often generation hits max_tokens limit

    # Counters for error taxonomy
    format_error_count = 0
    logic_error_count = 0
    non_truncated_count = 0

    for i, (prompt, data_item) in enumerate(zip(prompts, data)):
        if i % 50 == 0:
            logger.info(f"Processing GSM8K example {i + 1}/{len(prompts)}")

        response = requests.post(
            base_url,
            json={
                "model": model_path,
                "prompt": prompt,
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": False,
            },
            timeout=600,
        )

        if response.status_code != 200:
            logger.error(f"Server request failed: {response.text}")
            continue

        result = response.json()
        choice = result["choices"][0]
        output_text = choice["text"]
        finish_reason = choice.get("finish_reason")
        if finish_reason == "length":
            truncated_count += 1

        reward_val = reward_from_text(output_text, data_item["answer"], "boxed")
        # Ensure reward is 0/1 int for typing and downstream tables
        try:
            reward_int = int(round(float(reward_val)))
        except Exception:
            reward_int = 0

        # Determine formatting correctness from last \boxed{...}
        formatting_ok, _ = _parse_boxed_numeric(output_text)
        is_truncated = finish_reason == "length"

        # Extract raw boxed content as "Model Answer" (if present)
        boxed_matches = re.findall(r"\\boxed\s*\{([^}]*)\}", output_text, flags=re.DOTALL)
        model_answer = ""
        if boxed_matches:
            # Preserve content but strip surrounding $ and trim whitespace
            content = boxed_matches[-1].strip()
            model_answer = content.replace("$", "").strip()

        # Extract gold numeric answer from the gold field (text after '####')
        gold_numeric_answer = _extract_gold_numeric_str(data_item["answer"])

        # Count error types only for non-truncated completions
        if not is_truncated:
            non_truncated_count += 1
            if reward_val == 0:
                if not formatting_ok:
                    format_error_count += 1
                else:
                    logic_error_count += 1

        # Per-example completion length in tokens
        completion_len = len(tok.encode(output_text, add_special_tokens=False))

        results.append(
            {
                "id": data_item.get("id", i),
                "question": data_item["question"],
                "output": output_text,
                "reward": reward_int,
                "gold": data_item["answer"],
                "finish_reason": finish_reason,
                "formatting_ok": formatting_ok,
                "is_truncated": is_truncated,
                "model_answer": model_answer,
                "gold_numeric_answer": gold_numeric_answer,
                "completion_len": completion_len,
            }
        )

    # Counts for system-level vs conditional metrics
    rewards = [int(r["reward"]) for r in results]
    n_total = len(results)
    n_pass = sum(rewards)
    n_trunc = truncated_count
    n_fmt = format_error_count
    n_logic = logic_error_count
    non_truncated = n_total - n_trunc
    parsed = non_truncated - n_fmt if non_truncated > 0 else 0

    # System-level rates (uniform over all N)
    pass_at_1 = (n_pass / n_total) if n_total else 0.0
    trunc_rate = (n_trunc / n_total) if n_total else 0.0
    fmt_err_rate = (n_fmt / n_total) if n_total else 0.0
    logic_err_rate = (n_logic / n_total) if n_total else 0.0

    # Conditional reasoning metrics
    fmt_given_not_trunc = (n_fmt / non_truncated) if non_truncated > 0 else 0.0
    pass_given_parsed = (n_pass / parsed) if parsed > 0 else 0.0
    logic_given_parsed = (n_logic / parsed) if parsed > 0 else 0.0

    logger.info(f"GSM8K evaluation complete: {pass_at_1:.3f} pass@1 on {n_total} examples")

    # Bootstrap CI over per-example rewards (for pass@1)
    stats = (
        compute_bootstrap_ci_binary(rewards, n_boot=bootstrap_samples, alpha=ci_alpha)
        if results
        else {
            "mean": 0.0,
            "std": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
        }
    )

    # Completion length stats (token counts) for GSM8K completions
    comp_lens = (
        [len(tok.encode(r["output"], add_special_tokens=False)) for r in results] if results else []
    )
    comp_p50 = _percentile(comp_lens, 0.5) if comp_lens else 0.0
    comp_p95 = _percentile(comp_lens, 0.95) if comp_lens else 0.0

    # --- Bootstrap CIs for additional GSM8K metrics ---
    if comp_lens:
        p50_ci = compute_bootstrap_ci_percentile(
            [float(x) for x in comp_lens],
            q=0.5,
            n_boot=bootstrap_samples,
            alpha=ci_alpha,
        )
        p95_ci = compute_bootstrap_ci_percentile(
            [float(x) for x in comp_lens],
            q=0.95,
            n_boot=bootstrap_samples,
            alpha=ci_alpha,
        )
    else:
        p50_ci = {"ci_lower": 0.0, "ci_upper": 0.0}
        p95_ci = {"ci_lower": 0.0, "ci_upper": 0.0}

    # Truncation rate CI (binary)
    if results:
        trunc_flags = [1 if r["is_truncated"] else 0 for r in results]
        trunc_ci = compute_bootstrap_ci_binary(
            trunc_flags, n_boot=bootstrap_samples, alpha=ci_alpha
        )
    else:
        trunc_ci = {"ci_lower": 0.0, "ci_upper": 0.0}

    # System-level format/logical error rate CIs (binary flags over all examples)
    if results:
        fmt_flags = [
            1
            if ((int(r["reward"]) == 0) and (not r["formatting_ok"]) and (not r["is_truncated"]))
            else 0
            for r in results
        ]
        logic_flags = [
            1 if ((int(r["reward"]) == 0) and r["formatting_ok"] and (not r["is_truncated"])) else 0
            for r in results
        ]
        fmt_ci = compute_bootstrap_ci_binary(fmt_flags, n_boot=bootstrap_samples, alpha=ci_alpha)
        logic_ci = compute_bootstrap_ci_binary(
            logic_flags, n_boot=bootstrap_samples, alpha=ci_alpha
        )
    else:
        fmt_ci = {"ci_lower": 0.0, "ci_upper": 0.0}
        logic_ci = {"ci_lower": 0.0, "ci_upper": 0.0}

    return {
        "gsm8k_pass@1": pass_at_1,
        "gsm8k_n_examples": len(results),
        "gsm8k_results": results,
        "gsm8k_pass@1_ci_lower": stats["ci_lower"],
        "gsm8k_pass@1_ci_upper": stats["ci_upper"],
        "gsm8k_completion_len_p50": comp_p50,
        "gsm8k_completion_len_p95": comp_p95,
        "gsm8k_truncation_rate": trunc_rate,
        "gsm8k_format_error_rate": fmt_err_rate,
        "gsm8k_logic_error_rate": logic_err_rate,
        "gsm8k_format_error_rate_given_not_trunc": fmt_given_not_trunc,
        "gsm8k_pass_given_parsed": pass_given_parsed,
        "gsm8k_logic_error_rate_given_parsed": logic_given_parsed,
        "gsm8k_completion_len_p50_ci_lower": p50_ci["ci_lower"],
        "gsm8k_completion_len_p50_ci_upper": p50_ci["ci_upper"],
        "gsm8k_completion_len_p95_ci_lower": p95_ci["ci_lower"],
        "gsm8k_completion_len_p95_ci_upper": p95_ci["ci_upper"],
        "gsm8k_truncation_rate_ci_lower": trunc_ci["ci_lower"],
        "gsm8k_truncation_rate_ci_upper": trunc_ci["ci_upper"],
        "gsm8k_format_error_rate_ci_lower": fmt_ci["ci_lower"],
        "gsm8k_format_error_rate_ci_upper": fmt_ci["ci_upper"],
        "gsm8k_logic_error_rate_ci_lower": logic_ci["ci_lower"],
        "gsm8k_logic_error_rate_ci_upper": logic_ci["ci_upper"],
    }


def run_lm_eval(
    model_path: str,
    tasks: list[str],
    num_fewshot: int,
    batch_size: int,
    max_new_tokens: int,
    limit: int | None,
    output_dir: Path,
    is_local_checkpoint: bool,
    server_host: str = "127.0.0.1",
    server_port: int = 8000,
    tokenizer_path: str | None = None,
) -> dict[str, Any]:
    """Run lm-eval harness with vLLM server."""

    # Test server connection first
    import requests

    try:
        # Test the /v1/models endpoint to verify server is ready
        models_response = requests.get(f"http://{server_host}:{server_port}/v1/models", timeout=10)
        if models_response.status_code != 200:
            logger.error(f"Server not responding: {models_response.status_code}")
            raise RuntimeError(f"vLLM server not ready: {models_response.status_code}")

        models_data = models_response.json()
        logger.info(f"Successfully connected to vLLM server with {len(models_data['data'])} models")

        # Test completions endpoint
        test_payload = {"model": model_path, "prompt": "Test", "max_tokens": 1, "temperature": 0.0}
        completions_response = requests.post(
            f"http://{server_host}:{server_port}/v1/completions", json=test_payload, timeout=10
        )
        if completions_response.status_code != 200:
            logger.error(f"Completions endpoint test failed: {completions_response.text}")
            raise RuntimeError(
                f"vLLM completions endpoint not working: {completions_response.status_code}"
            )

        logger.info("vLLM server endpoints verified successfully")

    except Exception as e:
        logger.error(f"Failed to connect to vLLM server: {e}")
        raise

    # For local-completions, we need to specify the full completions URL
    completions_url = f"http://{server_host}:{server_port}/v1/completions"

    # Use tokenizer_path if provided and we have a local checkpoint
    if is_local_checkpoint and tokenizer_path:
        actual_tokenizer_path = tokenizer_path
    else:
        actual_tokenizer_path = model_path

    # Build model args - use the full completions URL for local-completions
    model_args = (
        f"model={model_path},"
        f"base_url={completions_url},"
        f"num_concurrent=10,"
        f"tokenized_requests=false,"
        f"tokenizer={actual_tokenizer_path},"
        f"tokenizer_backend=huggingface",
        f"max_tokens={max_new_tokens}",
    )

    # Prepare output path
    model_name = model_path.replace("/", "_").replace("\\", "_")
    output_path = output_dir / f"{model_name}_lm_eval"
    output_path.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd: list[str] = [
        "python",
        "-m",
        "lm_eval",
        "--model",
        "local-completions",
        "--model_args",
        ",".join(model_args),
        "--tasks",
        ",".join(tasks),
        "--num_fewshot",
        str(num_fewshot),
        "--batch_size",
        str(batch_size),
        "--output_path",
        str(output_path),
        "--log_samples",
    ]

    if limit:
        cmd.extend(["--limit", str(limit)])

    # Set environment
    import os

    env = os.environ.copy()
    env["OPENAI_API_KEY"] = "EMPTY"

    logger.info(f"Running lm-eval with command: {' '.join(cmd)}")
    logger.info(f"Model args: {model_args}")
    logger.info(f"Output will be saved to: {output_path}")

    # Run with timeout and better error handling
    try:
        result = subprocess.run(
            cmd,
            text=True,
            env=env,
            timeout=1800,  # 30 minute timeout
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
    except subprocess.TimeoutExpired:
        logger.error("lm-eval command timed out after 30 minutes")
        raise RuntimeError("lm-eval timed out")

    if result.returncode != 0:
        logger.error(f"lm-eval failed with return code {result.returncode}")
        logger.error(f"STDOUT: {result.stdout}")
        logger.error(f"STDERR: {result.stderr}")
        raise RuntimeError(f"lm-eval failed with return code {result.returncode}")

    logger.info("lm-eval completed successfully")
    if result.stdout:
        logger.info(f"lm-eval STDOUT: {result.stdout[-1000:]}")

    # Parse results
    results_file = output_path / "results.json"
    if not results_file.exists():
        # Look for newest results_*.json anywhere under output_path
        # (includes subdirs like Qwen__Qwen2.5-Math-1.5B/)
        candidates = sorted(
            output_path.rglob("results_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            logger.warning(f"Results file not found under {output_path}")
            return {}
        results_file = candidates[0]
        logger.info(f"Using discovered results file at {results_file}")

    with results_file.open(encoding="utf-8") as f:
        raw = json.load(f)

    logger.info(f"Loaded results from {results_file}")
    return raw.get("results", {})


def assign_lm_eval_tasks_to_shard(
    tasks_to_run: list[str],
    num_shards: int,
    shard_id: int,
    logger: logging.Logger | None = None,
) -> list[str]:
    """
    Greedy weighted partitioning of lm-eval tasks across shards.

    Uses a simple bin-packing heuristic based on a hand-crafted cost
    per task (approximate runtime). Returns the list of tasks assigned
    to the given shard_id.
    """
    if num_shards <= 1:
        return tasks_to_run

    if not (0 <= shard_id < num_shards):
        raise ValueError(f"shard_id {shard_id} must be in [0, {num_shards})")

    # Heuristic costs based on empirical/runtime ranking:
    # truthfulqa_mc2 < winogrande ≲ arc_challenge < hellaswag
    # < wikitext ≲ mmlu ≲ hendrycks_math
    task_cost: dict[str, int] = {
        "truthfulqa_mc2": 1,
        "winogrande": 2,
        "arc_challenge": 2,
        "hellaswag": 3,
        "wikitext": 4,
        "mmlu": 5,
        "hendrycks_math": 6,
    }

    # Sort tasks by descending cost so we place heavy ones first
    weighted_tasks = sorted(
        tasks_to_run,
        key=lambda t: task_cost.get(t, 1),
        reverse=True,
    )

    # Greedy bin packing: assign each task to shard with smallest current load
    shard_tasks: list[list[str]] = [[] for _ in range(num_shards)]
    shard_loads: list[int] = [0 for _ in range(num_shards)]

    for t in weighted_tasks:
        cost = task_cost.get(t, 1)
        j = min(range(num_shards), key=lambda i: shard_loads[i])
        shard_tasks[j].append(t)
        shard_loads[j] += cost

    if logger is not None:
        logger.info(
            "[lm-eval] partitioned tasks across %d shards: loads=%s; shard_%d_tasks=%s",
            num_shards,
            shard_loads,
            shard_id,
            shard_tasks[shard_id],
        )

    return shard_tasks[shard_id]


def log_gsm8k_to_wandb(
    gsm8k_results: dict[str, Any],
    model_path: str,
) -> None:
    if os.environ.get("WANDB_DISABLED") == "true":
        logger.info("WANDB_DISABLED=true, skipping W&B loggin")
        return

    wandb.log(
        {
            "metrics/gsm8k_pass@1": gsm8k_results["gsm8k_pass@1"],
            "metrics/gsm8k_n_examples": gsm8k_results["gsm8k_n_examples"],
            "metrics/gsm8k_pass@1_ci_lower": gsm8k_results.get("gsm8k_pass@1_ci_lower", 0.0),
            "metrics/gsm8k_pass@1_ci_upper": gsm8k_results.get("gsm8k_pass@1_ci_upper", 0.0),
            "metrics/gsm8k_completion_len_p50": gsm8k_results.get("gsm8k_completion_len_p50", 0.0),
            "metrics/gsm8k_completion_len_p95": gsm8k_results.get("gsm8k_completion_len_p95", 0.0),
            "metrics/gsm8k_truncation_rate": gsm8k_results.get("gsm8k_truncation_rate", 0.0),
            "metrics/gsm8k_format_error_rate": gsm8k_results.get("gsm8k_format_error_rate", 0.0),
            "metrics/gsm8k_logic_error_rate": gsm8k_results.get("gsm8k_logic_error_rate", 0.0),
            "metrics/gsm8k_format_error_rate_given_not_trunc": gsm8k_results.get(
                "gsm8k_format_error_rate_given_not_trunc", 0.0
            ),
            "metrics/gsm8k_pass_given_parsed": gsm8k_results.get("gsm8k_pass_given_parsed", 0.0),
            "metrics/gsm8k_logic_error_rate_given_parsed": gsm8k_results.get(
                "gsm8k_logic_error_rate_given_parsed", 0.0
            ),
            "metrics/gsm8k_completion_len_p50_ci_lower": gsm8k_results.get(
                "gsm8k_completion_len_p50_ci_lower", 0.0
            ),
            "metrics/gsm8k_completion_len_p50_ci_upper": gsm8k_results.get(
                "gsm8k_completion_len_p50_ci_upper", 0.0
            ),
            "metrics/gsm8k_completion_len_p95_ci_lower": gsm8k_results.get(
                "gsm8k_completion_len_p95_ci_lower", 0.0
            ),
            "metrics/gsm8k_completion_len_p95_ci_upper": gsm8k_results.get(
                "gsm8k_completion_len_p95_ci_upper", 0.0
            ),
            "metrics/gsm8k_truncation_rate_ci_lower": gsm8k_results.get(
                "gsm8k_truncation_rate_ci_lower", 0.0
            ),
            "metrics/gsm8k_truncation_rate_ci_upper": gsm8k_results.get(
                "gsm8k_truncation_rate_ci_upper", 0.0
            ),
            "metrics/gsm8k_format_error_rate_ci_lower": gsm8k_results.get(
                "gsm8k_format_error_rate_ci_lower", 0.0
            ),
            "metrics/gsm8k_format_error_rate_ci_upper": gsm8k_results.get(
                "gsm8k_format_error_rate_ci_upper", 0.0
            ),
            "metrics/gsm8k_logic_error_rate_ci_lower": gsm8k_results.get(
                "gsm8k_logic_error_rate_ci_lower", 0.0
            ),
            "metrics/gsm8k_logic_error_rate_ci_upper": gsm8k_results.get(
                "gsm8k_logic_error_rate_ci_upper", 0.0
            ),
        }
    )

    # Log full completions table (one row per prompt) and overwrite within run
    comp_cols = [
        "Prompt",
        "Gold Reasoning",
        "Model Completion",
        "Model Answer",
        "Gold Answer",
        "Correct",
        "Completion Length",
        "Truncated",
        "Stop Reason",
    ]
    comp_rows: list[list[Any]] = []
    examples = cast(list[dict[str, Any]], gsm8k_results.get("gsm8k_results", []))
    for ex in examples:
        comp_rows.append(
            [
                ex.get("question", ""),
                ex.get("gold", ""),
                ex.get("output", ""),
                ex.get("model_answer", ""),
                ex.get("gold_numeric_answer", ""),
                int(ex.get("reward", 0)),
                int(ex.get("completion_len", 0)),
                int(1 if ex.get("is_truncated") else 0),
                str(ex.get("finish_reason") or ""),
            ]
        )
    completions_table = wandb.Table(columns=comp_cols, data=comp_rows)
    table_key = f"Model Completions on GSM8k Test Set for {model_path}"
    # Put into run.summary to overwrite on subsequent writes within the same run
    wandb.run.summary[table_key] = completions_table


def main(
    model_path: str = "Qwen/Qwen2.5-Math-1.5B",
    eval_suites: list[str] | None = None,
    limit: int | None = None,
    output_dir: str = "./artifacts/eval",
    # GSM8K specific args
    gsm8k_eval_path: str = "artifacts/gsm8k/test.jsonl",
    gsm8k_max_tokens: int = 1024,
    gsm8k_k_shot: int = 8,
    gsm8k_bootstrap_samples: int = 10,
    gsm8k_ci_alpha: float = 0.05,
    # lm-eval specific args
    lm_eval_tasks: list[str] | None = None,
    lm_eval_fewshot: int = 4,
    lm_eval_batch_size: int = 8,
    lm_eval_max_tokens: int = 2048,
    # vLLM args
    tp_size: int = 1,
    gpu_mem_util: float = 0.92,
    # data-parallel sharding
    num_shards: int = 1,
    shard_id: int = 0,
    server_port: int = 8000,
    *,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    """Run evaluation suite with shared vLLM server."""

    # Default eval suites
    if eval_suites is None:
        eval_suites = ["all"]

    # Default lm-eval tasks
    if lm_eval_tasks is None:
        lm_eval_tasks = [
            "hendrycks_math",
            "mmlu",
            "arc_challenge",
            "hellaswag",
            "winogrande",
            "truthfulqa_mc2",
            "wikitext",
        ]

    # Determine if model is local checkpoint or HF repo
    model_path_obj = Path(model_path)
    is_local_checkpoint = model_path_obj.exists() and model_path_obj.is_dir()

    # For local checkpoints, try to find the base model for tokenizer
    tokenizer_path = model_path
    if is_local_checkpoint:
        logger.info(f"Using local checkpoint: {model_path}")

        # Try to find base model info for tokenizer
        config_file = model_path_obj / "config.json"
        if config_file.exists():
            try:
                with config_file.open(encoding="utf-8") as f:
                    config = json.load(f)
                # Look for common base model identifiers
                base_model_candidates = [
                    config.get("_name_or_path"),
                    config.get("base_model_name_or_path"),
                    config.get("model_name_or_path"),
                ]
                for candidate in base_model_candidates:
                    if candidate and "/" in candidate and not Path(candidate).exists():
                        # This looks like a HF model name, use it for tokenizer
                        tokenizer_path = candidate
                        logger.info(f"Using base model for tokenizer: {tokenizer_path}")
                        break
            except Exception as e:
                logger.warning(f"Could not read config for base model info: {e}")

    else:
        logger.info(f"Using HuggingFace model: {model_path}")

    # Log evaluation configuration
    logger.info("Evaluation configuration:")
    logger.info(f"  Model: {model_path}")
    logger.info(f"  Tokenizer: {tokenizer_path}")
    logger.info(f"  Eval suites: {eval_suites}")
    logger.info(f"  Limit: {limit}")

    logger.info(f"  Limit: {limit}")

    # Build per-run eval folder: artifacts/eval/{YYYYMMDD}_{wandb_run}
    # use pid if wandb is disabled
    date_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")
    wb_run_id = os.environ.get("WANDB_RUN_ID")
    if not wb_run_id:
        wb_disabled = os.environ.get("WANDB_DISABLED") == "true"
        run_obj = getattr(wandb, "run", None)
        if (not wb_disabled) and run_obj is not None and getattr(run_obj, "id", None):
            wb_run_id = run_obj.id
        else:
            wb_run_id = f"local-{os.getpid()}"

    per_run_suffix = f"{date_str}_{wb_run_id}"
    root = Path(output_dir)
    if root.name == per_run_suffix or str(root).endswith(per_run_suffix):
        output_path = root
    else:
        output_path = root / per_run_suffix
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results directory: {output_path}")

    all_results = {}

    try:
        # Determine what evaluations to run
        run_gsm8k = "all" in eval_suites or "gsm8k" in eval_suites
        run_lm_eval_suites = (
            "all" in eval_suites
            or "lm_eval" in eval_suites
            or any(task in eval_suites for task in lm_eval_tasks)
        )

        # Use shared vLLM server for all evaluations
        with VLLMServerManager(
            model_path, tp_size=tp_size, gpu_mem_util=gpu_mem_util, port=server_port
        ) as server:
            # Run GSM8K if requested
            if run_gsm8k:
                gsm8k_results = run_gsm8k_eval(
                    model_path=model_path,
                    eval_path=gsm8k_eval_path,
                    limit=limit,
                    max_new_tokens=gsm8k_max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    k_shot=gsm8k_k_shot,
                    server_host=server.host,
                    server_port=server.port,
                    tokenizer_path=tokenizer_path,
                    bootstrap_samples=gsm8k_bootstrap_samples,
                    ci_alpha=gsm8k_ci_alpha,
                    shard_id=shard_id,
                    num_shards=num_shards,
                )
                all_results.update(gsm8k_results)

                log_gsm8k_to_wandb(
                    gsm8k_results=gsm8k_results,
                    model_path=model_path,
                )

            # Run lm-eval benchmarks if requested
            if run_lm_eval_suites:
                if "all" not in eval_suites and "lm_eval" not in eval_suites:
                    tasks_to_run = [task for task in lm_eval_tasks if task in eval_suites]
                else:
                    tasks_to_run = lm_eval_tasks

                tasks_for_this_shard = assign_lm_eval_tasks_to_shard(
                    tasks_to_run=tasks_to_run,
                    num_shards=num_shards,
                    shard_id=shard_id,
                    logger=logger,
                )

                if tasks_for_this_shard:
                    # For lm-eval, pass the tokenizer path
                    lm_eval_tokenizer_path = tokenizer_path if is_local_checkpoint else model_path

                    lm_eval_results = run_lm_eval(
                        model_path=model_path,
                        tasks=tasks_for_this_shard,
                        num_fewshot=lm_eval_fewshot,
                        batch_size=lm_eval_batch_size,
                        max_new_tokens=lm_eval_max_tokens,
                        limit=limit,
                        output_dir=output_path,
                        is_local_checkpoint=is_local_checkpoint,
                        server_host=server.host,
                        server_port=server.port,
                        tokenizer_path=lm_eval_tokenizer_path,
                    )
                    all_results["lm_eval"] = lm_eval_results

                    # Log lm-eval results to W&B
                    for task, metrics in lm_eval_results.items():
                        if not isinstance(metrics, dict):
                            continue
                        to_log = {}
                        for metric_name, val in metrics.items():
                            if isinstance(val, int | float):
                                # keep the original metric name to stay 1:1 with lm-eval
                                to_log[f"metrics/lm_eval/{task}/{metric_name}"] = val
                        if to_log:
                            wandb.log(to_log)

                    # --- W&B Table: lm-eval results (append across runs via artifact) ---
                    def _metric_or_empty(task: str, metric_keys: list[str]) -> str:
                        v: Any = None
                        task_dict = lm_eval_results.get(task, {})
                        if isinstance(task_dict, dict):
                            for mk in metric_keys:
                                if mk in task_dict and isinstance(task_dict[mk], int | float):
                                    v = task_dict[mk]
                                    break
                        return f"{float(v):.4f}" if isinstance(v, int | float) else ""

                    lm_cols: list[str] = [
                        "Model ID",
                        "mmlu pass@1",
                        "MATH pass@1",
                        "arc-challenge pass@1",
                        "hellaswag pass@1",
                        "truthfulqa pass@1",
                        "winogrande pass@1",
                        "wikitext bits-per-byte",
                    ]

                    # Fallbacks cover slight metric-name variations across lm-eval versions
                    row_values: list[str] = [
                        model_path,
                        _metric_or_empty("mmlu", ["acc,none", "acc"]),
                        _metric_or_empty("hendrycks_math", ["exact_match,none", "exact_match"]),
                        _metric_or_empty("arc_challenge", ["acc_norm,none", "acc_norm"]),
                        _metric_or_empty("hellaswag", ["acc_norm,none", "acc_norm"]),
                        _metric_or_empty("truthfulqa_mc2", ["acc,none", "acc"]),
                        _metric_or_empty("winogrande", ["acc,none", "acc"]),
                        _metric_or_empty(
                            "wikitext", ["bits_per_byte,none", "bits_per_byte", "bpb,none", "bpb"]
                        ),
                    ]

                    lm_table_art_name = "lm-eval-results"
                    lm_table_filename = "lm_eval_results.csv"
                    lm_csv_rows: list[list[str]] = []

                    try:
                        api = wandb.Api()
                        latest_ref = (
                            f"{wandb.run.entity}/{wandb.run.project}/{lm_table_art_name}:latest"
                        )
                        art = api.artifact(latest_ref)
                        dl_dir = Path(art.download())
                        prior_csv = dl_dir / lm_table_filename
                        if prior_csv.exists():
                            with prior_csv.open(newline="", encoding="utf-8") as f:
                                reader = csv.reader(f)
                                lm_csv_rows = [r for r in reader]
                    except Exception:
                        lm_csv_rows = []

                    # Ensure header and overwrite row for the same model_id if present
                    if not lm_csv_rows or lm_csv_rows[0] != lm_cols:
                        lm_csv_rows = [lm_cols, row_values]
                    else:
                        header = lm_csv_rows[0]
                        existing_rows = lm_csv_rows[1:]
                        replaced = False
                        for idx, r in enumerate(existing_rows):
                            if r and r[0] == model_path:  # Model ID column
                                existing_rows[idx] = row_values
                                replaced = True
                                break
                        if not replaced:
                            existing_rows.append(row_values)
                        lm_csv_rows = [header] + existing_rows

                    # Write updated CSV locally
                    lm_table_path = output_path / lm_table_filename
                    with lm_table_path.open("w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerows(lm_csv_rows)

                    # Log artifact with alias "latest" so next run can update in place
                    lm_table_artifact = wandb.Artifact(lm_table_art_name, type="evaluation-table")
                    lm_table_artifact.add_file(str(lm_table_path))
                    wandb.run.log_artifact(lm_table_artifact, aliases=["latest"])

                    # Also log a W&B Table in this run for visualization
                    lm_wb_table = wandb.Table(columns=lm_cols, data=lm_csv_rows[1:])
                    wandb.log({"lm-eval results": lm_wb_table})
                else:
                    logger.info(
                        f"[lm-eval] shard_id={shard_id} has no tasks assigned; skipping lm-eval"
                    )

        # Save results
        if num_shards > 1:
            results_file = output_path / f"results_shard{shard_id}_of_{num_shards}.json"
        else:
            results_file = output_path / "results.json"

        with results_file.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

        logger.info(f"Saved results to {results_file}")

        # Create W&B artifact
        artifact = wandb.Artifact("eval", type="evaluation")
        artifact.add_file(str(results_file))
        wandb.run.log_artifact(artifact)

        return all_results

    finally:
        if os.environ.get("WANDB_DISABLED") != "true":
            run = getattr(wandb, "run", None)
            if run is not None:
                wandb.run.finish()
