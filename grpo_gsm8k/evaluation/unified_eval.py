from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path
from types import TracebackType
from typing import Any

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
        for i in range(120):  # Wait up to 120 seconds
            try:
                response = requests.get(f"{base_url}/v1/models", timeout=5)
                if response.status_code == 200:
                    logger.info(f"vLLM server is ready after {i+1} seconds")
                    return self
            except Exception:
                if i % 10 == 0:  # Log every 10 seconds
                    logger.info(f"Waiting for vLLM server... ({i+1}s)")
            time.sleep(1)

        # Get server logs for debugging
        if self.process and self.process.poll() is not None:
            stdout, stderr = self.process.communicate()
            logger.error(f"vLLM server startup failed. STDOUT: {stdout}")
            logger.error(f"vLLM server startup failed. STDERR: {stderr}")

        raise RuntimeError("vLLM server failed to start within 120 seconds")

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


def run_gsm8k_eval(
    model_path: str,
    eval_path: str,
    limit: int | None,
    max_new_tokens: int,
    k_shot: int,
    server_host: str = "127.0.0.1",
    server_port: int = 8000,
    tokenizer_path: str | None = None,
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
    for i, (prompt, data_item) in enumerate(zip(prompts, data)):
        if i % 50 == 0:
            logger.info(f"Processing GSM8K example {i+1}/{len(prompts)}")

        response = requests.post(
            base_url,
            json={
                "model": model_path,
                "prompt": prompt,
                "max_tokens": max_new_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
                "stream": False,
            },
            timeout=60,
        )

        if response.status_code != 200:
            logger.error(f"Server request failed: {response.text}")
            continue

        result = response.json()
        output_text = result["choices"][0]["text"]

        reward = reward_from_text(output_text, data_item["answer"], "boxed")

        results.append(
            {
                "id": data_item.get("id", i),
                "question": data_item["question"],
                "output": output_text,
                "reward": reward,
                "gold": data_item["answer"],
            }
        )

    pass_at_1 = sum(r["reward"] for r in results) / len(results) if results else 0.0
    logger.info(f"GSM8K evaluation complete: {pass_at_1:.3f} pass@1 on {len(results)} examples")

    return {
        "gsm8k_pass@1": pass_at_1,
        "gsm8k_n_examples": len(results),
        "gsm8k_results": results,
    }


def main(
    model_path: str = "Qwen/Qwen2.5-Math-1.5B",
    eval_suites: list[str] | None = None,
    limit: int | None = None,
    wandb_project: str = "grpo-gsm8k",
    run_name: str | None = None,
    output_dir: str = "./artifacts/unified_eval",
    # GSM8K specific args
    gsm8k_eval_path: str = "artifacts/gsm8k/test.jsonl",
    gsm8k_max_tokens: int = 1024,
    gsm8k_k_shot: int = 8,
    # lm-eval specific args
    lm_eval_tasks: list[str] | None = None,
    lm_eval_fewshot: int = 4,
    lm_eval_batch_size: int = 8,
    lm_eval_max_tokens: int = 2048,
    # vLLM args
    tp_size: int = 1,
    gpu_mem_util: float = 0.92,
) -> dict[str, Any]:
    """Run unified evaluation suite with shared vLLM server."""

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
        model_display_name = model_path_obj.name

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
        model_display_name = model_path

    # Log evaluation configuration
    logger.info("Evaluation configuration:")
    logger.info(f"  Model: {model_path}")
    logger.info(f"  Tokenizer: {tokenizer_path}")
    logger.info(f"  Eval suites: {eval_suites}")
    logger.info(f"  Limit: {limit}")

    # Initialize W&B
    config = {
        "model_path": model_path,
        "tokenizer_path": tokenizer_path,
        "is_local_checkpoint": is_local_checkpoint,
        "eval_suites": eval_suites,
        "limit": limit,
        "gsm8k_k_shot": gsm8k_k_shot,
        "lm_eval_fewshot": lm_eval_fewshot,
        "tp_size": tp_size,
        "gpu_mem_util": gpu_mem_util,
    }

    run = wandb.init(
        project=wandb_project,
        name=run_name or f"unified_eval_{model_display_name}",
        config=config,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

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
        with VLLMServerManager(model_path, tp_size=tp_size, gpu_mem_util=gpu_mem_util) as server:
            # Run GSM8K if requested
            if run_gsm8k:
                gsm8k_results = run_gsm8k_eval(
                    model_path=model_path,
                    eval_path=gsm8k_eval_path,
                    limit=limit,
                    max_new_tokens=gsm8k_max_tokens,
                    k_shot=gsm8k_k_shot,
                    server_host=server.host,
                    server_port=server.port,
                    tokenizer_path=tokenizer_path,
                )
                all_results.update(gsm8k_results)

                # Log GSM8K results to W&B
                wandb.log(
                    {
                        "metrics/gsm8k_pass@1": gsm8k_results["gsm8k_pass@1"],
                        "metrics/gsm8k_n_examples": gsm8k_results["gsm8k_n_examples"],
                    }
                )

            # Run lm-eval benchmarks if requested
            if run_lm_eval_suites:
                if "all" not in eval_suites and "lm_eval" not in eval_suites:
                    tasks_to_run = [task for task in lm_eval_tasks if task in eval_suites]
                else:
                    tasks_to_run = lm_eval_tasks

                if tasks_to_run:
                    # For lm-eval, pass the tokenizer path
                    lm_eval_tokenizer_path = tokenizer_path if is_local_checkpoint else model_path

                    lm_eval_results = run_lm_eval(
                        model_path=model_path,
                        tasks=tasks_to_run,
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
                    for task, task_results in lm_eval_results.items():
                        if isinstance(task_results, dict):
                            for metric, value in task_results.items():
                                if isinstance(value, int | float):
                                    wandb.log({f"metrics/{task}_{metric}": value})

        # Save unified results
        results_file = output_path / "unified_results.json"
        with results_file.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

        logger.info(f"Saved unified results to {results_file}")

        # Create W&B artifact
        artifact = wandb.Artifact("unified-eval", type="evaluation")
        artifact.add_file(str(results_file))
        run.log_artifact(artifact)

        return all_results

    finally:
        run.finish()


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
        f"num_concurrent=1,"
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
            capture_output=True,
            text=True,
            env=env,
            timeout=1800,  # 30 minute timeout
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
    if results_file.exists():
        with results_file.open(encoding="utf-8") as f:
            lm_eval_results = json.load(f)
        logger.info(f"Loaded results from {results_file}")
        return lm_eval_results.get("results", {})
    else:
        logger.warning(f"Results file not found at {results_file}")
        if output_path.exists():
            files = list(output_path.glob("*"))
            logger.info(f"Files in output directory: {files}")
        return {}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Unified evaluation suite for GSM8K and lm-eval benchmarks"
    )

    # Model and output args
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B",
        help="HuggingFace model repo or local checkpoint directory",
    )
    parser.add_argument(
        "--eval_suites",
        nargs="+",
        default=["all"],
        help="Evaluation suites to run: 'all', 'gsm8k', 'lm_eval', or specific task names",
    )
    parser.add_argument("--limit", type=int, help="Limit number of examples per evaluation")
    parser.add_argument("--output_dir", type=str, default="./artifacts/unified_eval")

    # W&B args
    parser.add_argument("--wandb_project", type=str, default="grpo-gsm8k")
    parser.add_argument("--run_name", type=str, help="W&B run name")

    # GSM8K specific args
    parser.add_argument("--gsm8k_eval_path", type=str, default="artifacts/gsm8k/test.jsonl")
    parser.add_argument("--gsm8k_max_tokens", type=int, default=1024)
    parser.add_argument("--gsm8k_k_shot", type=int, default=8)

    # lm-eval specific args
    parser.add_argument(
        "--lm_eval_tasks",
        nargs="+",
        default=[
            "hendrycks_math",
            "mmlu",
            "arc_challenge",
            "hellaswag",
            "winogrande",
            "truthfulqa_mc2",
            "wikitext",
        ],
    )
    parser.add_argument("--lm_eval_fewshot", type=int, default=4)
    parser.add_argument("--lm_eval_batch_size", type=int, default=8)
    parser.add_argument("--lm_eval_max_tokens", type=int, default=2048)

    # vLLM args
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--gpu_mem_util", type=float, default=0.92)

    args = parser.parse_args()
    main(**vars(args))
