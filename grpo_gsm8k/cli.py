"""
CLI for running evaluation and data preparation workflows for the grpo-gsm8k project.

This script provides a unified interface to:
- Prepare evaluation data with pinned revision and optional caching.
- Collect system information for reproducibility.
- Run model evaluation using vllm.
- Log run metadata and manifest for provenance.
- Export the exact Python environment used for the run.
- Snapshot external datasets for reproducibility.

Usage:
    python -m grpo_gsm8k.cli eval [options]

Subcommands:
    eval    Run evaluation with unified logging and artifact management.

Options:
    --model_id          Model identifier (default: "Qwen/Qwen2.5-Math-1.5B").
    --eval_path         Path to evaluation data (default: "artifacts/gsm8k/val.jsonl").
    --limit             Limit number of evaluation samples.
    --batch_size        Batch size for evaluation (default: 8).
    --max_new_tokens    Maximum number of new tokens to generate (default: 384).
    --gpu_mem_util      GPU memory utilization for vllm (default: 0.92).
    --tp_size           Tensor parallel size for vllm (default: 1).
    --wandb_project     Weights & Biases project name (default: "grpo-gsm8k").
    --run_name          Optional run name for logging.
    --out_dir           Output directory for prepared data (default: "artifacts/gsm8k").
    --seed              Random seed for data preparation (default: 31415).
    --eval_n            Number of evaluation samples to prepare (default: 800).
    --revision          Data/code revision to pin (default: "main").
    --cache_dir         Cache directory for HuggingFace datasets
                        (default: "/workspace/.cache/huggingface").

Artifacts and logs are stored in timestamped run directories under "artifacts/runs".
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import wandb

from grpo_gsm8k import data_prep
from grpo_gsm8k.fast_eval_vllm import main as vllm_eval_main
from grpo_gsm8k.repro import write_run_manifest
from grpo_gsm8k.sft import train_sft_on_r1_pairs


def _sh(cmd: list[str], **kw: Any) -> None:
    subprocess.run(cmd, check=True, **kw)


def _git_sha_short() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "no-git"


def _run_dir() -> Path:
    rid = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{_git_sha_short()}"
    rd = Path("artifacts/runs") / rid
    rd.mkdir(parents=True, exist_ok=True)
    return rd


def _setup_logging(run_dir: Path, level: int = logging.INFO) -> Path:
    """
    Configure root logging to stream to stderr and write a file under the run directory.
    Returns the log file path.
    """
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run.log"
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),  # VS Code terminal
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,  # reconfigure if something set handlers earlier
    )
    # Quiet some noisy libraries unless needed
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("wandb").setLevel(logging.INFO)
    logging.getLogger(__name__).info("Logging initialized -> %s", log_path)
    return log_path


def cmd_eval(args: argparse.Namespace) -> None:
    run_dir = _run_dir()
    print(f"Run dir: {run_dir}")
    _setup_logging(run_dir)
    logging.getLogger(__name__).info("Run dir: %s", run_dir)

    # 1) System info
    _sh(["bash", "scripts/collect_system_info.sh", str(run_dir)])

    # 2) Data prep (pins revision; also writes JSONL + optional HF snapshot)
    data_prep.main(
        out_dir=args.out_dir,
        seed=args.seed,
        eval_n=args.eval_n,
        revision=args.revision,
        cache_dir=args.cache_dir,
    )

    # 3) Manifest
    write_run_manifest(
        str(run_dir / "run_manifest.json"),
        extras={
            "model_id": args.model_id,
            "eval_path": args.eval_path,
            "revision": args.revision,
        },
    )

    # 4) Eval with vLLM
    vllm_eval_main(
        model_id=args.model_id,
        eval_path=args.eval_path,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        gpu_mem_util=args.gpu_mem_util,
        tp_size=args.tp_size,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
    )

    # 5) Lock/export the exact env used in this run
    (run_dir / "locks").mkdir(exist_ok=True, parents=True)
    _sh(
        [
            "uv",
            "export",
            "--format",
            "requirements-txt",
            "--frozen",
            "--output-file",
            str(run_dir / "locks" / "requirements.lock.txt"),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

    # 6) Snapshot any external data directory for provenance
    data_dir = os.environ.get("DATA_DIR", None)
    if data_dir is not None and os.path.exists(data_dir):
        _sh(["bash", "scripts/snapshot_dataset.sh", data_dir, str(run_dir)])
    else:
        print("DATA_DIR not set or does not exist; skipping dataset snapshot")


def cmd_sft(args: argparse.Namespace) -> None:
    run_dir = _run_dir()
    print(f"Run dir: {run_dir}")
    _setup_logging(run_dir)
    logging.getLogger(__name__).info("Run dir: %s", run_dir)

    # 1) System info
    _sh(["bash", "scripts/collect_system_info.sh", str(run_dir)])

    # 2) Manifest
    write_run_manifest(
        str(run_dir / "run_manifest.json"),
        extras={
            "model_id": args.model_id,
            "train_data_path": args.train_data_path,
            "note": "SFT run",
        },
    )

    # 3) W&B init
    wandb_kwargs: dict[str, Any] = {
        "project": args.wandb_project,
        "name": args.run_name,
        "config": {
            k: getattr(args, k)
            for k in [
                "model_id",
                "train_data_path",
                "eval_data_path",
                "microbatch_size",
                "max_total_tokens",
                "min_tokens_per_update",
                "num_epochs",
                "max_update_steps",
                "learning_rate",
                "adamw_beta1",
                "adamw_beta2",
                "adamw_eps",
                "weight_decay",
                "eval_every",
                "eval_examples",
                "vllm_device",
                "vllm_gpu_memory_util",
                "checkpoint_dir",
                "resume_from",
                "max_grad_norm",
                "gen_max_new_tokens",
                "gen_temperature",
                "gen_top_p",
                "model_dtype",
            ]
            if hasattr(args, k)
        },
    }
    if args.wandb_entity:
        wandb_kwargs["entity"] = args.wandb_entity
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode
    wandb.init(**wandb_kwargs)

    # Simple optional dtype mapping (auto if not provided)
    model_dtype: torch.dtype | None = None
    if args.model_dtype:
        _dtype_map = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        key = args.model_dtype.lower()
        if key not in _dtype_map:
            raise ValueError(f"Unsupported --model_dtype '{args.model_dtype}'")
        model_dtype = _dtype_map[key]

    # 4) Run SFT
    train_sft_on_r1_pairs(
        train_data_path=args.train_data_path,
        eval_data_path=getattr(args, "eval_data_path", None),
        model_id=args.model_id,
        device=args.device,
        vllm_device=args.vllm_device,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_util,
        microbatch_size=args.microbatch_size,
        num_epochs=args.num_epochs,
        min_tokens_per_update=args.min_tokens_per_update,
        max_update_steps=args.max_update_steps,
        learning_rate=args.learning_rate,
        adamw_beta1=args.adamw_beta1,
        adamw_beta2=args.adamw_beta2,
        adamw_eps=args.adamw_eps,
        weight_decay=args.weight_decay,
        max_total_tokens=args.max_total_tokens,
        eval_every=args.eval_every,
        eval_examples=args.eval_examples,
        gen_max_new_tokens=args.gen_max_new_tokens,
        gen_temperature=args.gen_temperature,
        gen_top_p=args.gen_top_p,
        model_dtype=model_dtype,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from,
        max_grad_norm=args.max_grad_norm,
    )

    # 5) Finish W&B
    wandb.finish()

    # 6) Lock/export environment
    (run_dir / "locks").mkdir(exist_ok=True, parents=True)
    _sh(
        [
            "uv",
            "export",
            "--format",
            "requirements-txt",
            "--frozen",
            "--output-file",
            str(run_dir / "locks" / "requirements.lock.txt"),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("eval", help="Run evaluation with unified logging & artifacts")
    e.add_argument("--model_id", default="Qwen/Qwen2.5-Math-1.5B")
    e.add_argument("--eval_path", default="artifacts/gsm8k/val.jsonl")
    e.add_argument("--limit", type=int, default=None)
    e.add_argument("--batch_size", type=int, default=8)
    e.add_argument("--max_new_tokens", type=int, default=384)
    e.add_argument("--gpu_mem_util", type=float, default=0.92)
    e.add_argument("--tp_size", type=int, default=1)
    e.add_argument("--wandb_project", default="grpo-gsm8k")
    e.add_argument("--run_name", default=None)
    # data prep args
    e.add_argument("--out_dir", default="artifacts/gsm8k")
    e.add_argument("--seed", type=int, default=31415)
    e.add_argument("--eval_n", type=int, default=800)
    e.add_argument("--revision", default="main")  # or a commit SHA
    e.add_argument("--cache_dir", default="/workspace/.cache/huggingface")
    e.set_defaults(func=cmd_eval)

    s = sub.add_parser("sft", help="Run SFT on R1 traces with W&B logging")
    s.add_argument("--train_data_path", default="artifacts/tokenized/*.pt")
    s.add_argument(
        "--eval_data_path",
        default="artifacts/tokenized/val_for_vllm.jsonl",
        help="Path to pre-rendered eval set (JSONL)",
    )
    s.add_argument("--model_id", default="Qwen/Qwen2.5-Math-1.5B")
    s.add_argument("--device", default="cuda:0")
    s.add_argument("--vllm_device", default="cuda:1")
    s.add_argument("--vllm_gpu_memory_util", type=float, default=0.85)
    s.add_argument("--microbatch_size", type=int, default=2)
    s.add_argument("--num_epochs", type=int, default=1)
    s.add_argument("--min_tokens_per_update", type=int, default=4096)
    s.add_argument("--max_update_steps", type=int, default=None)
    s.add_argument("--learning_rate", type=float, default=1e-5)
    s.add_argument("--adamw_beta1", type=float, default=0.9)
    s.add_argument("--adamw_beta2", type=float, default=0.95)
    s.add_argument("--adamw_eps", type=float, default=1e-8)
    s.add_argument("--weight_decay", type=float, default=0.0)
    s.add_argument("--max_grad_norm", type=float, default=1.0)
    s.add_argument("--max_total_tokens", type=int, default=2048)
    s.add_argument("--eval_every", type=int, default=4)
    s.add_argument("--eval_examples", type=int, default=None)
    s.add_argument("--gen_max_new_tokens", type=int, default=2048)
    s.add_argument("--gen_temperature", type=float, default=0.0)
    s.add_argument("--gen_top_p", type=float, default=1.0)
    s.add_argument(
        "--model_dtype",
        default=None,
        help="Optional dtype override: bfloat16|float16|float32 (auto if omitted).",
    )
    s.add_argument(
        "--checkpoint_dir", default=None, help="Directory to save model/tokenizer checkpoints"
    )
    s.add_argument(
        "--resume_from",
        default=None,
        help="Checkpoint directory or a root containing step_* subdirs to resume from",
    )
    # W&B
    s.add_argument("--wandb_project", default="grpo-gsm8k")
    s.add_argument("--wandb_entity", default=None)
    s.add_argument("--run_name", default=None, help="Optional run name for W&B logging")
    s.add_argument(
        "--wandb_mode",
        default=None,
        help="online|offline|disabled (sets WANDB_MODE)",
    )
    s.set_defaults(func=cmd_sft)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
