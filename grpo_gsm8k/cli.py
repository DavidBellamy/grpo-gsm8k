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
    --model_id          Model identifier (default: "Qwen/Qwen2.5-7B-Instruct").
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


def cmd_eval(args: argparse.Namespace) -> None:
    run_dir = _run_dir()
    print(f"Run dir: {run_dir}")

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

    # 1) System info
    _sh(["bash", "scripts/collect_system_info.sh", str(run_dir)])

    # 2) Manifest
    write_run_manifest(
        str(run_dir / "run_manifest.json"),
        extras={
            "model_id": args.model_id,
            "data_path": args.data_path,
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
                "data_path",
                "microbatch_size",
                "gradient_accumulation_steps",
                "num_epochs",
                "max_steps",
                "learning_rate",
                "adamw_beta1",
                "adamw_beta2",
                "adamw_eps",
                "weight_decay",
                "max_total_tokens",
                "log_every",
                "eval_every",
                "eval_examples",
                "vllm_device",
                "vllm_gpu_memory_util",
                "max_grad_norm",
                "teacher_eval_every",
                "teacher_eval_examples",
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
    def _wb_log(step: int, metrics: dict[str, float]) -> None:
        wandb.log(metrics, step=step)

    def _on_eval(step: int, gen_log: dict[str, Any]) -> None:
        per = gen_log.get("per_example", [])
        if not per:
            return
        table = wandb.Table(columns=["step", "prompt", "response", "length", "entropy"])
        limit = min(len(per), getattr(args, "log_examples", 4))
        for row in per[:limit]:
            table.add_data(
                step,
                row.get("prompt", ""),
                row.get("response", ""),
                row.get("length", 0),
                row.get("mean_token_entropy", 0.0),
            )
        wandb.log({"eval/examples": table}, step=step)

    result = train_sft_on_r1_pairs(
        data_path=args.data_path,
        model_id=args.model_id,
        device=args.device,
        microbatch_size=args.microbatch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        adamw_beta1=args.adamw_beta1,
        adamw_beta2=args.adamw_beta2,
        adamw_eps=args.adamw_eps,
        weight_decay=args.weight_decay,
        max_total_tokens=args.max_total_tokens,
        log_every=args.log_every,
        eval_every=args.eval_every,
        eval_examples=args.eval_examples,
        vllm_device=args.vllm_device,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_util,
        wb_log=_wb_log,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        resume_from=args.resume_from,
        max_grad_norm=args.max_grad_norm,
        teacher_eval_every=args.teacher_eval_every,
        teacher_eval_examples=args.teacher_eval_examples,
        gen_max_new_tokens=args.gen_max_new_tokens,
        gen_temperature=args.gen_temperature,
        gen_top_p=args.gen_top_p,
        model_dtype=model_dtype,
    )

    # 5) Final metrics
    wandb.log(result)
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
    e.add_argument("--model_id", default="Qwen/Qwen2.5-7B-Instruct")
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
    s.add_argument("--data_path", default="artifacts/r1_sft_pairs.jsonl")
    s.add_argument("--model_id", default="Qwen/Qwen2.5-Math-1.5B")
    s.add_argument("--device", default="cuda:0")
    s.add_argument("--microbatch_size", type=int, default=2)
    s.add_argument("--gradient_accumulation_steps", type=int, default=8)
    s.add_argument("--num_epochs", type=int, default=1)
    s.add_argument("--max_steps", type=int, default=None)
    s.add_argument("--learning_rate", type=float, default=1e-5)
    s.add_argument("--adamw_beta1", type=float, default=0.9)
    s.add_argument("--adamw_beta2", type=float, default=0.95)
    s.add_argument("--adamw_eps", type=float, default=1e-8)
    s.add_argument("--weight_decay", type=float, default=0.0)
    s.add_argument("--max_total_tokens", type=int, default=2048)
    s.add_argument("--log_every", type=int, default=10)
    s.add_argument("--eval_every", type=int, default=200)
    s.add_argument("--eval_examples", type=int, default=4)
    s.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Clip gradient norm to this value (>0). Set <=0 to disable.",
    )
    s.add_argument(
        "--teacher_eval_every",
        type=int,
        default=None,
        help="Frequency (steps) of teacher-forced eval on training GPU (defaults to --eval_every).",
    )
    s.add_argument(
        "--teacher_eval_examples",
        type=int,
        default=4,
        help="Number of examples for teacher-forced eval.",
    )
    s.add_argument(
        "--gen_max_new_tokens",
        type=int,
        default=128,
        help="Max new tokens for async vLLM generation.",
    )
    s.add_argument(
        "--gen_temperature",
        type=float,
        default=0.0,
        help="Temperature for async vLLM generation.",
    )
    s.add_argument(
        "--gen_top_p",
        type=float,
        default=1.0,
        help="Top-p for async vLLM generation.",
    )
    s.add_argument(
        "--vllm_device",
        default="cuda:1",
        help="Device for the persistent vLLM generation worker (set to None to disable).",
    )
    s.add_argument(
        "--vllm_gpu_memory_util",
        type=float,
        default=0.85,
        help="GPU memory utilization fraction for vLLM worker.",
    )
    s.add_argument(
        "--model_dtype",
        default=None,
        help="Optional dtype override: bfloat16|float16|float32 (auto if omitted).",
    )
    # W&B
    s.add_argument("--wandb_project", default="grpo-gsm8k")
    s.add_argument("--wandb_entity", default=None)
    s.add_argument("--run_name", default=None)
    s.add_argument(
        "--wandb_mode",
        default=None,
        help="online|offline|disabled (sets WANDB_MODE)",
    )
    # Checkpoints and example logging
    s.add_argument(
        "--checkpoint_dir",
        default=None,
        help="Directory to save model/tokenizer checkpoints",
    )
    s.add_argument(
        "--checkpoint_every",
        type=int,
        default=None,
        help="Save a checkpoint every N steps",
    )
    s.add_argument(
        "--log_examples",
        type=int,
        default=4,
        help="Max number of eval examples to log to W&B per eval",
    )
    s.add_argument(
        "--resume_from",
        default=None,
        help="Checkpoint directory or a root containing step_* subdirs to resume from",
    )
    s.set_defaults(func=cmd_sft)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
