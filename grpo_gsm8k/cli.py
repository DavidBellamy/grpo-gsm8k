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
    --eval_path         Path to evaluation data (default: "artifacts/gsm8k/test_eval.jsonl").
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

from grpo_gsm8k import data_prep
from grpo_gsm8k.fast_eval_vllm import main as vllm_eval_main
from grpo_gsm8k.repro import write_run_manifest


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
        ]
    )

    # 6) Snapshot any external data directory for provenance
    data_dir = os.environ.get("DATA_DIR", None)
    if data_dir is not None and os.path.exists(data_dir):
        _sh(["bash", "scripts/snapshot_dataset.sh", data_dir, str(run_dir)])
    else:
        print("DATA_DIR not set or does not exist; skipping dataset snapshot")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("eval", help="Run evaluation with unified logging & artifacts")
    e.add_argument("--model_id", default="Qwen/Qwen2.5-7B-Instruct")
    e.add_argument("--eval_path", default="artifacts/gsm8k/test_eval.jsonl")
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

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
