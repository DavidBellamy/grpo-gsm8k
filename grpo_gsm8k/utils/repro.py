"""
Reproducibility utilities for evals and training.

Usage
-----
from grpo_gsm8k.repro import (
    SEED, seed_everything, write_run_manifest, stable_hash
)

seed_everything(SEED, deterministic=False)  # True => stricter reproducibility (slower)
write_run_manifest("artifacts/baselines/run_manifest.json", extras={
    "model_id": "Qwen/Qwen2.5-Math-1.5B",
    "eval_path": "artifacts/gsm8k/val.jsonl",
})
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import platform
import random
import subprocess
from typing import Any

import numpy as np
import torch
import wandb

# ---- Single source of truth for RNG seed -------------------------------------

SEED: int = 31415

# ---- Helpers -----------------------------------------------------------------


def stable_hash(obj: Any) -> str:
    """Stable short SHA1 hash for dicts/lists/strings (useful for dataset splits, etc.)."""
    try:
        s = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=repr)
    except Exception:
        s = repr(obj)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def seed_everything(seed: int = SEED, deterministic: bool = False) -> int:
    """
    Seed Python, NumPy, and PyTorch (if available).
    If deterministic=True, enable stricter PyTorch determinism (slower, disables some fast kernels).

    Notes
    -----
    - Deterministic mode aims for bitwise stability on the same hardware/stack,
      but can still differ across GPU architectures or library versions.
    - Fast (non-deterministic) mode may enable TF32 and cuDNN benchmarking for speed.

    Returns
    -------
    int: the seed used.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Avoid tokenizer thread nondeterminism noise in logs/perf (not correctness).
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Optional: encourage deterministic cuBLAS matmuls if needed.
    if deterministic:
        # Note: 4096 uses more workspace.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # NumPy
    if np is not None:
        try:
            np.random.seed(seed)
        except Exception:
            pass

    # PyTorch
    if torch is not None:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            # Fast/strict math toggles
            if deterministic:
                # Disable TF32 and enforce deterministic algorithms.
                try:
                    torch.backends.cuda.matmul.allow_tf32 = False
                except Exception:
                    pass
                torch.use_deterministic_algorithms(True, warn_only=False)
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
            else:
                # Allow fast kernels
                try:
                    torch.backends.cuda.matmul.allow_tf32 = True
                except Exception:
                    pass
                torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    return seed


def _try_run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=2)
        return out.decode("utf-8", errors="ignore").strip()
    except Exception:
        return None


def write_run_manifest(extras: dict[str, Any] | None = None) -> None:
    """
    Log a small JSON manifest with environment, package, and GPU info.
    Expects an active wandb run
    """
    info: dict[str, Any] = {
        "seed": SEED,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM"),
        },
        "packages": {},
        "gpu": {},
    }

    # Torch / GPU info
    if torch is not None:
        try:
            info["packages"]["torch"] = torch.__version__
            if torch.cuda.is_available():
                info["gpu"]["name"] = torch.cuda.get_device_name(0)
                info["gpu"]["count"] = torch.cuda.device_count()
                info["gpu"]["capability"] = torch.cuda.get_device_capability(0)
                info["gpu"]["cuda_compiled"] = torch.version.cuda
                # TF32 state (if available)
                try:
                    info["gpu"]["allow_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
                except Exception:
                    pass
        except Exception:
            pass

    system_info(info)

    # Package versions
    for mod in ("vllm", "transformers", "tokenizers", "datasets", "flash_attn", "numpy"):
        try:
            m = __import__(mod)
            info["packages"][mod] = getattr(m, "__version__", "unknown")
        except Exception:
            continue

    if extras:
        info.update(extras)

    if getattr(wandb, "run", None) is None:
        logging.getLogger(__name__).info("W&B run not active; skipping manifest upload.")
        return

    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%SZ")
    artifact = wandb.Artifact(name="run-manifest", type="config")
    with artifact.new_file(f"run_manifest_{ts}.json", mode="w") as f:
        json.dump(info, f, indent=2, sort_keys=True, ensure_ascii=False)
    wandb.run.log_artifact(artifact)
    logging.getLogger(__name__).info("Logged run manifest to W&B (artifact=run-manifest).")


def system_info(info: dict[str, Any] | None = None) -> dict:
    if info is None:
        info = {}
    # Try driver info via nvidia-smi
    smi = _try_run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    if smi:
        info["gpu"] = info.get("gpu", {})
        info["gpu"]["nvidia_smi"] = smi

    return info
