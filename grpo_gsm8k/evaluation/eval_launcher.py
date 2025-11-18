import logging
import multiprocessing as mp
import os
import subprocess
from inspect import signature
from typing import Any

import wandb

from grpo_gsm8k.evaluation.aggregate_gsm8k_results import run_aggregate
from grpo_gsm8k.evaluation.eval import main

logger = logging.getLogger(__name__)


def get_visible_gpu_ids() -> list[str]:
    """Return list of visible CUDA device IDs as strings."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        # Respect pre-filtering done by the job launcher / cluster
        ids = [x.strip() for x in env.split(",") if x.strip()]
        return ids

    # Fallback: query nvidia-smi if CUDA_VISIBLE_DEVICES is not set
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            encoding="utf-8",
        )
        ids = [line.strip() for line in out.splitlines() if line.strip()]
        return ids
    except Exception:
        # If this fails, be conservative
        return []


def _worker_eval(shard_id: int, gpu_id: str, cfg: dict[str, Any]) -> None:
    num_shards = cfg["num_shards"]
    base_port = cfg["base_port"]

    # Disable W&B for this subprocess
    os.environ["WANDB_DISABLED"] = "true"

    # Bind this worker to a single GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    logger.info(
        f"[worker] shard_id={shard_id} num_shards={cfg['num_shards']} "
        f"bound_to_gpu={gpu_id} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )

    # Optionally make sure workers never talk to W&B, regardless of env
    os.environ.setdefault("WANDB_DISABLED", "true")

    base_kwargs: dict[str, Any] = cfg
    base_kwargs["shard_id"] = shard_id
    base_kwargs["num_shards"] = num_shards
    base_kwargs["server_port"] = base_port + shard_id

    sig = signature(main)
    allowed = set(sig.parameters.keys())
    kwargs = {k: v for k, v in base_kwargs.items() if k in allowed}

    main(**kwargs)


def launch_eval(cfg: dict[str, Any]) -> None:
    num_shards = cfg["num_shards"]
    gpu_ids = get_visible_gpu_ids()
    num_gpus = len(gpu_ids)

    if num_shards > num_gpus:
        raise RuntimeError(
            f"Requested num_shards={num_shards}, but only {num_gpus} GPUs visible: "
            f"{gpu_ids!r}. Either reduce num_shards or expose more GPUs."
        )

    if num_shards <= 1:
        # single-process path, W&B enabled
        base_kwargs: dict[str, Any] = cfg
        base_kwargs["shard_id"] = 0
        base_kwargs["server_port"] = cfg["base_port"]

        sig = signature(main)
        allowed = set(sig.parameters.keys())
        kwargs = {k: v for k, v in base_kwargs.items() if k in allowed}

        main(**kwargs)
        return

    # Multi-process data-parallel
    procs: list[mp.Process] = []
    for shard_id in range(num_shards):
        gpu_id = gpu_ids[shard_id]
        p = mp.Process(target=_worker_eval, args=(shard_id, gpu_id, cfg))
        p.start()
        procs.append(p)

    # Wait for all workers
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Shard process {p.pid} exited with code {p.exitcode}")

    eval_suites = cfg.get("eval_suites") or ["all"]
    run_gsm8k = ("all" in eval_suites) or ("gsm8k" in eval_suites)
    if run_gsm8k:
        run_aggregate(cfg)

    wandb.run.finish()
