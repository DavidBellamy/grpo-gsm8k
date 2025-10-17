"""
Hydra-driven CLI for SFT and evaluation workflows.

- Configs live under: grpo_gsm8k/conf/
- Top-level switch: `command` in config (e.g., eval or sft)
- Example:
    python scripts/cli.py                          # uses defaults (command=eval)
    python scripts/cli.py command=sft                      # run SFT with defaults
    python scripts/cli.py command=eval eval.limit=100 eval.model_path=Qwen/Qwen2.5-Math-1.5B
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from grpo_gsm8k.utils.repro import write_run_manifest


def _setup_logging(run_dir: Path, level: int = logging.INFO) -> Path:
    """Configure root logging to stream to stderr and write a file under the run directory."""
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
        force=True,
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("wandb").setLevel(logging.INFO)
    logging.getLogger(__name__).info("Logging initialized -> %s", log_path)
    return log_path


def _flatten_dict(d: Mapping[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten nested mappings into a single level with dotted keys for easy W&B filtering."""
    items: dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, Mapping):
            items.update(_flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def _log_hydra_config_artifact(run: Any, cfg: DictConfig, _run_dir: Path) -> None:
    """Serialize Hydra config to YAML and log as a W&B artifact."""
    ts = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    yaml_text = OmegaConf.to_yaml(cfg)

    artifact = wandb.Artifact(name="hydra-config", type="config")
    # Write directly into the artifact's staging area (no project-local file)
    with artifact.new_file(f"hydra_config_{ts}.yaml", mode="w") as f:
        f.write(yaml_text)
    run.log_artifact(artifact)
    logging.getLogger(__name__).info("Logged Hydra config YAML to W&B artifact")


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Hydra changes the working directory to its run dir. Log within that directory.
    _setup_logging(Path("."), logging.INFO)

    # Convert config to plain dicts
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(cfg_dict, dict)

    command = cfg_dict.get("command", None)
    if command not in {"eval", "sft", "data", "traces"}:
        raise ValueError(f"Unknown command: {command}")

    # Sub-config for the selected command
    sub_cfg: dict[str, Any] = cfg_dict[command]

    # W&B config lives under a dedicated namespace
    wb_cfg: dict[str, Any] = cfg_dict.get("wandb", {}) or {}
    project = wb_cfg.get("project", "grpo-gsm8k")
    run_name = wb_cfg.get("run_name")
    entity = wb_cfg.get("entity")

    # Log the entire (flattened) config to W&B for easy filtering
    wb_config = _flatten_dict(cfg_dict)

    run = wandb.init(project=project, name=run_name, entity=entity, config=wb_config)
    _log_hydra_config_artifact(run, cfg, Path("."))
    write_run_manifest(extras={"command": command})

    try:
        if command == "eval":
            from grpo_gsm8k.evaluation.eval import main as eval_main

            eval_main(**sub_cfg)
        elif command == "data":
            from grpo_gsm8k.data.data_prep import main as data_main

            data_main(**sub_cfg)
        elif command == "traces":
            from grpo_gsm8k.traces.format_r1_traces import main as format_traces
            from grpo_gsm8k.traces.r1_traces import main as gen_traces

            asyncio.run(gen_traces(**sub_cfg["gen_traces"]))
            format_traces(**sub_cfg["format_traces"])
        else:  # command == "sft"
            from grpo_gsm8k.training.sft import train_sft_on_r1_pairs

            train_sft_on_r1_pairs(**sub_cfg)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
