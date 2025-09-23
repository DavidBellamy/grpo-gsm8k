# Simple RL for Math Reasoning

⚠️ Under construction. More soon!

This project is for SFT, expert iteration, and RL post-training of Qwen models on GSM8k problems. It is designed for single-node jobs.

Includes: **reproducible containers**, **determinism toggles**, **dataset pinning/snapshots**, and a **vLLM baseline eval**.

---

## Quickstart — **no build required**

Most users can run the **prebuilt public image** from GHCR.

```bash
# Use the public image
IMAGE=ghcr.io/davidbellamy/grpo-gsm8k:latest
# (Optionally pin by digest for immutability)
# IMAGE=ghcr.io/davidbellamy/grpo-gsm8k@sha256:<digest>

# Run a baseline eval with GPU
docker run --rm -it --gpus all \
  -v "$PWD":/workspace \
  -v "$HOME/.cache/huggingface":/workspace/.cache/huggingface \
  -e WANDB_MODE=offline \
  "$IMAGE" \
  python -m grpo_gsm8k.cli eval --limit 100
```

This will:

* snapshot system info to `artifacts/runs/<timestamp>_<gitsha>/sys/`
* prepare GSM8K JSONL files to `artifacts/gsm8k/`
* run a vLLM eval and write to `artifacts/baselines/`

> **GPU required** for evaluation.

---

## Generate GSM8k Splits

We reserve 512 problems from the original training set for the validation split. 

```python
python -m grpo_gsm8k.data_prep \
  --out-dir artifacts/gsm8k \
  --eval-n 512 \
  --seed 31415 \
  --revision main \
  --cache-dir /workspace/.cache/huggingface \
  --snapshot-dir artifacts/gsm8k_hf_snapshot
```

## Generate DeepSeek R1 Reasoning Traces

We generate DeepSeek R1 reasoning traces for problems in the GSM8k train set via DeepSeek's official API. You need to create an account and get your API key. This command can be run locally:

```bash
export DEEPSEEK_API_KEY=...

uv run --no-project --with aiohttp \
  python grpo_gsm8k/r1_traces.py \
    --infile artifacts/gsm8k/train.jsonl \
    --outfile artifacts/deepseek_r1_gsm8k_traces.jsonl \
    --concurrency 4 \
    --max-tokens 2048 \
    --max-retries 5 \
    --offpeak 0.25 \
    --limit 5
```

---

## Run on RunPod (provider notes)

The container is provider-agnostic. For RunPod specifically:

1. **Network Volume**: rent one in the same region as your pod.
2. **Launch a GPU pod** (H100 SXM or similar). Set the pod template to `rainbow_unicorn`.
3. **Edit the pod template's env vars**:

   ```bash
  VSCODE_TUNNEL_NAME=runpod-gpu
  XDG_CONFIG_HOME=/workspace/.config
  VSCODE_AGENT_FOLDER=/workspace/.vscode-server
  UV_PROJECT_ENVIRONMENT=/workspace/.venv
  XDG_CACHE_HOME=/workspace/.cache
  HF_HOME=/workspace/.cache/huggingface
  PIP_CACHE_DIR=/workspace/.cache/pip
  GIT_CONFIG_GLOBAL=/workspace/dotfiles/gitconfig
   ```

On the first pod you deploy, run:
```bash
  git config -f /workspace/dotfiles/gitconfig user.name "your_name"
  git config -f /workspace/dotfiles/gitconfig user.email "123456+youruser@users.noreply.github.com"
  git config -f /workspace/dotfiles/gitconfig user.useConfigOnly true
```


4. **Secrets**: store `WANDB_API_KEY` as a secret (don’t expose as plain env var).
5. **Command**: default `bash` is fine; the image’s `entrypoint.sh` will:

   * load determinism/env shims (`/etc/profile.d/env.det.sh`)
   * ensure `/workspace` exists
   * set caches to `/workspace/.cache`
   * create/activate a persistent venv at `/workspace/.venv` (via `uv`)
   * (best-effort) start a VS Code tunnel if `VSCODE_TUNNEL_NAME` is set

Then exec into the pod or use the tunnel and run the same command as in **Quickstart**.

> **What’s RunPod-specific here?** Only **how** you attach a network volume at `/workspace` and **where** you set env vars/secrets in the RunPod UI. The env names themselves are generic.

---

## Portable usage (works anywhere the image runs)

### One-shot pipeline (sysinfo → data prep → vLLM eval → locks → data snapshot)

```bash
python -m grpo_gsm8k.cli eval \
  --model_id "Qwen/Qwen2.5-7B-Instruct" \
  --eval_path "artifacts/gsm8k/val.jsonl" \
  --limit 800 \
  --max_new_tokens 384 \
  --gpu_mem_util 0.92 \
  --tp_size 1 \
  --wandb_project "grpo-gsm8k" \
  --run_name "qwen25_7b_eval" \
  --out_dir "artifacts/gsm8k" \
  --seed 31415 \
  --eval_n 800 \
  --revision "main" \
  --cache_dir "/workspace/.cache/huggingface"
```

### Prepare data only (defaults)

```bash
python -m grpo_gsm8k.data_prep
```

### Prepare data with custom args

```bash
python - <<'PY'
from grpo_gsm8k.data_prep import main
main(out_dir="artifacts/gsm8k", seed=31415, eval_n=800, revision="main",
     cache_dir="/workspace/.cache/huggingface")
PY
```

### vLLM eval only

```bash
python -m grpo_gsm8k.fast_eval_vllm \
  --model_id Qwen/Qwen2.5-7B-Instruct \
  --eval_path artifacts/gsm8k/val.jsonl \
  --limit 200 \
  --max_new_tokens 384 \
  --gpu_mem_util 0.92
```

---

## Developer setup (only if you’re modifying the image or code)

These steps are for contributors. End users can stick to the **Quickstart**.

The stack: this repo uses `uv` for Python packages, `ruff` for linting/formatting, `mypy` for type checking, `docker` for building container images, `wandb` for logging experiments, `pytest` for tests, and is designed to run on Ubuntu 22.04 ("Jammy") with CUDA 12.8 with Python 3.10, PyTorch 2.7 and vLLM 0.10.

### Local tooling

```bash
# macOS (example)
brew install docker uv python@3.10

# clone
git clone https://github.com/DavidBellamy/grpo-gsm8k.git
cd grpo-gsm8k

# pre-commit hooks (optional but recommended)
uv tool install pre-commit
uv tool run pre-commit install
uv tool run pre-commit run --all-files

# lockfile for reproducible Python deps
uv lock --python 3.10
```

### Build the image (optional)

Note: it is much quicker to build the image and push it to GHCR on a GitHub runner via GitHub Actions.

```bash
# Generic build (host arch)
docker build -t grpo-gsm8k .

# Cross-build for linux/amd64 (useful on Apple Silicon when targeting x86_64 GPU hosts)
docker buildx build --platform=linux/amd64 -t grpo-gsm8k .
```

### Publish to GHCR (optional)

```bash
# login (requires PAT with write:packages)
echo <YOUR_GITHUB_PAT> | docker login ghcr.io -u <GITHUB_USERNAME> --password-stdin

# tag & push
docker tag grpo-gsm8k:latest ghcr.io/<GITHUB_USERNAME>/grpo-gsm8k:latest
docker push ghcr.io/<GITHUB_USERNAME>/grpo-gsm8k:latest

# alt: build & push in one step
# docker buildx build --platform=linux/amd64 \
#   -t ghcr.io/<GITHUB_USERNAME>/grpo-gsm8k:latest \
#   --push .
```

---

## Artifacts & reproducibility

* **Run dirs:** `artifacts/runs/<UTCSTAMP>_<gitsha>/`

  * `sys/…` — driver/GPU/topo, env, `pip.freeze`, timestamps, etc.
  * `run_manifest.json` — environment + package versions
  * `locks/requirements.lock.txt` — exact deps used (`uv export --strict`)
  * `data.sha256` — checksums for `/data` (from `scripts/snapshot_dataset.sh`)
* **Datasets:** `artifacts/gsm8k/{train,test,val}.jsonl` and optional HF snapshot `artifacts/gsm8k_hf_snapshot/`
* **Baselines:** `artifacts/baselines/*.jsonl` — inputs, model outputs, reward, gold

**Determinism toggles**

* Container defaults favor speed (`NVIDIA_TF32_OVERRIDE=1` in the Dockerfile). For stricter reproducibility at runtime set `NVIDIA_TF32_OVERRIDE=0` and call `seed_everything(..., deterministic=True)` (see `repro.py`).
* Additional env defaults in `scripts/env.det.sh`:
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `TOKENIZERS_PARALLELISM=false`, `PYTHONHASHSEED=0`, `OMP_NUM_THREADS=1`.

---

## Project layout

```
grpo-gsm8k/
├── grpo_gsm8k/
│   ├── cli.py                # sysinfo → data prep → vLLM eval → locks → snapshots
│   ├── data_prep.py          # fetch + pin GSM8K; write JSONL and optional HF snapshot
│   ├── fast_eval_vllm.py     # baseline eval via vLLM (greedy, temp=0)
│   ├── prompts.py            # Qwen-style chat templates; batch rendering helpers
│   ├── repro.py              # seeds, manifests, sampling params, env capture
│   └── reward_fn.py          # parse \boxed{...}; exact-match vs GSM8K gold
├── scripts/                  # sysinfo, env, locks, dataset snapshot, remote sync (rclone/B2)
├── tests/
├── Dockerfile                # CUDA 12.8, uv, snapshot-locked apt
├── pyproject.toml            # deps, linters, pytest config, uv cu128 index
└── artifacts/                # outputs (created at runtime)
```

---

## Developer tooling & tests

```bash
uv sync --dev
pre-commit install
pre-commit run --all-files
```

Run (non-slow) tests locally with:
```bash
PYTHONPATH=. uv run --no-project \
  --with pytest,pytest-asyncio,aiohttp,datasets,torch,transformers,vllm \
  python -m pytest -q
```

Run parity tests in the same environment as RunPod (container, no GPU) with:
```bash
docker run --rm -it -v "$PWD":/workspace ghcr.io/davidbellamy/grpo-gsm8k:latest \
  bash -lc 'pytest -q -m "not slow"'
```

Run all tests (incl. ones needing GPU) with:
```bash
docker run --rm -it --gpus all -v "$PWD":/workspace ghcr.io/davidbellamy/grpo-gsm8k:latest \
  bash -lc 'pytest -q'
```

---

## Troubleshooting

* **No GPUs in Docker**: install/configure NVIDIA Container Toolkit; run with `--gpus all`.
* **CUDA OOM**: reduce `--max_new_tokens`, lower `--gpu_mem_util` (e.g., `0.80`), increase `--tp_size`, or switch to a smaller model.
* **Slow downloads**: mount a persistent HF cache (`-v $HOME/.cache/huggingface:/workspace/.cache/huggingface`).
* **Pin dataset revision**: pass `--revision <commit-or-tag>` to `cli eval` (propagates to `datasets.load_dataset`).
* **WANDB offline/online**: use `WANDB_MODE=offline` locally; store `WANDB_API_KEY` as a secret on your provider.

---

## License

MIT — see [`LICENSE`](LICENSE).

---
