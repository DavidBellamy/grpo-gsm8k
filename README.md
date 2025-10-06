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

## Pre-tokenize R1 SFT pairs (optional)

Before SFT, pre-tokenize the prompt/response pairs from `artifacts/r1_sft_pairs.jsonl` into fixed-length tensor shards that match the trainer’s expectations in [grpo_gsm8k/sft.py](grpo_gsm8k/sft.py). This uses the same chat rendering and tokenization helpers as training: [`grpo_gsm8k.prompts.render_batch`](grpo_gsm8k/prompts.py) and [`grpo_gsm8k.tokenize.tokenize_prompt_and_output`](grpo_gsm8k/tokenize.py).

- Tokenize to `.pt` shards (N, T) with `input_ids`, `labels`, `response_mask`, plus `pad_token_id` and `meta`:

```bash
uv run python scripts/shell/pretokenize_r1_pairs.py \
  --model_id "Qwen/Qwen2.5-Math-1.5B" \
  --infile artifacts/r1_sft_pairs.jsonl \
  --out_dir artifacts/tokenized \
  --max_total_tokens 2048
```

- Notes:
  - Left padding; set T to your train `--max_total_tokens` (default 2048).
  - Build `attention_mask` at load time as `(input_ids != pad_token_id).long()`.
  - Format aligns with the trainer’s per-step tensors; you can stream microbatches directly to GPU.

---

## Prepare GSM8K validation set for vLLM evaluation (required for SFT)

To evaluate Qwen checkpoints during SFT, you must pre-render the GSM8K validation set questions into Qwen chat prompts and parse the ground truth numeric answers. This is required for async vLLM eval during SFT.

- Prepare the eval set:

```bash
uv run python scripts/shell/prep_val_for_vllm.py \
  --model_id "Qwen/Qwen2.5-Math-1.5B" \
  --infile artifacts/gsm8k/val.jsonl \
  --outfile artifacts/tokenized/val_for_vllm.jsonl
```

- The output JSONL contains:
  - `prompt`: Qwen chat-templated prompt for vLLM generation
  - `gold`: original ground truth answer string (for logging)
  - `gold_num`: normalized numeric answer (for exact-match evaluation)

- **Note:** SFT requires this pre-rendered eval set. If not provided, training will halt with an error.

---

## SFT with strict async vLLM evaluation

During SFT, async vLLM evaluation is always enabled and strictly requires a pre-rendered eval set (`artifacts/tokenized/val_for_vllm.jsonl`). No fallback or disabling is allowed.

- Example SFT run:

```bash
python -m grpo_gsm8k.cli sft \
  --data_path artifacts/r1_sft_pairs.jsonl \
  --model_id Qwen/Qwen2.5-Math-1.5B \
  --vllm_device cuda:1 \
  --eval_set_path artifacts/tokenized/val_for_vllm.jsonl \
  --eval_every 4 \
  --eval_examples 64
```

- If `--eval_set_path` is missing or invalid, training will halt with an error.

- During evaluation, the trainer:
  - Uses only the pre-rendered prompts and numeric golds for vLLM generation and exact-match.
  - Logs the original ground truth answer string alongside Qwen-generated reasoning to W&B.

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


4. **Secrets**: store `WANDB_API_KEY` as a secret.
5. **Command**: default `bash` is fine; the image’s `entrypoint.sh` will set up env + venv.

---

## Portable usage (works anywhere the image runs)

### One-shot pipeline

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

### Prepare data (defaults)

```bash
python -m grpo_gsm8k.data_prep
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

## Developer setup (editable install + tests)

From project root:

```bash
# Runtime deps only
uv sync
# Add dev tools (pytest, ruff, mypy, pre-commit)
uv sync --dev
uv pip install -e .
# Run full test suite
pytest
# Or quiet
pytest -q
```

+ To include slow tests (those marked with @pytest.mark.slow), run:
+
+ ```bash
+ pytest --runslow
+ ```
+
+ To run only the slow tests:
+ ```bash
+ pytest -m slow --runslow
+ ```

Alternative one-shot (no editable install, just add src on path):

```bash
PYTHONPATH=. uv run python -m pytest -q
```

Lint / format / type-check:

```bash
ruff check .
ruff format .
mypy grpo_gsm8k
```

---

## Developer tooling & tests (summary)

| Task | Command |
|------|---------|
| Install runtime | uv sync |
| Install runtime + dev | uv sync --dev |
| Editable install | uv pip  install -e . |
| Run tests | pytest |
| Run tests (incl. slow) | pytest --runslow |
| Lint | ruff check . |
| Format | ruff format . |
| Type-check | mypy grpo_gsm8k |

---

## Artifacts & reproducibility

* Run dirs: `artifacts/runs/<UTCSTAMP>_<gitsha>/`
* Datasets: `artifacts/gsm8k/{train,val,test}.jsonl`
* Baselines: `artifacts/baselines/*.jsonl`

Determinism: set `NVIDIA_TF32_OVERRIDE=0` and seed via `repro.py` if needed.

---

## Project layout

```
grpo-gsm8k/
├── grpo_gsm8k/
├── tests/
├── artifacts/
└── pyproject.toml
```

---

## Troubleshooting

* Module not found: ensure `uv pip install -e .` or run with `PYTHONPATH=.`
* Dev deps missing: run `uv sync --dev`
* CUDA OOM: lower `--gpu_mem_util` or use smaller model

---

## License

MIT — see `LICENSE`.
