# Copilot Instructions for grpo-gsm8k

Purpose: RL-style fine-tuning/evaluation scaffolding for math reasoning (GSM8K). Single-node focus with reproducible containers and pinned environments.

## Repo Map (what matters)
- `grpo_gsm8k/cli.py`: End-to-end runs (sysinfo → data prep → vLLM eval → env lock/snapshot).
- `grpo_gsm8k/gsm8k_eval.py`: Baseline eval via vLLM (greedy, temp=0 by default).
- `grpo_gsm8k/data_prep.py`: Fetch + pin GSM8K; write JSONL to `artifacts/gsm8k/`.
- `grpo_gsm8k/prompts.py`: Qwen-style chat templates & batch render helpers.
- `grpo_gsm8k/reward_fn.py`: Parse numeric answers and compute exact-match style rewards.
- Core utilities (backed by tests):
  - `tokenize.py`: prompt/output tokenize → concat → shifted `input_ids`/`labels` + `response_mask`.
  - `get_response_log_probs.py`: `(B,T)` log p(label|prefix) and optional per-token entropy.
  - `per_token_entropy.py`: stable per-time-step entropy via `log_softmax`.
  - `masked_normalize.py`: masked sum ÷ constant (bool or {0,1} mask).
  - `sft.py`: microbatch train step (masked NLL, grad-accum scaling, backward) + metadata.
  - `log_generations.py`: generate, decode response-only tokens, entropy/length stats, optional rewards.
- `tests/`: precise contracts using lightweight stubs. Treat as spec.
- `artifacts/`: Outputs at runtime (data, baselines, run manifests/locks, snapshots).

## Environment & Tooling
- Python 3.10; deps via `uv` (exact versions in `uv.lock`). Linters: `ruff`; types: `mypy`.
- Image built/pushed on `main` via `.github/workflows/build.yml` to GHCR.

## Dev Commands (copy/paste)
- Install dev deps: `uv sync --dev`
- Fast tests (no slow/GPU):
  ```bash
  PYTHONPATH=. uv run --no-project --with pytest,torch,transformers,datasets \
    python -m pytest -q -m "not slow"
  ```
- Lint/format/type-check:
  ```bash
  uv run ruff check . && uv run ruff format .
  uv run mypy grpo_gsm8k
  ```
- End-to-end eval (writes to `artifacts/`):
  ```bash
  python -m grpo_gsm8k.cli eval --limit 100
  ```

## Patterns & Conventions (critical)
- Tokenization (`tokenize_prompt_and_output`):
  - Tokenize prompt/output separately (no special tokens), concat; produce `(B, max_len-1)` `input_ids` and `labels` (left/right shift).
  - `response_mask`=1 for positions sourced from output tokens in `labels`; 0 for prompt/pad.
  - If no `pad_token_id`, fall back to `eos_token_id` or 0 for padding.
- Scoring (`get_response_log_probs`): `log_softmax(logits).gather(-1, labels[...,None]).squeeze(-1)` → `(B,T)`.
  - Optional: per-token entropy via `compute_entropy(logits)`.
- Entropy (`compute_entropy`): `-(softmax*log_softmax).sum(-1)` over vocab per time step.
- Masked reductions: `masked_normalize(tensor, mask, C, dim)` expects same shape; mask can be bool or {0,1}.
- Training step (`sft_microbatch_train_step`): masked sum of log-probs → NLL; scale by `1/gradient_accumulation_steps`; call `loss.backward()`; return `(loss, metadata)`.
- Generation logging (`log_generations`): use `model.generate(..., return_dict_in_generate=True, output_scores=True)`; exclude prompt portion; compute response lengths by counting non-pad tokens in the generated slice; average entropy from `out.scores` via `compute_entropy`.

## Data, Artifacts, Repro
- Data JSONL under `artifacts/gsm8k/{train,val,test}.jsonl`.
- Run dirs: `artifacts/runs/<UTCSTAMP>_<gitsha>/` with `sys/`, `locks/requirements.lock.txt`, and `run_manifest.json`.

## Integration Points
- Transformers/vLLM for modeling; HuggingFace Datasets for data; optional Weights & Biases.
- DeepSeek R1 traces via `grpo_gsm8k/r1_traces.py` (requires `DEEPSEEK_API_KEY`).

## Guidance for AI Edits
- Match test contracts exactly; keep public APIs stable; prefer surgical changes.
- Don’t alter dep versions; use `uv.lock`. Write outputs to `artifacts/`; honor `cache_dir` args.
- Keep helpers pure where possible; leave side effects to CLI orchestration.

If any conventions are unclear during changes, note them in the PR to refine this doc.

## Python & Typing (3.10)
- Use PEP 585 builtins for generics: `list[str]`, `dict[str, Any]`, `tuple[int, ...]`, `set[T]`.
  - Never use `typing.Dict`, `typing.List`, `typing.Optional`, or `typing.Union`.
  - Prefer `X | Y | None` over `Optional[X | Y]`.
- Assume Python 3.10. Prefer:
  - Structural typing (`TypedDict`, `Protocol`) when useful; `dataclasses.dataclass` for records.
  - `pathlib.Path` for filesystem paths.
  - f-strings; avoid `.format()` except when unavoidable.
- Default typing imports:
  ```python
  from __future__ import annotations
  from typing import Any, Iterable, Iterator, Mapping, MutableMapping, Sequence
  ```
- Add precise types for all public functions (`disallow_untyped_defs = true` in mypy).
- Prefer returning lightweight `dataclass`/`TypedDict` over raw nested dicts when shape matters.

## Ruff & Style
- Max line length: 100.
- Follow Ruff rules: `E,F,I,UP,ARG`.
- Write code that satisfies pyupgrade (e.g., `dict[...]` not `Dict[...]`; `|` unions).
- Remove unused args with leading underscore if needed.
- Imports: stdlib, then third-party, then first-party (`grpo_gsm8k`), alphabetized per group.
- Use logging instead of print:
  ```python
  import logging
  logger = logging.getLogger(__name__)

  def setup_logging(level: int = logging.INFO) -> None:
      logging.basicConfig(
          level=level,
          format="%(asctime)s %(levelname)s %(name)s: %(message)s",
      )
  ```

## Testing (pytest)
- Tests live in `tests/`, quiet `-q` by default; respect markers.
- Use `@pytest.mark.slow` for long runs; keep unit tests fast.
- Use `@pytest.mark.serial` when tests must not run concurrently.
- Prefer functional tests that mock I/O and avoid GPUs by default.
- For GPU behavior: gate with `pytest.importorskip("torch")` and `torch.cuda.is_available()`.
- Provide small, deterministic fixtures; seed torch/numpy when relevant.

## vLLM / Transformers
- Prefer `transformers` for configs/tokenizers; keep sampling params explicit (temperature, top_p, max_tokens).
- For vLLM, surface engine knobs (tensor parallel, kv-cache) via CLI args; keep defaults reasonable.
- Raise specific exceptions; avoid `bare except:`. Convert exceptions to user messages at the CLI layer only.

## Data & I/O
- Use `datasets` for data loading when possible; type example dicts with `TypedDict` if stable.
- Use `pathlib.Path` ops (`read_text`, `write_text`, `mkdir(parents=True, exist_ok=True)`).
- Keep file encodings explicit: `encoding="utf-8"`.

## W&B rank-0 logging
```python
def log_metrics(step: int, metrics: dict[str, float], is_main: bool = True) -> None:
    if is_main:
        wandb.log(metrics, step=step)
```
