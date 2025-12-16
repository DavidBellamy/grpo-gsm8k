import argparse
import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from grpo_gsm8k.data.prompts import render_batch


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def trigrams(token_ids: Iterable[int]) -> list[tuple[int, int, int]]:
    ids = list(token_ids)
    return [(ids[i], ids[i + 1], ids[i + 2]) for i in range(len(ids) - 2)]


def jaccard(a: set[tuple[int, int, int]], b: set[tuple[int, int, int]]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def main(
    val_path: Path,
    model: str,
    m: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    out_csv: Path,
    limit: int | None,
) -> None:
    rows = load_jsonl(val_path)
    if limit is not None:
        rows = rows[: limit]

    # Tokenizer path: for HF id, use it; for local ckpt dir, try its tokenizer subdir; else fallback
    model_path = model
    model_dir = Path(model_path)
    if model_dir.exists() and model_dir.is_dir():
        tok_path = (
            model_dir / "tokenizer" if (model_dir / "tokenizer").exists() else model_dir  # fallback
        )
    else:
        tok_path = model_path

    tok = AutoTokenizer.from_pretrained(str(tok_path), use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token or "<|pad|>"
    tok.padding_side = "left"

    questions = [r["question"] for r in rows]
    prompts = render_batch(
        tok, questions, add_generation_prompt=True
    )  # grpo_gsm8k.data.prompts.render_batch

    llm = LLM(model=model_path, dtype="bfloat16", gpu_memory_utilization=0.92)
    samp = SamplingParams(
        n=m,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        logprobs=0,  # not needed here
        prompt_logprobs=0,
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "distinct3",
                "avg_pairwise_jaccard",
                "num_samples",
                "total_trigrams",
                "unique_trigrams",
                "temperature",
                "top_p",
                "max_new_tokens",
            ],
        )
        w.writeheader()

        # Batch generate to leverage vLLM, but compute metrics per prompt
        # We iterate prompts to keep the per-prompt grouping clear.
        for i, (p, r) in enumerate(zip(prompts, rows)):
            outs = llm.generate([p], samp)
            choice_groups = outs[0].outputs  # list of n choices for this prompt

            # Collect token ids (fall back to tokenizing text if ids unavailable)
            per_sample_trigrams: list[set[tuple[int, int, int]]] = []
            union_trigrams: list[tuple[int, int, int]] = []

            for ch in choice_groups:
                # Prefer engine-provided token_ids (fast and accurate)
                toks = getattr(ch, "token_ids", None)
                if toks is None:
                    # Fallback: tokenize generated text
                    text = ch.text or ""
                    toks = tok.encode(text, add_special_tokens=False)
                tri = trigrams(toks)
                per_sample_trigrams.append(set(tri))
                union_trigrams.extend(tri)

            total_tris = len(union_trigrams)
            unique_tris = len(set(union_trigrams))
            distinct3 = (unique_tris / total_tris) if total_tris > 0 else 0.0

            # Average pairwise Jaccard over all pairs in M samples
            M = len(per_sample_trigrams)
            if M <= 1:
                avg_jacc = 0.0
            else:
                s = 0.0
                cnt = 0
                for a in range(M):
                    for b in range(a + 1, M):
                        s += jaccard(per_sample_trigrams[a], per_sample_trigrams[b])
                        cnt += 1
                avg_jacc = (s / cnt) if cnt > 0 else 0.0

            w.writerow(
                {
                    "id": r.get("id", f"val_{i}"),
                    "distinct3": f"{distinct3:.6f}",
                    "avg_pairwise_jaccard": f"{avg_jacc:.6f}",
                    "num_samples": M,
                    "total_trigrams": total_tris,
                    "unique_trigrams": unique_tris,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_new_tokens": max_new_tokens,
                }
            )

    print(f"Wrote metrics to {out_csv}")


def _cli() -> None:
    ap = argparse.ArgumentParser(
        description="Sample M completions per GSM8k validation prompt with vLLM and compute distinct-3 and average Jaccard across trigrams."
    )
    ap.add_argument(
        "--val-path",
        type=Path,
        default=Path("artifacts/gsm8k/val.jsonl"),
        help="Path to GSM8k validation JSONL",
    )
    ap.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B",
        help="HF model id or local checkpoint directory under artifacts/checkpoints/",
    )
    ap.add_argument("--m", type=int, default=5, help="Number of samples per prompt")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--out-csv", type=Path, default=Path("artifacts/mode_collapse_val.csv"))
    ap.add_argument("--limit", type=int, default=None, help="Limit number of prompts")
    args = ap.parse_args()

    main(
        val_path=args.val_path,
        model=args.model,
        m=args.m,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        out_csv=args.out_csv,
        limit=args.limit,
    )


if __name__ == "__main__":
    _cli()
