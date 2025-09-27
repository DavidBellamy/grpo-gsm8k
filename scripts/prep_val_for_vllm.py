from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from grpo_gsm8k.prompts import render_batch
from grpo_gsm8k.reward_fn import normalize_number


def load_val_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = obj.get("question", "")
            a = obj.get("answer", "")
            _id = obj.get("id", "")
            rows.append({"id": str(_id), "question": str(q), "answer": str(a)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare val set prompts for vLLM eval.")
    ap.add_argument("--infile", type=Path, default=Path("artifacts/gsm8k/val.jsonl"))
    ap.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    ap.add_argument("--outfile", type=Path, default=Path("artifacts/tokenized/val_for_vllm.jsonl"))
    args = ap.parse_args()

    rows = load_val_rows(args.infile)
    args.outfile.parent.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token or "<|pad|>"
    tok.padding_side = "left"

    questions = [r["question"] for r in rows]
    prompts = render_batch(tok, questions, add_generation_prompt=True)

    with args.outfile.open("w", encoding="utf-8") as w:
        for r, p in zip(rows, prompts):
            gold = r["answer"]
            gold_tail = gold.split("####")[-1] if isinstance(gold, str) and "####" in gold else gold
            gold_num = normalize_number(gold_tail)
            rec = {
                "id": r["id"],
                "prompt": p,  # chat-templated prompt for vLLM
                "question": r["question"],
                "gold": gold,  # original gold solution (for W&B side-by-side)
                "gold_num": gold_num,  # normalized numeric gold for exact-match
            }
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {args.outfile} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
