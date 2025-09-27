from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from grpo_gsm8k.prompts import render_batch
from grpo_gsm8k.tokenize import tokenize_prompt_and_output


def load_jsonl_pairs(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            p = obj.get("prompt", "")
            r = obj.get("response", "")
            rows.append({"prompt": str(p), "response": str(r)})
    return rows


def chunks(xs: list[Any], n: int) -> Iterator[list[Any]]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", type=Path, default=Path("artifacts/r1_sft_pairs.jsonl"))
    ap.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    ap.add_argument("--out_dir", type=Path, default=Path("artifacts/tokenized"))
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--max_total_tokens", type=int, default=2048)  # matches sft.py default
    ap.add_argument("--shard_size", type=int, default=10000)  # examples per shard
    args = ap.parse_args()

    rows = load_jsonl_pairs(args.infile)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token or "<|pad|>"
    tok.padding_side = "left"
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)

    # Save tokenizer snapshot alongside tokenized shards (helps exact reproducibility)
    tok.save_pretrained(args.out_dir / "tokenizer_snapshot")

    prompts = [r["prompt"] for r in rows]
    responses = [r["response"] for r in rows]

    def pad_to(x: torch.Tensor, T: int, pad: int) -> torch.Tensor:
        if x.size(1) >= T:
            return x[:, :T]
        pad_cols = T - x.size(1)
        if x.dtype == torch.long:
            pad_block = torch.full((x.size(0), pad_cols), pad, dtype=torch.long)
        else:
            pad_block = torch.zeros((x.size(0), pad_cols), dtype=x.dtype)
        return torch.cat([x, pad_block], dim=1)

    shard_idx = 0
    written = 0
    for shard_start in range(0, len(prompts), args.shard_size):
        shard_end = min(shard_start + args.shard_size, len(prompts))
        p_shard = prompts[shard_start:shard_end]
        r_shard = responses[shard_start:shard_end]

        in_rows: list[torch.Tensor] = []
        lab_rows: list[torch.Tensor] = []
        mask_rows: list[torch.Tensor] = []

        for p_batch, r_batch in zip(
            chunks(p_shard, args.batch_size), chunks(r_shard, args.batch_size)
        ):
            # Apply Qwen chat template
            chat_prompts = render_batch(tok, p_batch, add_generation_prompt=True)
            # Tokenize prompt/output
            out = tokenize_prompt_and_output(chat_prompts, r_batch, tok)
            inp = pad_to(out["input_ids"], args.max_total_tokens, pad_id)
            lab = pad_to(out["labels"], args.max_total_tokens, pad_id)
            msk = pad_to(out["response_mask"], args.max_total_tokens, 0)
            in_rows.append(inp.cpu())
            lab_rows.append(lab.cpu())
            mask_rows.append(msk.cpu())

        input_ids = (
            torch.cat(in_rows, dim=0)
            if in_rows
            else torch.empty(0, args.max_total_tokens, dtype=torch.long)
        )
        labels = (
            torch.cat(lab_rows, dim=0)
            if lab_rows
            else torch.empty(0, args.max_total_tokens, dtype=torch.long)
        )
        response_mask = (
            torch.cat(mask_rows, dim=0)
            if mask_rows
            else torch.empty(0, args.max_total_tokens, dtype=torch.long)
        )

        lens = (input_ids != pad_id).sum(dim=1)

        shard_obj = {
            "input_ids": input_ids.contiguous(),
            "labels": labels.contiguous(),
            "response_mask": response_mask.contiguous(),
            "len": lens,
            "pad_token_id": int(pad_id),
            "meta": {
                "model_id": args.model_id,
                "max_total_tokens": int(args.max_total_tokens),
                "start_idx": int(shard_start),
                "end_idx": int(shard_end),
                "count": int(shard_end - shard_start),
            },
        }
        shard_path = (
            args.out_dir / f"r1_sft_pairs_max{args.max_total_tokens}_shard_{shard_idx:05d}.pt"
        )
        torch.save(shard_obj, shard_path)
        written += shard_obj["meta"]["count"]
        shard_idx += 1
        print(f"Wrote {shard_path} (n={shard_obj['meta']['count']})")

    print(f"Done. Tokenized {written} examples to {args.out_dir}")


if __name__ == "__main__":
    main()
