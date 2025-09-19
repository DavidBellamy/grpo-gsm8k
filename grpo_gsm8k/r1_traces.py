"""
Generate and save DeepSeek-R1 reasoning traces for GSM8K (train set).

- Reads: artifacts/gsm8k/train.jsonl (fields: question, answer, id)
- Writes: out/deepseek_r1_gsm8k_traces.jsonl (one JSON per line)
- Also writes: out/manifest_<timestamp>.json (run metadata + aggregate token/cost stats)
- Resumable: skips ids already present in the output file
- Concurrency: configurable; polite rate limiting & retries
- Provider: uses official DeepSeek API only (cheapest, with off-peak discounts)
- Reasoning content: saved from `message.reasoning_content` (DeepSeek official)
- Final answer: saved from `message.content`
- Cost estimator: uses live token usage returned by API + current price table

Usage:
    DEEPSEEK_API_KEY=... python generate_r1_gsm8k.py \
        --infile artifacts/gsm8k/train.jsonl \
        --outfile artifacts/deepseek_r1_gsm8k_traces.jsonl \
        --concurrency 4 \
        --max-tokens 2048 \
        --max-retries 5 \
        --offpeak 0.25 \
        --limit 100

Args:
    --infile         Path to input JSONL file (default: artifacts/gsm8k/train.jsonl)
    --outfile        Path to output JSONL file (default: artifacts/deepseek_r1_gsm8k_traces.jsonl)
    --concurrency    Number of concurrent API requests (default: 4)
    --max-tokens     Max tokens per DeepSeek completion (default: 2048)
    --max-retries    Max retries per sample (default: 5)
    --offpeak        Optional discount multiplier for DeepSeek off-peak (e.g., 0.25 for 75% off)
    --limit          Max number of *unique* prompts to send to DeepSeek API (default: 5)
    --samples-per-prompt  Number of R1 responses to collect per prompt (default: 1)

Env vars:
    DEEPSEEK_API_KEY   required

Notes: the total number of samples collected will be limit * samples-per-prompt.
        Although if the output file already contains some samples, the script will
        skip those prompts and only collect the remaining ones, up to the limit.
"""

import argparse
import asyncio
import json
import os
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp

from grpo_gsm8k.reward_fn import (
    extract_answer_colon,
    normalize_number,
    reward_from_text,
)

# --------------------------- Pricing (USD per 1M tokens) ---------------------------
PRICES = {
    "deepseek": {
        # Official docs (2025-01-20 release; still current as of 2025-09-19):
        # Input: $0.55/M (cache miss), $0.14/M (cache hit); Output: $2.19/M
        # We conservatively assume cache miss for inputs.
        "input_per_m": 0.55,
        "output_per_m": 2.19,
        # Optional off-peak discount multiplier (0.25 means 75% off). Set via --offpeak 0.25
    },
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_seen_counts(outfile: Path) -> dict[str, int]:
    """
    Return how many samples have already been collected per id, to support resume.
    """
    seen: dict[str, int] = {}
    if outfile.exists():
        with outfile.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    _id = obj.get("id")
                    if _id is None:
                        continue
                    seen[_id] = seen.get(_id, 0) + 1
                except Exception:
                    continue
    return seen


async def backoff_sleep(attempt: int) -> None:
    # Exponential backoff with jitter
    base = min(60, (2**attempt))
    await asyncio.sleep(base * (0.5 + random.random() * 0.75))


async def call_deepseek(
    session: aiohttp.ClientSession, api_key: str, prompt: str, max_tokens: int
) -> dict[str, Any]:
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "deepseek-reasoner",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    async with session.post(url, json=payload, headers=headers, timeout=120) as resp:
        if resp.status != 200:
            txt = await resp.text()
            raise RuntimeError(f"DeepSeek API error {resp.status}: {txt[:500]}")
        data = await resp.json()
        return data


def summarize_usage(usage: dict[str, Any] | None, offpeak: float | None) -> dict[str, Any]:
    in_tok = usage.get("prompt_tokens", 0) if usage else 0
    out_tok = usage.get("completion_tokens", 0) if usage else 0

    price_in = PRICES["deepseek"]["input_per_m"]
    price_out = PRICES["deepseek"]["output_per_m"]

    if offpeak is not None:
        price_in *= offpeak
        price_out *= offpeak

    cost = (in_tok / 1_000_000.0) * price_in + (out_tok / 1_000_000.0) * price_out
    return {
        "prompt_tokens": in_tok,
        "completion_tokens": out_tok,
        "total_tokens": in_tok + out_tok,
        "estimated_cost_usd": round(cost, 6),
        "unit_prices_per_1M": {"input": price_in, "output": price_out},
    }


async def worker(
    name: str,
    queue: asyncio.Queue,
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    seen_counts: dict[str, int],
    outfile: Path,
    manifest: dict[str, Any],
    stats: dict[str, Any],
) -> None:
    api_key = args.api_key

    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break
        entry_id, q, gold = item

        # How many samples already collected for this id?
        already = seen_counts.get(entry_id, 0)
        if already >= args.samples_per_prompt:
            queue.task_done()
            continue

        # Pre-parse gold once (GSM8K '#### <number>' tail)
        gold_tail = gold.split("####")[-1] if isinstance(gold, str) and "####" in gold else gold
        gold_number = normalize_number(gold_tail)

        # Collect remaining samples for this prompt
        for sample_idx in range(already, args.samples_per_prompt):
            tries = 0
            while True:
                tries += 1
                t0 = time.time()
                try:
                    # Build a simple prompt. R1 auto-thinks; we request final numeric as
                    # ANSWER: <number>
                    prompt = (
                        "Solve the math word problem. Show your reasoning.\n"
                        "At the very end, output the final numeric answer alone on its own line "
                        "as: ANSWER: <number>\n\n"
                        f"Problem: {q}\n"
                    )

                    data = await call_deepseek(session, api_key, prompt, args.max_tokens)
                    choice = data["choices"][0]["message"]
                    reasoning = choice.get("reasoning_content", "")
                    final = choice.get("content", "")
                    usage = data.get("usage", {})

                    usage_summary = summarize_usage(usage, args.offpeak)

                    # Parse R1 numeric answer from "ANSWER: <number>"
                    pred_src = extract_answer_colon(final)
                    final_number = normalize_number(pred_src)

                    # 1/0 correctness using the 'answer' parser (R1 format)
                    correct = int(reward_from_text(final, gold, parser="answer"))

                    rec = {
                        "id": entry_id,
                        "sample_index": sample_idx,
                        "question": q,
                        "gold_answer": gold,
                        "gold_number": gold_number,
                        "model": "deepseek-reasoner",
                        "provider": "deepseek",
                        "reasoning": reasoning,
                        "final": final,
                        "final_number": final_number,
                        "correct": correct,
                        "usage": usage_summary,
                        "raw_usage": usage,
                        "latency_sec": round(time.time() - t0, 3),
                        "ts_utc": datetime.now(timezone.utc).isoformat(),
                        "run_id": manifest["run_id"],
                        "worker": name,
                    }

                    with outfile.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                    # Update resume state and global stats
                    seen_counts[entry_id] = sample_idx + 1
                    stats["n_done"] += 1  # count responses
                    stats["prompt_tokens"] += usage_summary["prompt_tokens"]
                    stats["completion_tokens"] += usage_summary["completion_tokens"]
                    stats["cost_usd"] += usage_summary["estimated_cost_usd"]
                    break  # success for this sample

                except Exception as e:
                    if tries <= args.max_retries:
                        await backoff_sleep(tries)
                        continue
                    # Log a failure record (keeps place; doesn't advance seen_counts)
                    rec = {
                        "id": entry_id,
                        "sample_index": sample_idx,
                        "error": str(e)[:500],
                        "ts_utc": datetime.now(timezone.utc).isoformat(),
                        "run_id": manifest["run_id"],
                        "worker": name,
                    }
                    with outfile.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    break  # give up on this sample

        queue.task_done()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=str, default="artifacts/gsm8k/train.jsonl")
    parser.add_argument("--outfile", type=str, default="artifacts/deepseek_r1_gsm8k_traces.jsonl")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=2048, dest="max_tokens")
    parser.add_argument("--max-retries", type=int, default=5, dest="max_retries")
    parser.add_argument(
        "--samples-per-prompt",
        type=int,
        default=1,
        dest="samples_per_prompt",
        help="Number of R1 responses to collect per prompt (default: 1)",
    )
    parser.add_argument(
        "--offpeak",
        type=float,
        default=None,
        help="Optional discount multiplier for DeepSeek off-peak (e.g., 0.25 for 75% off)",
    )
    parser.add_argument(
        "--limit", type=int, default=5, help="Max number of prompts to send to DeepSeek API"
    )
    args = parser.parse_args()

    # Credentials
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("Missing DEEPSEEK_API_KEY")
    args.api_key = api_key

    infile = Path(args.infile)
    outfile = Path(args.outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    # Prepare manifest
    run_id = str(uuid.uuid4())
    manifest = {
        "run_id": run_id,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "infile": str(infile),
        "outfile": str(outfile),
        "provider": "deepseek",
        "model": "deepseek-reasoner",
        "max_tokens": args.max_tokens,
        "concurrency": args.concurrency,
        "offpeak_multiplier": args.offpeak,
        "notes": "GSM8K train reasoning traces with DeepSeek-R1",
        "pricing": PRICES["deepseek"],
    }

    # Load existing to resume (count samples per prompt)
    seen = load_seen_counts(outfile)

    # Build task queue
    queue: asyncio.Queue = asyncio.Queue()
    total = 0
    added = 0
    with infile.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            q = obj.get("question")
            a = obj.get("answer")
            _id = obj.get("id")
            if _id is None:
                continue
            total += 1
            # Skip prompts already fully collected
            if seen.get(_id, 0) >= args.samples_per_prompt:
                continue
            if args.limit is not None and added >= args.limit:
                break
            await queue.put((_id, q, a))
            added += 1

    # Stats
    already_done_samples = sum(seen.values())
    total_expected_samples = total * args.samples_per_prompt
    stats = {
        "n_total_in_file": total,  # prompts
        "n_already_done": already_done_samples,  # samples
        "n_remaining": max(0, total_expected_samples - already_done_samples),  # samples
        "n_done": 0,  # samples completed this run
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost_usd": 0.0,
        "limit": args.limit,
        "samples_per_prompt": args.samples_per_prompt,
    }

    # Write manifest now
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    manifest_path = outfile.parent / f"manifest_{ts}.json"
    with manifest_path.open("w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)

    connector = aiohttp.TCPConnector(limit_per_host=args.concurrency)
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=60, sock_read=120)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Spawn workers
        workers = [
            asyncio.create_task(
                worker(f"w{i+1}", queue, session, args, seen, outfile, manifest, stats)
            )
            for i in range(args.concurrency)
        ]

        # Progress printer
        async def progress() -> None:
            last = time.time()
            while any(not w.done() for w in workers):
                await asyncio.sleep(5)
                elapsed = time.time() - last
                last = time.time()
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"samples_done={stats['n_done']} rem_prompts={queue.qsize()} "
                    f"cost=${stats['cost_usd']:.4f} tok_in={stats['prompt_tokens']} "
                    f"tok_out={stats['completion_tokens']} "
                    f"elapsed={elapsed:.1f}s"
                )

        prog_task = asyncio.create_task(progress())

        # Drain queue
        await queue.join()
        for _ in workers:
            await queue.put(None)
        await asyncio.gather(*workers, return_exceptions=True)
        prog_task.cancel()

    # Finalize manifest with stats
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["aggregate"] = {
        "estimated_cost_usd": round(stats["cost_usd"], 6),
        "prompt_tokens": stats["prompt_tokens"],
        "completion_tokens": stats["completion_tokens"],
        "n_completed_this_run_samples": stats["n_done"],
        "n_total_prompts_in_file": stats["n_total_in_file"],
        "samples_per_prompt": args.samples_per_prompt,
    }
    with manifest_path.open("w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)

    print("Done.")
    print(f"Wrote traces to: {outfile}")
    print(f"Manifest: {manifest_path}")
    # Quick per-sample average
    if stats["n_done"] > 0:
        avg_out = stats["completion_tokens"] / max(1, stats["n_done"])
        print(f"Average completion tokens per solved sample: {avg_out:.1f}")


if __name__ == "__main__":
    asyncio.run(main())
