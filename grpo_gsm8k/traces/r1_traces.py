"""
Generate and save DeepSeek-R1 reasoning traces for GSM8K (train set).

- Reads: artifacts/gsm8k/train.jsonl (fields: question, answer, id)
- Writes: out/deepseek_r1_gsm8k_traces.jsonl (one JSON per line)
- Also writes: out/manifest_<timestamp>.json (run metadata + aggregate token/cost stats)
- Resumable: skips ids already present in the output file
- Concurrency: configurable; polite rate limiting & retries
- Provider: uses official DeepSeek API only
- Reasoning content: saved from `message.reasoning_content` (DeepSeek official)
- Final answer: saved from `message.content`

Usage:
    DEEPSEEK_API_KEY=... python generate_r1_gsm8k.py \
        --infile artifacts/gsm8k/train.jsonl \
        --outfile artifacts/deepseek_r1_gsm8k_traces.jsonl \
        --concurrency 52 \
        --max-tokens 2048 \
        --max-retries 5 \
        --limit 100

Args:
    --infile         Path to input JSONL file (default: artifacts/gsm8k/train.jsonl)
    --outfile        Path to output JSONL file (default: artifacts/deepseek_r1_gsm8k_traces.jsonl)
    --concurrency    Number of concurrent API requests (default: 52)
    --max-tokens     Max tokens per DeepSeek completion (default: 2048)
    --max-retries    Max retries per sample (default: 5)
    --limit          Max number of *unique* prompts to send to DeepSeek API (default: None)
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

from grpo_gsm8k.evaluation.reward_fn import (
    extract_answer_colon,
    normalize_number,
    reward_from_text,
)


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


def summarize_usage(usage: dict[str, Any] | None) -> dict[str, int]:
    in_tok = usage.get("prompt_tokens", 0) if usage else 0
    out_tok = usage.get("completion_tokens", 0) if usage else 0
    return {
        "prompt_tokens": int(in_tok),
        "completion_tokens": int(out_tok),
        "total_tokens": int(in_tok + out_tok),
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
        entry_id, q, gold, sample_idx = item

        # Pre-parse gold once (GSM8K '#### <number>' tail)
        gold_tail = gold.split("####")[-1] if isinstance(gold, str) and "####" in gold else gold
        gold_number = normalize_number(gold_tail)

        tries = 0
        while True:
            tries += 1
            t0 = time.time()
            try:
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

                usage_summary = summarize_usage(usage)
                pred_src = extract_answer_colon(final)
                final_number = normalize_number(pred_src)
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

                # Update resume state and stats
                seen_counts[entry_id] = max(seen_counts.get(entry_id, 0), sample_idx + 1)
                stats["n_done"] += 1
                stats["prompt_tokens"] += usage_summary["prompt_tokens"]
                stats["completion_tokens"] += usage_summary["completion_tokens"]
                break
            except Exception as e:
                if tries <= args.max_retries:
                    await backoff_sleep(tries)
                    continue
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
                break
        queue.task_done()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=str, default="artifacts/gsm8k/train.jsonl")
    parser.add_argument("--outfile", type=str, default="artifacts/deepseek_r1_gsm8k_traces.jsonl")
    parser.add_argument("--concurrency", type=int, default=52)
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
        "--limit", type=int, default=None, help="Max number of prompts to send to DeepSeek API"
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
        "notes": "GSM8K train reasoning traces with DeepSeek-R1",
    }

    # Load existing to resume (count samples per prompt)
    seen = load_seen_counts(outfile)

    # Build task queue
    queue: asyncio.Queue = asyncio.Queue()
    total = 0
    added = 0
    tasks_enqueued = 0
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
            already = seen.get(_id, 0)
            for sample_idx in range(already, args.samples_per_prompt):
                await queue.put((_id, q, a, sample_idx))
                tasks_enqueued += 1
            added += 1

    # Stats
    already_done_samples = sum(seen.values())
    total_expected_samples = tasks_enqueued + already_done_samples
    stats = {
        "n_total_in_file": total,  # prompts
        "n_already_done": already_done_samples,  # samples
        "n_remaining": max(0, total_expected_samples - already_done_samples),  # samples
        "n_done": 0,  # samples completed this run
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "limit": args.limit,
        "samples_per_prompt": args.samples_per_prompt,
        "prompts_enqueued_this_run": added,
        "tasks_enqueued_this_run": tasks_enqueued,
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
                    f"samples_done={stats['n_done']} rem_tasks={queue.qsize()} "
                    f"tok_in={stats['prompt_tokens']} "
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
        "prompt_tokens": stats["prompt_tokens"],
        "completion_tokens": stats["completion_tokens"],
        "n_completed_this_run_samples": stats["n_done"],
        "n_total_prompts_in_file": stats["n_total_in_file"],
        "samples_per_prompt": args.samples_per_prompt,
        "n_prompts_enqueued_this_run": stats["prompts_enqueued_this_run"],
        "n_tasks_enqueued_this_run": stats["tasks_enqueued_this_run"],
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
