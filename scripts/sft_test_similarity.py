from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import tqdm

PUNCT_RE = re.compile(r"[^\w\s#\.\-/:]")
MULTISPACE_RE = re.compile(r"\s+")


def normalize(s: str) -> str:
    if s is None:
        return ""
    s = s.lower().replace("\u2212", "-")
    s = PUNCT_RE.sub(" ", s)
    s = MULTISPACE_RE.sub(" ", s).strip()
    return s


def char_ngrams(s: str, n: int) -> set[str]:
    s = (s or "").replace(" ", "_")
    if len(s) < n:
        return set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def word_ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def jaccard(a: set[Any], b: set[Any]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


@dataclass
class TestItem:
    id: str
    question_raw: str
    answer_raw: str
    answer_norm: str
    answer_char_ngrams: set[str]
    answer_word_ngrams: set[tuple[str, ...]]


def read_jsonl(path: Path) -> list[dict]:
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def load_test_items(test_path: Path, char_n: int, word_n: int) -> list[TestItem]:
    raw = read_jsonl(test_path)
    out: list[TestItem] = []
    for o in raw:
        q = o.get("question", "")
        a = o.get("answer", "")
        _id = str(o.get("id", ""))
        a_norm = normalize(a)
        aw = a_norm.split()
        out.append(
            TestItem(
                id=_id,
                question_raw=q,
                answer_raw=a,
                answer_norm=a_norm,
                answer_char_ngrams=char_ngrams(a_norm, char_n),
                answer_word_ngrams=word_ngrams(aw, word_n),
            )
        )
    return out


def process_one_sft(
    si: int,
    sft_item: dict,
    test_items: list[TestItem],
    char_n: int,
    word_n: int,
) -> tuple[list[tuple[float, dict]], int]:
    prompt = sft_item.get("prompt", "") or ""
    response = sft_item.get("response", "") or ""

    resp_norm = normalize(response)

    resp_char = char_ngrams(resp_norm, char_n)
    resp_word = word_ngrams(resp_norm.split(), word_n)

    rows = []
    total_pairs = 0

    for t in test_items:
        total_pairs += 1
        char_j = jaccard(resp_char, t.answer_char_ngrams)
        word_j = jaccard(resp_word, t.answer_word_ngrams)
        containment = int(t.answer_norm and (t.answer_norm in resp_norm))

        score = (2.0 * containment) + max(char_j, word_j)

        row = dict(
            sft_index=si,
            sft_prompt=prompt,
            sft_response=response,
            test_id=t.id,
            test_question=t.question_raw,
            test_answer=t.answer_raw,
            char_n=char_n,
            word_n=word_n,
            char_jaccard=f"{char_j:.6f}",
            word_jaccard=f"{word_j:.6f}",
            full_answer_containment=bool(containment),
            sft_response_len=len(resp_norm),
            test_answer_len=len(t.answer_norm),
        )
        rows.append((score, row))

    return rows, total_pairs


def main() -> None:
    ap = argparse.ArgumentParser(
        description="All-pairs SFT-response vs Test-answer similarity (Jaccard + containment)."
    )
    ap.add_argument("--sft", type=Path, required=True, help="Path to r1_sft_pairs.jsonl")
    ap.add_argument("--test", type=Path, required=True, help="Path to gsm8k test.jsonl")
    ap.add_argument(
        "--full-out", type=Path, required=True, help="Path to write ALL pairs as gzipped CSV"
    )
    ap.add_argument("--top-out", type=Path, required=True, help="Path to write TOP-K CSV")
    ap.add_argument("--top-k", type=int, default=100, help="Top K rows to keep")
    ap.add_argument("--char-n", type=int, default=13, help="Char n-gram size")
    ap.add_argument("--word-n", type=int, default=5, help="Word n-gram size")
    ap.add_argument(
        "--workers", type=int, default=0, help="Number of worker processes (default: CPU count)"
    )
    args = ap.parse_args()

    sft_items = read_jsonl(args.sft)
    test_items = load_test_items(args.test, char_n=args.char_n, word_n=args.word_n)

    full_fields = [
        "sft_index",
        "test_id",
        "char_jaccard",
        "word_jaccard",
        "full_answer_containment",
        "sft_response_len",
        "test_answer_len",
        "char_n",
        "word_n",
        "sft_prompt",
        "sft_response",
        "test_question",
        "test_answer",
    ]

    args.full_out.parent.mkdir(parents=True, exist_ok=True)
    args.top_out.parent.mkdir(parents=True, exist_ok=True)
    gz_f = gzip.open(args.full_out, "wt", encoding="utf-8", newline="")
    full_writer = csv.DictWriter(gz_f, fieldnames=full_fields)
    full_writer.writeheader()

    import heapq
    from itertools import count

    top_heap: list[tuple[float, int, dict]] = []
    heapq.heapify(top_heap)
    tiebreak = count()  # unique increasing integers

    tasks = [(i, sft_items[i], test_items, args.char_n, args.word_n) for i in range(len(sft_items))]

    def consume_results(iterable: Iterator[tuple[list[tuple[float, dict]], int]]) -> int:
        total_pairs = 0
        iterator = tqdm(iterable, total=len(tasks), desc="[compare] SFT rows")
        for rows, cnt in iterator:
            total_pairs += cnt
            for score, row in rows:
                full_writer.writerow(row)
                if len(top_heap) < args.top_k:
                    heapq.heappush(top_heap, (score, next(tiebreak), row))
                else:
                    if score > top_heap[0][0]:
                        heapq.heapreplace(top_heap, (score, next(tiebreak), row))
        return total_pairs

    if args.workers and args.workers > 0:
        workers = args.workers or cpu_count()
        from functools import partial

        proc = partial(
            process_one_sft, test_items=test_items, char_n=args.char_n, word_n=args.word_n
        )

        with Pool(processes=workers) as pool:
            total_pairs = consume_results(
                iter(pool.starmap(proc, ((si, sft_items[si]) for si in range(len(sft_items)))))
            )

    else:

        def gen() -> Iterator[tuple[list[tuple[float, dict]], int]]:
            for si, sft_item in enumerate(sft_items):
                yield process_one_sft(si, sft_item, test_items, args.char_n, args.word_n)

        total_pairs = consume_results(gen())

    gz_f.close()

    top = sorted(top_heap, key=lambda x: x[0], reverse=True)[: args.top_k]
    with args.top_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["score"] + full_fields)
        writer.writeheader()
        for score, _tb, row in top:
            writer.writerow({"score": f"{score:.6f}", **row})

    print(f"[done] Total pairs written: {total_pairs}")
    print(f"[done] ALL pairs: {args.full_out}")
    print(f"[done] TOP-{args.top_k}: {args.top_out}")


if __name__ == "__main__":
    main()
