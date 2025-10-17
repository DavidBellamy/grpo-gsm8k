from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Patterns to drop R1's internal meta / instruction echoes
META = re.compile(
    r"("
    r"I (?:should|will) output.*(?:ANSWER|\\boxed)\s*:?.*|"
    r"At the very end.*(?:ANSWER|\\boxed).*|"
    r"As requested.*(?:ANSWER|\\boxed).*|"
    r"I need to output.*(?:ANSWER|\\boxed).*|"
    r"I will now provide.*final answer.*|"
    r"The final answer (?:is|will be).*|"
    r"I'll present the final.*|"
    r"Please note.*final.*answer.*|"
    r"Finally,?\s*output the answer.*|"
    r"Now,?\s*for the output,.*|"
    r"Finally,.*(?:write the answer|answer .*own\s*line|output\s*:).*|"
    r"Now,?\s*the output should be.*|"
    r"The problem says .*output the final numeric answer alone on its own line.*|"
    r"The question .*output the final numeric answer alone on its own line.*|"
    r"(?:instruction|instructions).*show your reasoning.*|"
    r"(?:It|it)\s*says.*show your reasoning.*|"
    r"show your reasoning.*|"
    r"The problem says .*show your reasoning.*|"
    r"So,\s*in my response,\s*I'll write my reasoning.*|"
    r"So,\s*in the reasoning,\s*I'll write the steps.*|"
    r"(?:So,\s*)?I'll write my reasoning step by step.*|"
    r"(?:But\s+|And\s+)?in the output format.*|"
    r"I should make sure about the format.*|"
    r"It says .*output the final numeric answer alone on its own line.*|"
    r"output .*final numeric answer alone on its own line.*|"
    r"Now,?\s*for the answer,.*(?:output|say).*(?:numeric value|number).*|"
    r"Final answer.*ANSWER\s*:.*|"
    r"So.*ANSWER\s*:.*"
    r")",
    re.I,
)
ANSWER_LINE = re.compile(r"^\s*ANSWER:\s*[-+]?\d+(?:\.\d+)?\s*$", re.I | re.M)
BOXED_ANY = re.compile(r"\\boxed\{[^}]+\}")
NUM_ANY = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


@dataclass
class Stats:
    processed: int = 0
    written: int = 0
    skipped: int = 0


def collapse_blank_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    for s in lines:
        if not s:
            if out and out[-1] == "":
                continue
        out.append(s)
    # Trim leading/trailing blanks
    while out and out[0] == "":
        out.pop(0)
    while out and out[-1] == "":
        out.pop()
    return out


def clean_reasoning(text: str) -> str:
    lines: list[str] = []
    for ln in text.splitlines():
        s = ln.rstrip()
        if not s:
            lines.append("")
            continue
        if META.search(s):
            continue
        if ANSWER_LINE.match(s):
            continue
        # Remove inline boxed markers in the reasoning body; we append a final boxed later
        if BOXED_ANY.search(s):
            s = BOXED_ANY.sub("", s).rstrip()
            if not s:
                continue
        lines.append(s)
    lines = collapse_blank_lines(lines)
    return "\n".join(lines)


def normalize_number(num: str | None, fallback_text: str | None = None) -> str:
    if num is not None and str(num).strip():
        s = str(num).replace(",", "").strip()
        m = NUM_ANY.search(s)
        if m:
            return m.group(0)
        return s
    # Fallback: try to parse from text
    if fallback_text:
        m = NUM_ANY.search(fallback_text)
        if m:
            return m.group(0)
    return ""


def build_response(
    reasoning: str, final_number: str | None, fallback_text: str | None = None
) -> str:
    body = clean_reasoning(reasoning or "")
    num = normalize_number(final_number, fallback_text=fallback_text)
    if not num:
        # As a last resort, end without a box (rare); keeps pipeline robust
        return body
    return f"{body}\n\n\\boxed{{{num}}}"


def format_r1_traces(
    infile: Path, outfile: Path, *, limit: int | None = None, encoding: str = "utf-8"
) -> Stats:
    stats = Stats()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with infile.open("r", encoding=encoding) as r, outfile.open("w", encoding=encoding) as w:
        for i, line in enumerate(r):
            if limit is not None and stats.written >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                stats.skipped += 1
                continue
            stats.processed += 1

            question = obj.get("question")
            reasoning = obj.get("reasoning")
            final_number = obj.get("final_number")
            final_text = obj.get("final") or obj.get("reasoning")
            if not question or not reasoning:
                stats.skipped += 1
                continue

            prompt = str(question).strip()
            response = build_response(str(reasoning), final_number, fallback_text=final_text)
            rec = {"prompt": prompt, "response": response}
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats.written += 1

    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Format DeepSeek R1 GSM8K traces into prompt/response pairs.")
    parser.add_argument(
        "--infile", type=Path, default=Path("artifacts/deepseek_r1_gsm8k_traces.jsonl")
    )
    parser.add_argument("--outfile", type=Path, default=Path("artifacts/r1_sft_pairs.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args(argv)


def main(infile: str, outfile: str, limit: int) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    stats = format_r1_traces(Path(infile), Path(outfile), limit=limit)
    logger.info(
        "formatted R1 traces: processed=%d written=%d skipped=%d",
        stats.processed,
        stats.written,
        stats.skipped,
    )
    logger.info("wrote %s", outfile)


if __name__ == "__main__":
    args = parse_args()
    main(args.infile, args.outfile, args.limit)
