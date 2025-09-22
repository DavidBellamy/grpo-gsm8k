import json
import os
from pathlib import Path
from typing import Any

import pytest

from grpo_gsm8k import r1_traces as mod

pytestmark = pytest.mark.asyncio


class DummyResp:
    def __init__(
        self, final: str, reasoning: str = "think", in_tok: int = 100, out_tok: int = 200
    ) -> None:
        self.data = {
            "choices": [{"message": {"content": final, "reasoning_content": reasoning}}],
            "usage": {"prompt_tokens": in_tok, "completion_tokens": out_tok},
        }


async def fake_call_deepseek(
    _session: Any, _key: str, _prompt: str, _max_tokens: int
) -> dict[str, Any]:
    # Return a fixed correct answer 42
    return DummyResp(final="... steps ...\nANSWER: 42\n").data


def write_jsonl(p: Path, rows: list[dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


async def run_main(
    tmp_path: Path,
    n_prompts: int = 3,
    samples: int = 2,
    limit: int | None = None,
    pre_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    infile = tmp_path / "in.jsonl"
    outfile = tmp_path / "out.jsonl"
    # Build small GSM8K-like input; gold ends with #### 42
    rows = [
        {"id": f"id{i}", "question": f"Q{i}", "answer": "work...\n#### 42"}
        for i in range(n_prompts)
    ]
    write_jsonl(infile, rows)

    if pre_rows:
        write_jsonl(outfile, pre_rows)

    # Patch network and env
    os.environ["DEEPSEEK_API_KEY"] = "test"
    old = mod.call_deepseek
    mod.call_deepseek = fake_call_deepseek  # type: ignore[assignment]
    try:
        # Build argv
        argv = [
            "prog",
            "--infile",
            str(infile),
            "--outfile",
            str(outfile),
            "--concurrency",
            "8",
            "--samples-per-prompt",
            str(samples),
        ]
        if limit is not None:
            argv += ["--limit", str(limit)]
        # Run
        import sys

        old_argv = sys.argv
        sys.argv = argv
        try:
            await mod.main()
        finally:
            sys.argv = old_argv
    finally:
        mod.call_deepseek = old  # restore

    # Read results
    out = [json.loads(line) for line in outfile.read_text(encoding="utf-8").splitlines()]
    return out


async def test_limit_applies_to_unique_prompts(tmp_path: Path) -> None:
    out = await run_main(tmp_path, n_prompts=5, samples=3, limit=2)
    # Expect 2 prompts * 3 samples each
    assert len(out) == 6
    # Check required fields and correctness
    for r in out:
        assert "id" in r and "sample_index" in r
        assert r["final_number"] == "42"
        assert r["correct"] == 1
        assert set(r["usage"]).issuperset({"prompt_tokens", "completion_tokens", "total_tokens"})


async def test_resume_adds_remaining_samples(tmp_path: Path) -> None:
    # Prepopulate 2 samples for id0
    pre = [{"id": "id0", "sample_index": 0}, {"id": "id0", "sample_index": 1}]

    out = await run_main(tmp_path, n_prompts=2, samples=3, limit=None, pre_rows=pre)
    # Should include 1 new row for id0 (sample_index 2)
    # and 3 for id1 = total 4 real rows + 2 pre = 6 lines
    # We only assert the newly added rows count and indices:
    added = [r for r in out if "final" in r]  # real new rows have full fields
    assert len(added) == 4
    id0_new = [r for r in added if r["id"] == "id0"]
    assert {r["sample_index"] for r in id0_new} == {2}


async def test_summarize_usage_has_no_cost() -> None:
    # Directly test helper
    s = mod.summarize_usage({"prompt_tokens": 10, "completion_tokens": 5})
    assert s == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
