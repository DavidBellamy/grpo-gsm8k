import json
from pathlib import Path
from typing import Any

import grpo_gsm8k.data_prep as dp


class FakeDS(dict):
    pass


def fake_load_dataset(_name: str, _config: str, **_: Any) -> FakeDS:
    # Tiny deterministic dataset
    train = [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(10)]
    test = [{"question": f"TQ{i}", "answer": f"TA{i}"} for i in range(4)]
    return FakeDS(train=train, test=test)


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]


def test_cli_split_and_sizes(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(dp, "load_dataset", fake_load_dataset)
    out_dir = tmp_path / "gsm8k"
    args = [
        "--out-dir",
        str(out_dir),
        "--eval-n",
        "3",
        "--seed",
        "123",
        "--no-snapshot",
    ]
    # Run main via its CLI parse
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    args_ns = dp.parse_args(args)
    dp.main(
        out_dir=args_ns.out_dir,
        seed=args_ns.seed,
        eval_n=args_ns.eval_n,
        revision=args_ns.revision,
        cache_dir=args_ns.cache_dir,
        make_hf_snapshot=args_ns.snapshot,
        snapshot_dir=args_ns.snapshot_dir,
    )

    train = read_jsonl(out_dir / "train.jsonl")
    val = read_jsonl(out_dir / "val.jsonl")
    test = read_jsonl(out_dir / "test.jsonl")

    assert len(val) == 3
    assert len(train) == 10 - 3
    assert len(test) == 4

    # Ensure train and val questions are disjoint
    train_q = {r["question"] for r in train}
    val_q = {r["question"] for r in val}
    assert train_q.isdisjoint(val_q)
