import argparse
import json
import random
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

from grpo_gsm8k.utils.repro import SEED, stable_hash


def main(
    out_dir: str = "artifacts/gsm8k",
    seed: int = SEED,
    eval_n: int = 512,
    revision: str = "main",
    cache_dir: str = "/workspace/.cache/huggingface",
    make_hf_snapshot: bool = True,
    snapshot_dir: str = "artifacts/gsm8k_hf_snapshot",
) -> None:
    """
    Prepare GSM8K data from Hugging Face datasets.

    Produces JSONL files:
      - train.jsonl (train minus val subset)
      - val.jsonl   (held-out subset sampled from train)
      - test.jsonl  (entire test split)

    Each line: {"question": ..., "answer": ..., "id": ...}
    """
    ds = load_dataset("openai/gsm8k", "main", revision=revision, cache_dir=cache_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Deterministic sampling
    random.seed(seed)
    train = list(ds["train"])
    test = list(ds["test"])

    # Sample eval subset from TRAIN (uniform, seeded), remove from train
    k = min(eval_n, len(train))
    eval_indices = set(random.sample(range(len(train)), k))
    val_rows = [ex for i, ex in enumerate(train) if i in eval_indices]
    train_rows = [ex for i, ex in enumerate(train) if i not in eval_indices]
    test_rows = test

    def dump(name: str, rows: list[dict[str, Any]]) -> None:
        path = Path(out_dir) / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                ex = {
                    "question": r["question"].strip(),
                    "answer": r["answer"].strip(),
                    "id": stable_hash({"q": r["question"]}),
                }
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"Wrote {path} ({len(rows)} rows)")

    dump("train", train_rows)
    dump("val", val_rows)
    dump("test", test_rows)

    if make_hf_snapshot:
        Path(snapshot_dir).mkdir(parents=True, exist_ok=True)
        DatasetDict(
            {
                "train": Dataset.from_list(train_rows),
                "val": Dataset.from_list(val_rows),
                "test": Dataset.from_list(test_rows),
            }
        ).save_to_disk(snapshot_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare GSM8K train/val/test splits.")
    parser.add_argument(
        "--out-dir",
        default="artifacts/gsm8k",
        dest="out_dir",
        help="Output directory for JSONL files (default: artifacts/gsm8k).",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED, help=f"Random seed for sampling (default: {SEED})."
    )
    parser.add_argument(
        "--eval-n",
        type=int,
        default=512,
        dest="eval_n",
        help="Number of eval (val) examples sampled from train (default: 512).",
    )
    parser.add_argument(
        "--revision", default="main", help='Dataset revision/branch/tag/commit (default: "main").'
    )
    parser.add_argument(
        "--cache-dir",
        default="/workspace/.cache/huggingface",
        dest="cache_dir",
        help="HF datasets cache dir (default: /workspace/.cache/huggingface).",
    )
    parser.add_argument(
        "--snapshot-dir",
        default="artifacts/gsm8k_hf_snapshot",
        dest="snapshot_dir",
        help="Directory to save HuggingFace snapshot (default: artifacts/gsm8k_hf_snapshot).",
    )
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(
        "--snapshot",
        dest="snapshot",
        action="store_true",
        help="Save a HF DatasetDict snapshot to disk (default).",
    )
    grp.add_argument(
        "--no-snapshot",
        dest="snapshot",
        action="store_false",
        help="Do not save a HF snapshot to disk.",
    )
    parser.set_defaults(snapshot=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(
        out_dir=args.out_dir,
        seed=args.seed,
        eval_n=args.eval_n,
        revision=args.revision,
        cache_dir=args.cache_dir,
        make_hf_snapshot=args.snapshot,
        snapshot_dir=args.snapshot_dir,
    )
