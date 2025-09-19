import json
import random
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

from grpo_gsm8k.repro import SEED, stable_hash


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


if __name__ == "__main__":
    main()
