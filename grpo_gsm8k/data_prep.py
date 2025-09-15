import json
import random
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset

from grpo_gsm8k.repro import SEED, stable_hash


def main(
    out_dir: str = "artifacts/gsm8k",
    seed: int = SEED,
    eval_n: int = 800,
    revision: str = "main",
    cache_dir: str = "/workspace/.cache/huggingface",
    make_hf_snapshot: bool = True,
    snapshot_dir: str = "artifacts/gsm8k_hf_snapshot",
) -> None:
    """Prepare GSM8K data from Hugging Face datasets.
    Produces train.jsonl, test_full.jsonl, and test_eval.jsonl (smaller eval subset).
    Each line is {"question":..., "answer":..., "id":...}.
    Args:
        out_dir: output directory
        seed: random seed for shuffling
        eval_n: number of eval examples to select from test set
        revision: git revision of the dataset to use (branch, tag, or commit hash)
        cache_dir: where to cache the dataset
    """
    ds = load_dataset("openai/gsm8k", "main", revision=revision, cache_dir=cache_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Freeze order for reproducibility
    random.seed(seed)
    train = list(ds["train"])
    test = list(ds["test"])

    # Small eval subset from test for fast iteration
    random.shuffle(test)
    eval_subset = test[:eval_n]

    def dump(name: str, rows: list[dict]) -> None:
        path = Path(out_dir) / f"{name}.jsonl"
        with path.open("w") as f:
            for r in rows:
                ex = {
                    "question": r["question"].strip(),
                    "answer": r["answer"].strip(),
                    "id": stable_hash({"q": r["question"]}),
                }
                f.write(json.dumps(ex) + "\n")
        print("Wrote", path)

    dump("train", train)
    dump("test_full", test)
    dump("test_eval", eval_subset)

    if make_hf_snapshot:
        Path(snapshot_dir).mkdir(parents=True, exist_ok=True)
        DatasetDict(
            {
                "train": Dataset.from_list(train),
                "test_full": Dataset.from_list(test),
                "test_eval": Dataset.from_list(eval_subset),
            }
        ).save_to_disk(snapshot_dir)


if __name__ == "__main__":
    main()
