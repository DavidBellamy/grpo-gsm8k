import collections
import glob
import math
import os
from collections import defaultdict
from collections.abc import Iterator
from typing import Any

import torch
from torch.utils.data import IterableDataset


class BucketMicrobatcher:
    """Accumulate samples into buckets by length, and emit microbatches of
    similar-length samples.
    """

    def __init__(
        self,
        microbatch_size: int,
        max_seq_len: int,
        num_buckets: int = 12,
        periodic_flush_every: int = 512,  # prevents starvation
    ):
        self.B = microbatch_size
        self.Lmax = max_seq_len
        self.flush_every = periodic_flush_every

        # log-spaced bucket edges from ~8 → max_seq_len
        edges = [
            int(round(math.exp(math.log(max_seq_len) * k / (num_buckets - 1))))
            for k in range(num_buckets)
        ]
        self.edges = [max(8, e) for e in edges]
        self.buckets: dict[int, collections.deque[dict]] = defaultdict(collections.deque)
        self._seen = 0

    def _bid(self, L: int) -> int:
        for i, e in enumerate(self.edges):
            if L <= e:
                return i
        return len(self.edges) - 1

    def _truncate(self, s: dict) -> dict:
        if s["len"] > self.Lmax:
            cut = self.Lmax
            s["input_ids"] = s["input_ids"][:cut]
            s["labels"] = s["labels"][:cut]
            s["len"] = cut
        return s

    def add(self, sample: dict) -> list[dict] | None:
        s = self._truncate(sample)
        self.buckets[self._bid(s["len"])].append(s)
        self._seen += 1

        # Emit a “perfect” batch if any bucket reached B
        for b in reversed(range(len(self.edges))):  # prefer tighter (longer) buckets first
            if len(self.buckets[b]) >= self.B:
                return [self.buckets[b].popleft() for _ in range(self.B)]

        # Periodic flush: build a mixed batch from nearest non-empty bins
        if self._seen % self.flush_every == 0:
            return self.flush_mixed()

        return None

    def flush_mixed(self) -> list[dict] | None:
        # Fill from longest down so padding stays low
        batch: list[dict[str, Any]] = []
        for b in reversed(range(len(self.edges))):
            while self.buckets[b] and len(batch) < self.B:
                batch.append(self.buckets[b].popleft())
            if len(batch) == self.B:
                return batch
        return batch or None

    def drain(self) -> Iterator[list[dict[str, Any]]]:
        while True:
            mb = self.flush_mixed()
            if not mb:
                break
            yield mb


class PTShardStream(IterableDataset):
    def __init__(self, train_data_path: str, shuffle_files: bool = False):
        paths = [train_data_path] if os.path.isfile(train_data_path) else glob.glob(train_data_path)
        if not paths:
            raise FileNotFoundError(f"No shards match {train_data_path}")
        self.paths = sorted(paths)
        self.shuffle = shuffle_files

    def __iter__(self) -> Iterator[dict[str, Any]]:
        paths = self.paths[:]
        if self.shuffle:
            import random

            random.shuffle(paths)
        for p in paths:
            x = torch.load(p, map_location="cpu")  # dict[str, Tensor]
            ids, labs, msk, lens = x["input_ids"], x["labels"], x["response_mask"], x["len"]
            B = ids.shape[0]
            for i in range(B):
                yield {
                    "input_ids": ids[i].tolist(),  # stay on CPU until collate
                    "labels": labs[i].tolist(),
                    "response_mask": msk[i].tolist(),
                    "len": int(lens[i]),
                }
