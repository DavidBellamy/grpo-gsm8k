import atexit
import contextlib
import csv
import os
import time
from collections.abc import Iterator
from pathlib import Path

import torch


class MemLogger:
    def __init__(self, path: str = "mem.csv") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", newline="")
        self._w = csv.writer(self._fh)
        if self.path.stat().st_size == 0:
            self._w.writerow(["ts", "tag", "alloc_gb", "resv_gb", "peak_gb", "free_gb", "total_gb"])
        self._fh.flush()

    def row(self, tag: str) -> None:
        a = torch.cuda.memory_allocated() / (1024**3)
        r = torch.cuda.memory_reserved() / (1024**3)
        p = torch.cuda.max_memory_allocated() / (1024**3)
        free, tot = torch.cuda.mem_get_info()
        self._w.writerow(
            [
                f"{time.time():.3f}",
                tag,
                f"{a:.2f}",
                f"{r:.2f}",
                f"{p:.2f}",
                f"{free/(1024**3):.2f}",
                f"{tot/(1024**3):.2f}",
            ]
        )
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


_memlog = MemLogger(os.getenv("MEMLOG_PATH", "logs/mem.csv"))


def mem_snapshot(tag: str) -> None:
    _memlog.row(tag)  # no prints; goes to CSV


@contextlib.contextmanager
def mem_region(tag: str) -> Iterator[None]:
    torch.cuda.reset_peak_memory_stats()
    mem_snapshot(f"{tag}:start")
    try:
        yield
    finally:
        mem_snapshot(f"{tag}:end")


atexit.register(_memlog.close)
