from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

# Public module attributes expected by the codebase.

class Table:
    def __init__(
        self,
        columns: Sequence[str] | None = None,
        data: Sequence[Sequence[Any]] | None = None,
        log_mode: str | None = None,
    ) -> None: ...
    def add_data(self, *row: Any) -> None: ...
    def add_row(self, *row: Any) -> None: ...

class Artifact:
    def __init__(
        self,
        name: str,
        type: str,
        description: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None: ...
    def add_file(self, local_path: str, name: str | None = None) -> None: ...
    def add_dir(self, local_path: str, name: str | None = None) -> None: ...

class Run:
    name: str
    id: str
    project: str
    entity: str
    summary: MutableMapping[str, Any]
    def log(self, metrics: dict[str, float], step: int | None = None) -> None: ...
    def log_artifact(
        self,
        artifact: Artifact,
        *,
        aliases: Sequence[str] | None = None,
    ) -> None: ...
    def finish(self) -> None: ...

class Api:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def artifact(self, name: str, type: str | None = None, **kwargs: Any) -> Any: ...

# Global handle to the current run (wandb.run)
run: Run  # made non-optional to satisfy attribute access in code

def init(*args: Any, **kwargs: Any) -> Run: ...
def log(
    data: Mapping[str, Any] | dict[str, Any],
    step: int | None = None,
) -> None: ...
def finish() -> None: ...
def define_metric(name: str, *args: Any, **kwargs: Any) -> None: ...
