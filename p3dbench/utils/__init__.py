"""Small shared helpers: JSONL I/O, hashing, logging, optional-dependency guards."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Iterator


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def ensure_parent(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    ensure_parent(Path(path))
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: Path, row: dict) -> None:
    ensure_parent(Path(path))
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, obj: Any, indent: int = 2) -> None:
    ensure_parent(Path(path))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=indent)
        fh.write("\n")


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class MissingDependencyError(RuntimeError):
    """Raised when a metric/format needs an optional extra that is not installed."""

    def __init__(self, dependency: str, extra: str, purpose: str):
        self.dependency = dependency
        self.extra = extra
        super().__init__(
            f"{purpose} requires '{dependency}', which is not installed. "
            f"Install it with: pip install -e \".[{extra}]\""
        )


def require(module_name: str, extra: str, purpose: str):
    """Import an optional dependency or raise a clear MissingDependencyError."""
    import importlib

    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise MissingDependencyError(module_name, extra, purpose) from exc
