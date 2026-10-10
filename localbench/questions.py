"""Benchmark question sets, bundled in localbench/data/ (no network). Sources and licenses: data/LICENSES.md."""
from __future__ import annotations

import gzip
import json
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).parent / "data"


@lru_cache(maxsize=None)
def load(name: str) -> list[dict]:
    with gzip.open(DATA / f"{name}.json.gz", "rt", encoding="utf-8") as f:
        return json.load(f)
