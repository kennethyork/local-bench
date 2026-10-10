from __future__ import annotations

import json
import platform
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

from . import gpu
from .backends import ollama_version

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def system_info() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "cpu": platform.processor(), "gpu": gpu.gpu_info(), "ollama": ollama_version()}
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:
        pass
    return info


def save(kind: str, model_label: str, data: dict) -> Path:
    RESULTS.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = RESULTS / f"{ts}_{kind}_{slug(model_label)}.json"
    data = {"kind": kind, "model": model_label, "timestamp": datetime.now().isoformat(timespec="seconds"),
            "system": system_info(), **data}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def load_all() -> list[dict]:
    out = []
    for p in sorted(RESULTS.glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(d, dict) and "kind" in d and "model" in d:  # skip state files etc.
            d["_file"] = p.name
            out.append(d)
    return out


def median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None
