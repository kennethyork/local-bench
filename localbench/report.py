"""Build results/report.html — a BridgeBench-style leaderboard for local models."""
from __future__ import annotations

import json
import math
from itertools import combinations, permutations
from pathlib import Path

from .categories import DEFAULT, registry
from .util import RESULTS, load_all


def _slim(rec: dict, keep_html: bool) -> dict:
    r = {k: rec.get(k) for k in ("id", "score", "detail", "output_tokens", "seconds", "truncated", "checks")}
    r["response"] = (rec.get("response") or "")[-700:]
    if keep_html:
        r["html"] = rec.get("html")
    return r


def ratings(cats: dict, use: list[str]) -> dict[str, float]:
    """Elo-scale Bradley-Terry rating from head-to-head results on identical items.

    Every item both models attempted is a 'game' (partial credit -> fractional win). Each
    category carries equal weight per pair, so a 250-question category can't drown a 28-item one.
    """
    models = [m for m in cats if any(c in cats[m] for c in use)]
    if len(models) < 2:
        return {m: 500.0 for m in models}
    W = {p: 0.0 for p in permutations(models, 2)}
    N = {p: 0.0 for p in permutations(models, 2)}
    for c in use:
        for a, b in combinations(models, 2):
            ra = {r["id"]: r["score"] for r in cats[a].get(c, {}).get("records", [])}
            rb = {r["id"]: r["score"] for r in cats[b].get(c, {}).get("records", [])}
            shared = ra.keys() & rb.keys()
            if not shared:
                continue
            w = 1 / len(shared)
            for i in shared:
                s = 0.5 + 0.5 * (ra[i] - rb[i])
                W[a, b] += w * s
                W[b, a] += w * (1 - s)
                N[a, b] += w
                N[b, a] += w
    for p in W:  # weak prior: half a drawn game per pair keeps undefeated models finite
        W[p] += 0.25
        N[p] += 0.5
    p = {m: 1.0 for m in models}
    for _ in range(500):
        new = {}
        for i in models:
            denom = sum(N[i, j] / (p[i] + p[j]) for j in models if j != i)
            new[i] = sum(W[i, j] for j in models if j != i) / denom
        g = math.exp(sum(math.log(v) for v in new.values()) / len(new))
        p = {m: v / g for m, v in new.items()}
    return {m: round(500 + 400 * math.log10(p[m]), 1) for m in models}


def collect(kwh_price: float = 0.15) -> dict:
    perf, cats, info, system = {}, {}, {}, None
    for d in load_all():  # chronological; later runs replace earlier ones per (model, category)
        system = d.get("system") or system
        m = d["model"]
        if d.get("model_info"):
            info[m] = {**info.get(m, {}), **{k: v for k, v in d["model_info"].items() if v}}
        if d["kind"] == "perf":
            perf[m] = {k: d.get(k) for k in ("timestamp", "settings", "load_s", "vram_loaded_mb",
                                             "vram_peak_mb", "summary")}
            perf[m]["by_ctx"] = [{k: v for k, v in r.items() if k != "runs"} for r in d["by_ctx"]]
        elif d["kind"] == "bench":
            for c in d["categories"]:
                cats.setdefault(m, {})[c["category"]] = {
                    **{k: c.get(k) for k in ("title", "n", "score", "ci95", "wins", "elapsed_s", "output_tokens",
                                             "energy_j", "avg_power_w", "wh_per_win", "extra")},
                    "timestamp": d["timestamp"], "think": d.get("settings", {}).get("think"),
                    "records": [_slim(r, c["category"] == "ui") for r in c["records"]],
                }
    reg = registry()
    meta = {k: {"title": c.title, "description": c.description, "method": c.method, "default": k in DEFAULT}
            for k, c in reg.items()}
    return {"system": system, "perf": perf, "cats": cats, "info": info, "meta": meta,
            "order": [k for k in reg if any(k in v for v in cats.values())],
            "default": DEFAULT, "rating": ratings(cats, DEFAULT), "kwh_price": kwh_price}


PUBLIC_SYSTEM_KEYS = ("gpu", "ollama", "torch", "transformers")  # what a benchmark reader needs


def sanitize(data: dict, full_system: bool = False, responses: bool = True, gallery: bool = True) -> dict:
    """Trim the report for public hosting: machine details, raw model output, generated pages."""
    if not full_system and data.get("system"):
        sysinfo = {k: data["system"][k] for k in PUBLIC_SYSTEM_KEYS if data["system"].get(k)}
        if "gpu" in sysinfo:
            sysinfo["gpu"] = {k: v for k, v in sysinfo["gpu"].items() if k != "driver"}
        data["system"] = sysinfo
    for cats in data["cats"].values():
        for name, c in cats.items():
            for r in c["records"]:
                if not responses:
                    r.pop("response", None)
                if not gallery:
                    r.pop("html", None)
    return data


def build(out: Path | None = None, kwh_price: float = 0.15, public: bool = False, full_system: bool = False,
          responses: bool = True, gallery: bool = True) -> Path:
    out = out or RESULTS / "report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    data = collect(kwh_price)
    if public:
        data = sanitize(data, full_system, responses, gallery)
    out.write_text(TEMPLATE.replace("/*__DATA__*/null", json.dumps(data).replace("</", "<\\/")), encoding="utf-8")
    return out


TEMPLATE = (Path(__file__).parent / "report_template.html").read_text(encoding="utf-8")
