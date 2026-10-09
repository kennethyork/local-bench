"""BridgeBench-style categories. Each category yields Items and grades responses into a 0..1 score."""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from .. import gpu
from ..backends import Backend

SEED = 1234


@dataclass
class Item:
    id: str
    messages: list[dict]
    payload: dict = field(default_factory=dict)  # whatever grade() needs
    max_tokens: int | None = None


@dataclass
class Grade:
    score: float  # 0..1
    detail: str = ""
    extra: dict = field(default_factory=dict)


class Category:
    name = "?"
    title = "?"
    description = ""
    method = ""  # one-line methodology shown in the report
    max_tokens = 1024
    uses_limit = True  # False for small curated sets (always run in full)

    def load(self) -> list[Item]:
        raise NotImplementedError

    def grade(self, item: Item, text: str) -> Grade:
        raise NotImplementedError

    def items(self, limit: int | None) -> list[Item]:
        items = self.load()
        random.Random(SEED).shuffle(items)  # same subset for every model -> comparable + pairwise rating
        return items[:limit] if (limit and self.uses_limit) else items

    def summarize(self, records: list[dict]) -> dict:
        return {}


def registry() -> dict[str, Category]:
    from .codegen import CodeGen
    from .debugging import Debugging
    from .hallucination import Hallucination
    from .reasoning import Knowledge, Reasoning
    from .refactoring import Refactoring
    from .security import Security
    from .ui import UIGen

    cats = [Debugging(), CodeGen(), Refactoring(), Security(), Hallucination(), UIGen(), Reasoning(), Knowledge()]
    return {c.name: c for c in cats}


DEFAULT = ["debugging", "codegen", "refactoring", "security", "hallucination", "ui", "reasoning"]


def run_category(backend: Backend, cat: Category, limit: int | None, max_tokens: int | None = None,
                 log=print, on_record=None, governor=None, num_ctx: int | None = None) -> dict:
    items = cat.items(limit)
    records, total_score = [], 0.0
    t0 = time.perf_counter()
    out_tokens = 0
    with gpu.GpuSampler(interval=0.2) as s:
        for i, it in enumerate(items, 1):
            mt = max_tokens or it.max_tokens or cat.max_tokens
            if backend.think:
                mt += 4096
            if governor:
                governor.before_request()
            t_item = time.perf_counter()
            try:
                r = backend.generate(it.messages, max_tokens=mt, temperature=0.0, num_ctx=num_ctx)
                text, n_out, err, trunc = r.text, r.output_tokens, None, r.extra.get("done_reason") == "length"
            except Exception as e:
                text, n_out, err, trunc = "", 0, str(e)[:200], False
            item_s = time.perf_counter() - t_item
            if governor:
                governor.after_request(n_out)
            g = cat.grade(it, text) if not err else Grade(0.0, f"generation error: {err}")
            out_tokens += n_out
            total_score += g.score
            rec = {"id": it.id, "score": round(g.score, 4), "detail": g.detail, "output_tokens": n_out,
                   "seconds": round(item_s, 2), "truncated": trunc,
                   "response": text[-1500:], **g.extra}
            records.append(rec)
            if on_record:
                on_record(it, text, rec)
            if i % 5 == 0 or i == len(items):
                log(f"  {cat.name:<13} {i:>4}/{len(items)}  score {100 * total_score / i:5.1f}")
    elapsed = time.perf_counter() - t0
    n = len(items)
    mean = total_score / n if n else 0.0
    var = sum((r["score"] - mean) ** 2 for r in records) / (n - 1) if n > 1 else 0.0
    energy_j = (s.avg_power_w or 0) * elapsed if gpu.available() else None
    wins = sum(r["score"] >= 0.999 for r in records)
    return {
        "category": cat.name, "title": cat.title, "n": n, "score": 100 * mean,
        "ci95": 100 * 1.96 * (var / n) ** 0.5 if n > 1 else 0.0,
        "wins": wins, "elapsed_s": elapsed, "output_tokens": out_tokens,
        "energy_j": energy_j, "avg_power_w": s.avg_power_w,
        "peak_power_w": s.peak_power_w, "peak_temp_c": s.peak_temp_c,
        "wh_per_win": (energy_j / 3600 / wins) if energy_j and wins else None,
        "extra": cat.summarize(records),
        "records": records,
    }
