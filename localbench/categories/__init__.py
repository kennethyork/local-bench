"""BridgeBench-style categories. Each category yields Items and grades responses into a 0..1 score."""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from .. import gpu
from ..backends import Backend, ServerDown, ensure_ollama

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


def _generate_with_recovery(backend: Backend, messages, max_tokens, num_ctx, log):
    """Generate; if the Ollama server died, restart it and retry once, else raise ServerDown."""
    import requests
    # refused/reset connections AND streams cut off mid-answer (server killed while generating)
    dropped = (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError)
    try:
        return backend.generate(messages, max_tokens=max_tokens, temperature=0.0, num_ctx=num_ctx)
    except dropped:
        if backend.kind != "ollama":
            raise
        log("  lost connection to Ollama - waiting for it to come back, then retrying this item")
        time.sleep(3)  # the Ollama tray app usually restarts its server by itself
        if not ensure_ollama(log=log):
            raise ServerDown("Ollama stopped and could not be restarted")
        try:
            return backend.generate(messages, max_tokens=max_tokens, temperature=0.0, num_ctx=num_ctx)
        except dropped as e:
            raise ServerDown(f"Ollama keeps dropping the connection: {e}") from e


def _answer(backend: Backend, it: Item, mt: int, num_ctx, log):
    """Generate one answer. Returns (text, output_tokens, error, truncated, seconds)."""
    t = time.perf_counter()
    try:
        r = _generate_with_recovery(backend, it.messages, mt, num_ctx, log)
        return r.text, r.output_tokens, None, r.extra.get("done_reason") == "length", time.perf_counter() - t
    except ServerDown:
        raise  # abort: a dead server must never be scored as wrong answers
    except Exception as e:
        return "", 0, str(e)[:200], False, time.perf_counter() - t


def run_category(backend: Backend, cat: Category, limit: int | None, max_tokens: int | None = None,
                 log=print, on_record=None, governor=None, num_ctx: int | None = None,
                 parallel: int = 1) -> dict:
    """Ask every item and grade it. With parallel > 1, questions go out in waves of that size
    (the server must have that many slots); results, grading and logs stay in question order."""
    from concurrent.futures import ThreadPoolExecutor

    items = cat.items(limit)
    records, total_score = [], 0.0
    t0 = time.perf_counter()
    out_tokens = 0
    pool = ThreadPoolExecutor(max_workers=parallel) if parallel > 1 else None
    try:
        with gpu.GpuSampler(interval=0.2) as s:
            for start in range(0, len(items), parallel):
                wave = items[start:start + parallel]
                if governor:
                    governor.before_request()
                t_wave = time.perf_counter()
                jobs = []
                for it in wave:
                    mt = max_tokens or it.max_tokens or cat.max_tokens
                    if backend.think:
                        mt += 4096
                    jobs.append((it, mt))
                if pool:
                    answers = list(pool.map(lambda j: _answer(backend, j[0], j[1], num_ctx, log), jobs))
                else:
                    answers = [_answer(backend, it, mt, num_ctx, log) for it, mt in jobs]
                wave_s, wave_tok = time.perf_counter() - t_wave, 0
                for k, ((it, _), (text, n_out, err, trunc, item_s)) in enumerate(zip(jobs, answers)):
                    i = start + k + 1
                    g = cat.grade(it, text) if not err else Grade(0.0, f"generation error: {err}")
                    out_tokens += n_out
                    wave_tok += n_out
                    total_score += g.score
                    rec = {"id": it.id, "score": round(g.score, 4), "detail": g.detail, "output_tokens": n_out,
                           "seconds": round(item_s, 2), "truncated": trunc,
                           "response": text[-1500:], **g.extra}
                    records.append(rec)
                    if on_record:
                        on_record(it, text, rec)
                    # one line per question, so slow or paced runs never look frozen
                    log(f"  {cat.name:<13} {i:>4}/{len(items)}  score {100 * total_score / i:5.1f}  "
                        f"({n_out} tok in {item_s:.1f}s)")
                if governor:
                    governor.after_request(wave_tok, wave_s, on_rest=lambda sec: log(
                        f"                     resting {sec:.0f}s ({governor.name} governor: GPU "
                        f"{gpu.temperature_c()}C / {gpu.power_w() or 0:.0f} W / fans {gpu.fan_percent()}%)"))
    finally:
        if pool:
            pool.shutdown(wait=True)
    elapsed = time.perf_counter() - t0
    n = len(items)
    mean = total_score / n if n else 0.0
    var = sum((r["score"] - mean) ** 2 for r in records) / (n - 1) if n > 1 else 0.0
    energy_j = (s.avg_power_w or 0) * elapsed if gpu.available() else None
    wins = sum(r["score"] >= 0.999 for r in records)
    return {
        "category": cat.name, "title": cat.title, "n": n, "score": 100 * mean,
        "ci95": 100 * 1.96 * (var / n) ** 0.5 if n > 1 else 0.0,
        "wins": wins, "elapsed_s": elapsed, "output_tokens": out_tokens, "parallel": parallel,
        "energy_j": energy_j, "avg_power_w": s.avg_power_w,
        "peak_power_w": s.peak_power_w, "peak_temp_c": s.peak_temp_c, "peak_fan_pct": s.peak_fan_pct,
        "wh_per_win": (energy_j / 3600 / wins) if energy_j and wins else None,
        "extra": cat.summarize(records),
        "records": records,
    }
