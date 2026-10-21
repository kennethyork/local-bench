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

    def grade_result(self, item: Item, text: str, extra: dict) -> Grade:
        """Grade with the backend's extras too (tool calls, for the Agent category). Defaults to grade()."""
        return self.grade(item, text)

    def items(self, limit: int | None) -> list[Item]:
        items = self.load()
        random.Random(SEED).shuffle(items)  # same subset for every model -> comparable + pairwise rating
        return items[:limit] if (limit and self.uses_limit) else items

    def summarize(self, records: list[dict]) -> dict:
        return {}


def registry() -> dict[str, Category]:
    from .backend import BackEnd
    from .codegen import CodeGen
    from .debugging import Debugging
    from .hallucination import Hallucination
    from .reasoning import Knowledge, Reasoning
    from .refactoring import Refactoring
    from .security import Security
    from .stack import Stack
    from .agent import Agent
    from .ui import UIGen

    cats = [Debugging(), CodeGen(), Refactoring(), Security(), Hallucination(), UIGen(), Reasoning(), Knowledge(), BackEnd(),
            Stack(), Agent()]
    return {c.name: c for c in cats}


DEFAULT = ["debugging", "codegen", "refactoring", "security", "hallucination", "ui", "reasoning"]

# `-c training`: what to watch when fine-tuning a coding model for a Tauri (Rust + TypeScript) agent.
# Stack and Agent measure the target skills; debugging, codegen and hallucination catch a fine-tune
# that gains on its own stack by forgetting general coding or by starting to make things up.
PRESETS = {"training": ["stack", "agent", "debugging", "codegen", "hallucination"]}


def _generate_with_recovery(backend: Backend, messages, max_tokens, num_ctx, log, tools=None):
    """Generate; if the Ollama server died, restart it and retry once, else raise ServerDown."""
    import requests
    # refused/reset connections AND streams cut off mid-answer (server killed while generating)
    dropped = (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError)
    kw = {"tools": tools} if tools else {}  # only tool categories pass it, so plain backends needn't take it
    try:
        return backend.generate(messages, max_tokens=max_tokens, temperature=0.0, num_ctx=num_ctx, **kw)
    except dropped:
        if backend.kind != "ollama":
            raise
        log("  lost connection to Ollama - waiting for it to come back, then retrying this item")
        time.sleep(3)  # the Ollama tray app usually restarts its server by itself
        if not ensure_ollama(log=log):
            raise ServerDown("Ollama stopped and could not be restarted")
        try:
            return backend.generate(messages, max_tokens=max_tokens, temperature=0.0, num_ctx=num_ctx, **kw)
        except dropped as e:
            raise ServerDown(f"Ollama keeps dropping the connection: {e}") from e


def _answer(backend: Backend, it: Item, mt: int, num_ctx, log):
    """Generate one answer. Returns (text, output_tokens, error, truncated, seconds, extra)."""
    t = time.perf_counter()
    try:
        r = _generate_with_recovery(backend, it.messages, mt, num_ctx, log, tools=it.payload.get("tools"))
        return (r.text, r.output_tokens, None, r.extra.get("done_reason") == "length", time.perf_counter() - t,
                r.extra)
    except ServerDown:
        raise  # abort: a dead server must never be scored as wrong answers
    except Exception as e:
        return "", 0, str(e)[:200], False, time.perf_counter() - t, {}


MAX_REST_BASIS_S = 90.0  # a rest is sized from at most this much preceding work (keeps rests bounded)


def run_category(backend: Backend, cat: Category, limit: int | None, max_tokens: int | None = None,
                 log=print, on_record=None, governor=None, num_ctx: int | None = None,
                 parallel: int = 1) -> dict:
    """Ask every item and grade it, keeping up to `parallel` questions in flight (the server must have
    that many slots). A new question starts as soon as one finishes; only when the governor reports
    heat, power or fan pressure does it let the in-flight answers finish and rest. Fixed-rate governor
    profiles (silent, balanced) pace after every `parallel` answers instead. Records stay in question order."""
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    items = cat.items(limit)
    jobs = []
    for it in items:
        mt = max_tokens or it.max_tokens or cat.max_tokens
        jobs.append((it, mt + 4096 if backend.think else mt))
    records: list = [None] * len(items)
    total_score, done_n, out_tokens = 0.0, 0, 0
    seg = {"tok": 0, "t0": time.perf_counter(), "since_rest": 0}
    t0 = time.perf_counter()

    def work(job):
        it, mt = job
        text, n_out, err, trunc, item_s, extra = _answer(backend, it, mt, num_ctx, log)
        return text, n_out, trunc, item_s, (cat.grade_result(it, text, extra or {}) if not err
                                            else Grade(0.0, f"generation error: {err}"))

    def finish(i, result):
        nonlocal total_score, done_n, out_tokens
        it = jobs[i][0]
        text, n_out, trunc, item_s, g = result
        done_n += 1
        out_tokens += n_out
        seg["tok"] += n_out
        seg["since_rest"] += 1
        total_score += g.score
        rec = {"id": it.id, "score": round(g.score, 4), "detail": g.detail, "output_tokens": n_out,
               "seconds": round(item_s, 2), "truncated": trunc, "response": text[-1500:], **g.extra}
        records[i] = rec
        if on_record:
            on_record(it, text, rec)
        # one line per question, so slow or paced runs never look frozen
        log(f"  {cat.name:<13} {done_n:>4}/{len(items)}  score {100 * total_score / done_n:5.1f}  "
            f"({n_out} tok in {item_s:.1f}s)")

    def rest():
        if governor:
            governor.after_request(seg["tok"], min(time.perf_counter() - seg["t0"], MAX_REST_BASIS_S),
                                   on_rest=lambda sec: log(
                f"                     resting {sec:.0f}s ({governor.name} governor: GPU "
                f"{gpu.temperature_c()}C / {gpu.power_w() or 0:.0f} W / fans {gpu.fan_percent()}%)"))
            governor.before_request()  # safety stop, fan and thermal waits
        seg.update(tok=0, t0=time.perf_counter(), since_rest=0)

    continuous = governor is None or governor.continuous()
    pool = ThreadPoolExecutor(max_workers=max(parallel, 1))
    try:
        with gpu.GpuSampler(interval=0.2) as s:
            if governor:
                governor.before_request()
            nxt, inflight = 0, {}
            while nxt < len(jobs) or inflight:
                # fixed-rate profiles work in batches of `parallel`, then pace; the rest refill freely
                while nxt < len(jobs) and len(inflight) < parallel and (
                        continuous or seg["since_rest"] + len(inflight) < parallel):
                    inflight[pool.submit(work, jobs[nxt])] = nxt
                    nxt += 1
                done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                for f in sorted(done, key=inflight.get):
                    finish(inflight.pop(f), f.result())
                if not governor or nxt >= len(jobs):
                    continue
                if (governor.needs_rest() if continuous else seg["since_rest"] >= parallel):
                    for f in sorted(inflight, key=inflight.get):  # let the answers in progress finish
                        finish(inflight[f], f.result())
                    inflight.clear()
                    rest()
            if governor:
                rest()  # pay off any pressure before the next category or model
    finally:
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
