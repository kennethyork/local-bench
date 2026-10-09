"""Benchmark the whole Ollama library (every size that fits this GPU), one model at a time.

pull -> precise fit check -> speed + categories -> delete (unless you already had it) -> next.
Progress lives in results/library_state.json, so the run can be stopped and resumed for days.
"""
from __future__ import annotations

import html
import json
import re
import shutil
import time
from datetime import datetime

import requests

from . import fit
from .backends import OLLAMA_URL, list_ollama_models
from .util import RESULTS

STATE = RESULTS / "library_state.json"
SITE = "https://ollama.com"
UA = {"User-Agent": "localbench (local benchmark runner)"}
QUANT = re.compile(r"(?:^|-)(q\d|iq\d|fp\d|bf16|int\d|mxfp|nvfp|qat|it-qat)", re.I)  # alt quants of a queued size
# not chat models / can't run here / same weights with only a context-length setting changed
NOT_CHAT = re.compile(r"(?:^|-)(text|base|cloud|mlx|\d+k)(?:-|$)", re.I)


def _wanted_tag(name: str) -> bool:
    return not (QUANT.search(name) or NOT_CHAT.search(name))


# --------------------------------------------------------------------------- scraping ollama.com

def _get(url: str) -> str:
    for attempt in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=30)
            r.raise_for_status()
            return r.text
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    return ""


def fetch_families() -> list[dict]:
    page = _get(f"{SITE}/library")
    fams = []
    for m in re.finditer(r'href="/library/([^"/:]+)" class="group w-full', page):
        name = m.group(1)
        block = page[m.end(): page.find('href="/library/', m.end()) if page.find('href="/library/', m.end()) > 0 else None]
        badges = [b.strip().lower() for b in re.findall(r"<span[^>]*rounded-md[^>]*>([^<]+)</span>", block)]
        pulls = re.search(r"<span[^>]*>([\d.]+[KMB]?)</span>\s*<span[^>]*>&nbsp;Pulls", block)
        fams.append({"family": name, "badges": badges, "pulls": pulls.group(1) if pulls else None})
    return fams


def _size_gb(s: str) -> float | None:
    m = re.match(r"([\d.]+)\s*(KB|MB|GB|TB)", s.strip(), re.I)
    if not m:
        return None
    return float(m.group(1)) * {"KB": 1e-6, "MB": 1e-3, "GB": 1, "TB": 1e3}[m.group(2).upper()]


def fetch_tags(family: str) -> list[dict]:
    page = _get(f"{SITE}/library/{family}/tags")
    tags, seen = [], set()
    for m in re.finditer(rf'href="/library/{re.escape(family)}:([^"]+)" class="md:hidden', page):
        tag = html.unescape(m.group(1))
        info = re.search(r'font-mono">\s*([0-9a-f]{12})</span>\s*•\s*([^•]+?)\s*•\s*([^•]*?)\s*•\s*'
                         r'(?:<span[^>]*>)?\s*([^•<]+?)\s*•', page[m.end(): m.end() + 1500], re.S)
        if not info or tag in seen:
            continue
        seen.add(tag)
        tags.append({"tag": f"{family}:{tag}", "digest": info.group(1), "size_gb": _size_gb(info.group(2)),
                     "context": info.group(3).strip(), "inputs": info.group(4).strip()})
    return tags


# --------------------------------------------------------------------------- planning

def choose(tags: list[dict]) -> list[dict]:
    """Default-quant chat tags, one per distinct model (digest)."""
    picked, digests = [], set()
    # explicit size tags first so 'llama3.2:3b' is preferred over its alias 'llama3.2:latest'
    for t in sorted(tags, key=lambda t: t["tag"].endswith(":latest")):
        name = t["tag"].split(":", 1)[1]
        if not _wanted_tag(name) or t["digest"] in digests:
            continue
        if "text" not in t["inputs"].lower():
            continue
        digests.add(t["digest"])
        picked.append(t)
    return picked


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"created": datetime.now().isoformat(timespec="seconds"), "preinstalled": None, "models": {}, "catalog": {}}


def save_state(st: dict):
    RESULTS.mkdir(exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1), encoding="utf-8")
    tmp.replace(STATE)


def build_queue(st: dict, budget_mb: float, refresh: bool, families: list[str] | None, log=print) -> list[dict]:
    if refresh or not st["catalog"]:
        log("Reading the Ollama library catalog from ollama.com ...")
        fams = fetch_families()
        cat = {}
        for i, f in enumerate(fams, 1):
            if "embedding" in f["badges"]:
                continue  # embedders can't answer prompts
            try:
                cat[f["family"]] = {"badges": f["badges"], "pulls": f["pulls"], "tags": choose(fetch_tags(f["family"]))}
            except Exception as e:
                log(f"  ! {f['family']}: {e}")
            if i % 25 == 0:
                log(f"  {i}/{len(fams)} families read")
            time.sleep(0.3)  # be polite to ollama.com
        st["catalog"] = cat
        st["catalog_date"] = datetime.now().isoformat(timespec="seconds")
        save_state(st)
    queue = []
    for fam, info in st["catalog"].items():
        if families and fam not in families:
            continue
        for t in info["tags"]:
            if not _wanted_tag(t["tag"].split(":", 1)[1]):  # re-apply filters to a cached catalog
                continue
            # download size alone must leave room for KV cache + overhead; precise check happens after pull
            t = {**t, "family": fam, "prefits": t["size_gb"] is not None and t["size_gb"] * 953.7 + 1500 <= budget_mb}
            queue.append(t)
    return sorted(queue, key=lambda t: t["size_gb"] or 0)


# --------------------------------------------------------------------------- ollama ops

def pull(tag: str, log=print) -> None:
    r = requests.post(f"{OLLAMA_URL}/api/pull", json={"model": tag, "stream": True}, stream=True, timeout=None)
    r.raise_for_status()
    last_pct, last_status = -10, None
    for line in r.iter_lines():
        if not line:
            continue
        d = json.loads(line)
        if d.get("error"):
            raise RuntimeError(d["error"])
        st = d.get("status", "")
        if d.get("total") and d.get("completed") is not None:
            pct = int(100 * d["completed"] / d["total"])
            if pct >= last_pct + 25:
                log(f"    {st[:30]} {pct}%")
                last_pct = pct
        elif st != last_status and not st.startswith("pulling "):
            log(f"    {st}")
        last_status = st


def delete(tag: str) -> None:
    requests.delete(f"{OLLAMA_URL}/api/delete", json={"model": tag}, timeout=60)


def models_dir_free_gb() -> float:
    import os
    path = os.environ.get("OLLAMA_MODELS") or os.path.expanduser("~/.ollama/models")
    while not os.path.exists(path):
        path = os.path.dirname(path)
    return shutil.disk_usage(path).free / 1e9


# --------------------------------------------------------------------------- main loop

def keep_awake(on: bool):
    """Ask Windows not to sleep while this process runs (per-process request; no settings changed)."""
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))
    except Exception:
        pass


def run_library(a, cats, speed_fn, bench_fn, done_fn, rebuild_fn, publish_fn, log=print):
    if not a.dry_run:
        keep_awake(True)
    try:
        _run_library(a, cats, speed_fn, bench_fn, done_fn, rebuild_fn, publish_fn, log)
    finally:
        keep_awake(False)


def _run_library(a, cats, speed_fn, bench_fn, done_fn, rebuild_fn, publish_fn, log=print):
    st = load_state()
    local = list_ollama_models()
    installed = {m["name"] for m in local}
    by_digest = {m.get("digest", "")[:12]: m["name"] for m in local}  # same weights under another tag
    if st["preinstalled"] is None:  # remember what was yours before the first library run
        st["preinstalled"] = sorted(installed)
        save_state(st)
    keep = set(st["preinstalled"]) | ({"*"} if a.keep else set())

    budget = fit.budget_mb(a.vram_headroom)
    queue = build_queue(st, budget or 0, a.refresh, [f.strip() for f in a.family.split(",")] if a.family else None, log)
    def wanted(t):
        status = st["models"].get(t["tag"], {}).get("status")
        return status in (None, "pending") or (status == "failed" and a.retry_failed)

    todo = [t for t in queue if wanted(t)]
    fits = [t for t in todo if t["prefits"]]
    too_big = [t for t in todo if not t["prefits"]]
    for t in too_big:
        st["models"][t["tag"]] = {"status": "skipped", "reason": f"{t['size_gb']} GB download exceeds the GPU budget",
                                  "size_gb": t["size_gb"]}
    save_state(st)

    counts = {}
    for v in st["models"].values():
        counts[v["status"]] = counts.get(v["status"], 0) + 1
    total_gb = sum(t["size_gb"] or 0 for t in fits)
    log(f"\nLibrary: {len(st['catalog'])} chat families, {len(queue)} model sizes. "
        f"Already: {counts.get('done', 0)} done, {counts.get('skipped', 0)} skipped (too big), "
        f"{counts.get('failed', 0)} failed.")
    log(f"To do: {len(fits)} models that fit, {total_gb:,.0f} GB to download in total "
        f"({'kept' if a.keep else 'deleted after each test, so only one at a time on disk'}).")
    if a.max_models:
        fits = fits[: a.max_models]
    if a.dry_run:
        for t in fits:
            log(f"  {t['tag']:<48} {t['size_gb']:>6.1f} GB  {t['inputs']}")
        return

    for i, t in enumerate(fits, 1):
        tag = by_digest.get(t["digest"], t["tag"])  # reuse your copy (and its results) if you have it
        label = f"ollama:{tag}"
        log(f"\n===== library [{i}/{len(fits)}] {t['tag']} ({t['size_gb']} GB)"
            + (f" = your {tag}" if tag != t["tag"] else "") + " =====")
        rec = st["models"].setdefault(t["tag"], {})
        rec.update({"size_gb": t["size_gb"], "started": datetime.now().isoformat(timespec="seconds"), "ran_as": tag})
        had_it = tag in installed or tag in keep
        try:
            if done_fn(label) and not a.redo:
                rec.update(status="done", reason="results already saved")
                save_state(st)
                continue
            if not had_it:
                need = (t["size_gb"] or 0) + 5
                if models_dir_free_gb() < need:
                    raise RuntimeError(f"not enough disk space for {t['size_gb']} GB (keeping 5 GB spare)")
                log("  pulling ...")
                pull(tag, log)
            f = fit.estimate(label, a.need_ctx, budget)
            log("  " + f.line())
            if not f.fits:
                rec.update(status="skipped", reason=f"needs ~{f.need_mb / 1024:.1f} GB VRAM")
                continue
            ok = speed_fn(label) if not a.no_speed else True
            if not ok:
                raise RuntimeError("speed test failed (see log above)")
            bench_fn(label)
            if not done_fn(label):
                raise RuntimeError("benchmark did not complete (see log above)")
            rec.update(status="done", finished=datetime.now().isoformat(timespec="seconds"))
        except KeyboardInterrupt:
            rec.update(status="pending", reason="interrupted")
            save_state(st)
            raise
        except Exception as e:
            rec.update(status="failed", reason=str(e)[:300])
            log(f"  FAILED: {e}")
        finally:
            if not had_it and "*" not in keep:
                delete(tag)
                log("  deleted (wasn't one of your models)")
            save_state(st)
            rebuild_fn()
        if a.publish and i % 5 == 0:
            publish_fn()
    if a.publish:
        publish_fn()
    log("\nLibrary run finished.")
