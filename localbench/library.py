"""Benchmark the whole Ollama library (every size that fits this GPU), one model at a time.

pull -> precise fit check -> speed + categories -> delete (unless you already had it) -> next.
Progress lives in results/library_state.json, so the run can be stopped and resumed for days.
"""
from __future__ import annotations

import html
import json
import re
import shutil
import threading
import time
from datetime import datetime

import requests

from . import fit
from .backends import (OLLAMA_URL, ensure_ollama, is_pulled, list_ollama_models, not_benchmarkable, ollama_entry,
                       ollama_up, pull, url)
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

def delete(tag: str, log=print) -> bool:
    try:
        r = requests.delete(f"{url()}/api/delete", json={"model": tag}, timeout=60)
        if r.status_code == 404:  # not on disk (e.g. its download failed): nothing to clean up
            return True
        r.raise_for_status()
        return True
    except requests.RequestException as e:
        log(f"  could not delete {tag} yet ({e.__class__.__name__}); will retry on the next run")
        return False


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
    if not ensure_ollama(log=log):
        log("Ollama isn't running. Start it and run this again.")
        return
    st = load_state()
    for tag in st.get("to_delete", []):  # cleanups that failed last time (server was down)
        if delete(tag, log):
            st["to_delete"].remove(tag)
    local = list_ollama_models()
    installed = {m["name"] for m in local}
    by_digest = {m.get("digest", "")[:12]: m["name"] for m in local}  # same weights under another tag
    if st["preinstalled"] is None:  # remember what was yours before the first library run
        st["preinstalled"] = sorted(installed)
        save_state(st)
    keep = set(st["preinstalled"]) | ({"*"} if a.keep else set())

    gpu_b = fit.gpu_budget_mb(a.vram_headroom)  # what the card can hold -> decides COMPATIBLE
    queue = build_queue(st, gpu_b or 0, a.refresh, [f.strip() for f in a.family.split(",")] if a.family else None, log)
    for rec in st["models"].values():  # older state files saved "too big" as a permanent skip
        if rec.get("status") == "skipped" and ("GPU budget" in rec.get("reason", "") or "VRAM" in rec.get("reason", "")):
            rec["status"] = "too_big"

    def wanted(t):
        rec = st["models"].get(t["tag"], {})
        status = rec.get("status")
        if status == "too_big":  # measured after download; only retried if the card could hold it after all
            return not rec.get("need_mb") or rec["need_mb"] <= (gpu_b or 0)
        if status == "done":  # benchmarked once = done for good, unless forced
            return a.redo
        return status in (None, "pending", "deferred") or (status == "failed" and a.retry_failed)

    todo = [t for t in queue if wanted(t)]
    compatible = [t for t in todo if t["prefits"]]
    incompatible = sum(1 for t in queue if not t["prefits"]) + sum(
        1 for t in queue if t["prefits"] and st["models"].get(t["tag"], {}).get("status") == "too_big" and not wanted(t))
    save_state(st)

    counts = {}
    for v in st["models"].values():
        counts[v["status"]] = counts.get(v["status"], 0) + 1
    total_gb = sum(t["size_gb"] or 0 for t in compatible)
    log(f"\nOllama library: {len(st['catalog'])} chat families, {len(queue)} local model sizes. "
        f"This GPU holds up to {(gpu_b or 0) / 1024:.1f} GB per model, so {incompatible} are too big for it.")
    log(f"Already: {counts.get('done', 0)} done, {counts.get('skipped', 0)} skipped (can't run locally), "
        f"{counts.get('failed', 0)} failed.")
    log(f"To do: {len(compatible)} compatible models, {total_gb:,.0f} GB to download in total "
        f"({'kept' if a.keep else 'deleted after each test, so only one at a time on disk'}).")
    if a.max_models:
        compatible = compatible[: a.max_models]
    if a.dry_run:
        for t in compatible:
            log(f"  {t['tag']:<48} {t['size_gb']:>6.1f} GB  {t['inputs']}")
        return

    def room_now(need_mb: float) -> bool:
        free_b = fit.free_budget_mb(a.vram_headroom)
        return free_b is None or need_mb <= free_b

    prefetch: dict[str, threading.Thread] = {}  # tag -> background download of the NEXT model

    def start_prefetch(nt, current_gb: float):
        """Download the next model while this one is benchmarked (via your normal Ollama server,
        which doesn't restart between models). Skipped when it wouldn't be needed or wouldn't fit."""
        if nt is None or a.no_prefetch:
            return
        ntag = by_digest.get(nt["digest"], nt["tag"])
        if ntag in installed or ntag in keep or ntag in prefetch or (done_fn(f"ollama:{ntag}") and not a.redo):
            return
        if not room_now((nt["size_gb"] or 0) * 953.7 + 1500):
            return
        if models_dir_free_gb() < (nt["size_gb"] or 0) + current_gb + 5:
            return
        try:
            requests.get(f"{OLLAMA_URL}/api/version", timeout=3).raise_for_status()
        except requests.RequestException:
            return  # normal Ollama server not running: download normally when it's this model's turn
        if ntag not in st.setdefault("to_delete", []):  # if the run is interrupted, clean it up next time
            st["to_delete"].append(ntag)
            save_state(st)

        def bg():
            try:
                pull(ntag, log=lambda *_: None, base=OLLAMA_URL)
            except Exception:
                pass  # the normal pull path will retry and report

        th = threading.Thread(target=bg, daemon=True)
        th.start()
        prefetch[ntag] = th
        log(f"  downloading the next model in the background: {ntag} ({nt['size_gb']} GB)")

    def process(t, header: str, nxt=None) -> str:
        """Pull -> check -> benchmark -> clean up one model. Returns its final status."""
        tag = by_digest.get(t["digest"], t["tag"])  # reuse your copy (and its results) if you have it
        label = f"ollama:{tag}"
        log(f"\n===== {header} {t['tag']} ({t['size_gb']} GB)" + (f" = your {tag}" if tag != t["tag"] else "") + " =====")
        rec = st["models"].setdefault(t["tag"], {})
        rec.update({"size_gb": t["size_gb"], "started": datetime.now().isoformat(timespec="seconds"), "ran_as": tag})
        had_it = tag in installed or tag in keep
        if done_fn(label) and not a.redo:
            rec.update(status="done", reason="results already saved")
            save_state(st)
            return "done"
        est = (t["size_gb"] or 0) * 953.7 + 1500
        if not room_now(est):  # compatible, but other apps hold the VRAM right now: don't even download yet
            rec.update(status="deferred", reason=f"other apps held {fit.other_apps_gb():.1f} GB of VRAM")
            log(f"  deferred: other apps hold {fit.other_apps_gb():.1f} GB of VRAM right now - will retry at the end")
            save_state(st)
            return "deferred"
        try:
            if not had_it:
                if tag in prefetch:
                    log("  finishing the background download ...")
                    prefetch.pop(tag).join()
                if not is_pulled(tag):
                    if models_dir_free_gb() < (t["size_gb"] or 0) + 5:
                        raise RuntimeError(f"not enough disk space for {t['size_gb']} GB (keeping 5 GB spare)")
                    log("  pulling ...")
                    pull(tag, log)
                if tag in st.get("to_delete", []):  # it's this model's turn now; the finally below cleans it up
                    st["to_delete"].remove(tag)
            why = not_benchmarkable(tag, ollama_entry(tag))  # some non-chat families aren't badged as such
            if why:
                rec.update(status="skipped", reason=why)
                log(f"  skip: {why}")
                return "skipped"
            f = fit.estimate(label, a.need_ctx, gpu_b)
            log("  " + f.line())
            if not f.fits:  # measured need beats the download-size estimate: too big for this card
                rec.update(status="too_big", need_mb=f.need_mb, reason=f"needs ~{f.need_mb / 1024:.1f} GB VRAM")
                return "too_big"
            if not room_now(f.need_mb or 0):
                rec.update(status="deferred", reason=f"other apps held {fit.other_apps_gb():.1f} GB of VRAM")
                log(f"  deferred: needs ~{f.need_mb / 1024:.1f} GB, other apps hold {fit.other_apps_gb():.1f} GB")
                return "deferred"
            start_prefetch(nxt, t["size_gb"] or 0)  # overlap the next download with this benchmark
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
            if not ollama_up():  # the server died: not this model's fault, keep it queued
                rec.update(status="pending", reason="Ollama stopped during the test")
                log("  Ollama stopped - this model stays queued.")
            else:
                rec.update(status="failed", reason=str(e)[:300])
                log(f"  FAILED: {e}")
        finally:
            if not had_it and "*" not in keep:
                if delete(tag, log):
                    log("  deleted (wasn't one of your models)")
                elif tag not in st.setdefault("to_delete", []):
                    st["to_delete"].append(tag)
            save_state(st)
            rebuild_fn()
        return rec.get("status")

    deferred = []
    for i, t in enumerate(compatible, 1):
        status = process(t, f"library [{i}/{len(compatible)}]", compatible[i] if i < len(compatible) else None)
        if status == "deferred":
            deferred.append(t)
        elif status == "pending" and not ensure_ollama(log=log):
            log("Ollama could not be restarted - stopping. Run again once it's up; progress is saved.")
            return
        if a.publish and i % 5 == 0:
            publish_fn()
    if deferred:
        log(f"\nRetrying {len(deferred)} model(s) deferred for VRAM ...")
        still = [t for t in deferred if process(t, "deferred") == "deferred"]
        if still:
            log(f"\n{len(still)} compatible model(s) are still waiting for VRAM: other apps hold "
                f"{fit.other_apps_gb():.1f} GB. Close them and run again; these will be picked up then.")
    if a.publish:
        publish_fn()
    log("\nLibrary run finished.")
