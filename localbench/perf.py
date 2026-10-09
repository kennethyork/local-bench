"""Speed / resource benchmark: load time, TTFT, prompt & generation throughput, VRAM, power."""
from __future__ import annotations

import random
import time

from . import gpu
from .backends import Backend
from .governor import Governor
from .util import median

_SENTENCES = [
    "The river delta shifted its course after the spring floods, leaving the old harbor silted and quiet.",
    "Engineers measured the bridge deflection at noon, when the steel was warmest and traffic heaviest.",
    "A small bakery on the corner sold out of rye bread every morning before seven o'clock.",
    "Migrating geese followed the coastline south, resting on sandbars when the wind turned against them.",
    "The committee reviewed fourteen proposals and funded three, citing budget limits and overlapping goals.",
    "Under the microscope, the leaf revealed thousands of stomata opening and closing with the humidity.",
    "Archivists digitized the ledgers page by page, correcting faded entries against the shipping manifests.",
    "By late autumn the orchard had produced twice the expected harvest, overwhelming the local cider press.",
    "The telescope tracked a faint comet for six nights before clouds rolled in from the western ridge.",
    "Volunteers repaired the trail with flat stones, building drainage channels where water pooled after rain.",
]


def build_prompt(target_tokens: int, nonce: int) -> list[dict]:
    """~target_tokens of filler notes + an instruction that invites a long answer.
    A random nonce at the very start defeats prompt-prefix caching between repetitions."""
    rng = random.Random(nonce)
    words_needed = int(target_tokens * 0.72)  # ~1.35 tokens per English word
    parts, n = [f"Session {nonce}. Field notes follow."], 0
    while n < words_needed:
        s = rng.choice(_SENTENCES)
        parts.append(s)
        n += len(s.split())
    notes = " ".join(parts)
    return [{"role": "user", "content": notes + "\n\nUsing the notes above as loose inspiration, write a long, "
             "detailed, multi-section essay about how small communities adapt to environmental change. "
             "Keep writing until you have covered at least ten distinct points."}]


def check_on_gpu(backend: Backend):
    """Abort if the model didn't load entirely into VRAM (CPU spill = not benchmarkable on this GPU)."""
    if backend.kind == "ollama":
        from .fit import ollama_spilled
        reason = ollama_spilled(backend.model)
        if reason:
            backend.unload()
            raise RuntimeError(f"does not fit this GPU: {reason}")


def run_perf(backend: Backend, ctx_sizes=(128, 1024, 4096), gen_tokens=256, reps=3,
             log=print, governor: Governor | None = None) -> dict:
    """Latency/throughput test. Under a governor only the power cap and thermal back-off
    apply; pacing would corrupt the numbers being measured."""
    num_ctx = max(ctx_sizes) + gen_tokens + 512
    if hasattr(backend, "num_ctx"):
        backend.num_ctx = num_ctx  # load at the context we test, not the server default
    log(f"[perf] {backend.label}: unloading for a cold start")
    backend.unload()
    time.sleep(0.5)
    baseline_mb = gpu.used_mb()

    with gpu.GpuSampler() as s_load:
        load_s = backend.load()
    log(f"  load: {load_s:.2f}s")
    check_on_gpu(backend)

    # warmup (also makes Ollama allocate the KV cache at the final num_ctx)
    backend.generate(build_prompt(64, 1), max_tokens=16, num_ctx=num_ctx)
    loaded_mb = gpu.used_mb()

    rows, peak_mb = [], loaded_mb
    for ctx in ctx_sizes:
        runs = []
        with gpu.GpuSampler() as s:
            for rep in range(reps):
                if governor:
                    governor.before_request()
                r = backend.generate(build_prompt(ctx, random.randint(1, 10**9)),
                                     max_tokens=gen_tokens, num_ctx=num_ctx, seed=rep)
                runs.append(r)
                log(f"  ctx~{ctx:>6} rep {rep + 1}/{reps}: prompt {r.prompt_tokens} tok, "
                    f"TTFT {r.ttft_s * 1000:7.1f} ms, prefill {r.prompt_tps or 0:8.1f} tok/s, "
                    f"gen {r.gen_tps or 0:6.1f} tok/s ({r.output_tokens} tok)")
        peak_mb = max(peak_mb, s.peak_mb)
        rows.append({
            "target_ctx": ctx,
            "prompt_tokens": median([r.prompt_tokens for r in runs]),
            "output_tokens": median([r.output_tokens for r in runs]),
            "ttft_ms": median([r.ttft_s * 1000 for r in runs]),
            "prompt_tps": median([r.prompt_tps for r in runs]),
            "gen_tps": median([r.gen_tps for r in runs]),
            "total_s": median([r.total_s for r in runs]),
            "gpu_util_avg": s.avg_util,
            "power_w_avg": s.avg_power_w,
            "vram_peak_mb": s.peak_mb or None,
            "peak_power_w": s.peak_power_w,
            "peak_temp_c": s.peak_temp_c,
            "runs": [{"prompt_tokens": r.prompt_tokens, "output_tokens": r.output_tokens,
                      "ttft_ms": r.ttft_s * 1000, "prompt_tps": r.prompt_tps, "gen_tps": r.gen_tps,
                      "total_s": r.total_s} for r in runs],
        })

    info = backend.info()
    backend.unload()
    gen_all = [row["gen_tps"] for row in rows]
    power = median([row["power_w_avg"] for row in rows])
    gen_med = median(gen_all)
    return {
        "settings": {"ctx_sizes": list(ctx_sizes), "gen_tokens": gen_tokens, "reps": reps, "num_ctx": num_ctx},
        "model_info": info,
        "load_s": load_s,
        "vram_baseline_mb": baseline_mb,
        "vram_loaded_mb": loaded_mb - baseline_mb if gpu.available() else None,
        "vram_peak_mb": peak_mb - baseline_mb if gpu.available() else None,
        "summary": {
            "gen_tps": gen_med,
            "ttft_ms_short": rows[0]["ttft_ms"],
            "prompt_tps_long": rows[-1]["prompt_tps"],
            "tokens_per_joule": (gen_med / power) if gen_med and power else None,
            "peak_power_w": max((r["peak_power_w"] or 0 for r in rows), default=None),
            "peak_temp_c": max((r["peak_temp_c"] or 0 for r in rows), default=None),
        },
        "by_ctx": rows,
    }


def cool_down(target_c: int = 50, max_s: float = 90.0, log=print):
    """Wait for the GPU to cool so each profile starts from a comparable temperature."""
    t0 = time.perf_counter()
    t = gpu.temperature_c()
    if t is not None and t > target_c:
        log(f"  cooling GPU {t}C -> {target_c}C ...")
    while t is not None and t > target_c and time.perf_counter() - t0 < max_s:
        time.sleep(2)
        t = gpu.temperature_c()


def run_sustained(backend: Backend, profile: str, requests: int = 8, gen_tokens: int = 256,
                  prompt_tokens: int = 1024, log=print, max_temp_c: int | None = 80) -> dict:
    """A realistic back-to-back workload under one governor profile (pacing ON).

    Measures what the profile is for: effective throughput vs. watts, heat and energy.
    """
    with Governor(profile, max_temp_c=max_temp_c) as gov, gpu.GpuSampler(interval=0.1) as s:
        t0 = time.perf_counter()
        out = 0
        for i in range(requests):
            gov.before_request()
            r = backend.generate(build_prompt(prompt_tokens, random.randint(1, 10**9)), max_tokens=gen_tokens,
                                 num_ctx=prompt_tokens + gen_tokens + 512, seed=i)
            out += r.output_tokens
            gov.after_request(r.output_tokens)
        wall = time.perf_counter() - t0
    energy = (s.avg_power_w or 0) * wall
    res = {"profile": profile, "requests": requests, "output_tokens": out, "wall_s": wall,
           "effective_tps": out / wall if wall else None, "avg_power_w": s.avg_power_w,
           "peak_power_w": s.peak_power_w, "peak_temp_c": s.peak_temp_c, "energy_j": energy,
           "tokens_per_joule": out / energy if energy else None, "governor": gov.summary()}
    log(f"  {profile:<8} {res['effective_tps']:7.1f} tok/s effective | avg {s.avg_power_w or 0:5.0f} W | "
        f"peak {s.peak_power_w or 0:5.0f} W | peak {s.peak_temp_c}C | {res['tokens_per_joule'] or 0:.2f} tok/J | "
        f"cap: {gov.cap.outcome}")
    return res