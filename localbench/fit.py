"""Only benchmark what this GPU can actually hold.

Estimate = weights + KV cache at the context we will really use + runtime overhead.
Two budgets:
  * gpu_budget_mb  - what the card can hold (total VRAM x headroom, minus a desktop reserve).
                     Decides whether a model is COMPATIBLE, i.e. part of the benchmark at all.
  * free_budget_mb - what's free right now (other apps subtracted). Decides only WHEN a
                     compatible model runs: if other apps are in the way it's deferred, not dropped.
After loading, Ollama's own /api/ps is checked too: any CPU spill stops that model.
"""
from __future__ import annotations

from dataclasses import dataclass

import requests

from . import gpu
from .backends import OLLAMA_URL, list_ollama_models, ollama_ps

OVERHEAD_MB = 800  # CUDA context, compute buffers, graph scratch


@dataclass
class Fit:
    model: str
    need_mb: float | None
    budget_mb: float | None
    weights_mb: float | None = None
    kv_mb: float | None = None
    note: str = ""

    @property
    def fits(self) -> bool:
        if self.need_mb is None or self.budget_mb is None:
            return True  # unknown -> allow, the post-load spill check still protects
        return self.need_mb <= self.budget_mb

    def line(self) -> str:
        if self.need_mb is None:
            return f"?  {self.model}  (size unknown: {self.note})"
        mark = "OK " if self.fits else "NO "
        return (f"{mark} {self.model:<44} needs ~{self.need_mb / 1024:5.1f} GB "
                f"(weights {self.weights_mb / 1024:.1f} + KV {self.kv_mb / 1024:.1f}) of {self.budget_mb / 1024:.1f} GB"
                + (f"  {self.note}" if self.note else ""))


DESKTOP_RESERVE_MB = 1024  # Windows desktop + driver; always in use on a display GPU


def gpu_budget_mb(headroom: float = 0.92) -> float | None:
    """What this card can hold for a model, regardless of what other apps happen to use now."""
    info = gpu.gpu_info()
    if not info:
        return None
    return info["vram_total_mb"] * headroom - DESKTOP_RESERVE_MB


def free_budget_mb(headroom: float = 0.92) -> float | None:
    """What a model can use right now (other apps' VRAM subtracted)."""
    info = gpu.gpu_info()
    if not info:
        return None
    other = max(gpu.used_mb(), DESKTOP_RESERVE_MB)  # desktop, browser, other apps
    return info["vram_total_mb"] * headroom - other


def other_apps_gb() -> float:
    """VRAM held by things other than the desktop reserve (apps worth closing)."""
    return max(gpu.used_mb() - DESKTOP_RESERVE_MB, 0) / 1024


def _kv_mb(layers, kv_heads, head_dim_k, head_dim_v, ctx, bytes_per=2) -> float:
    return layers * kv_heads * (head_dim_k + head_dim_v) * ctx * bytes_per / 2**20


def _same(a: str, b: str) -> bool:
    """Same Ollama model name? (case-insensitive; ':latest' optional)"""
    a, b = a.lower(), b.lower()
    return a == b or a == f"{b}:latest"


def estimate_ollama(name: str, num_ctx: int, budget: float | None) -> Fit:
    size = next((m.get("size") for m in list_ollama_models()
                 if _same(m["name"], name)), None)
    if size is None:
        return Fit(f"ollama:{name}", None, budget, note="not pulled yet")
    weights = size / 2**20
    try:
        mi = requests.post(f"{OLLAMA_URL}/api/show", json={"model": name}, timeout=30).json().get("model_info", {})
        arch = mi.get("general.architecture", "")
        g = lambda k: mi.get(f"{arch}.{k}")  # noqa: E731
        layers, heads = g("block_count"), g("attention.head_count")
        kv_heads = g("attention.head_count_kv") or heads
        if isinstance(kv_heads, list):  # per-layer list in some archs
            kv_heads = max(kv_heads)
        hd_k = g("attention.key_length") or (g("embedding_length") // heads if heads else None)
        hd_v = g("attention.value_length") or hd_k
        kv = _kv_mb(layers, kv_heads, hd_k, hd_v, num_ctx) if all((layers, kv_heads, hd_k)) else weights * 0.1
    except Exception:
        kv = weights * 0.1
    return Fit(f"ollama:{name}", weights + kv + OVERHEAD_MB, budget, weights, kv)


def estimate(spec_label: str, num_ctx: int, budget: float | None) -> Fit:
    """spec_label is 'ollama:<name>'."""
    return estimate_ollama(spec_label.partition(":")[2], num_ctx, budget)


def ollama_spilled(name: str) -> str | None:
    """After load: did Ollama put any of the model on the CPU? Returns a reason if so."""
    for m in ollama_ps():
        if _same(m.get("name", ""), name) or _same(m.get("model", ""), name):
            size, vram = m.get("size", 0), m.get("size_vram", 0)
            if size and vram < size * 0.99:
                return f"{(size - vram) / 2**30:.1f} GB spilled to CPU RAM ({100 * vram / size:.0f}% on GPU)"
    return None
