"""Every model runs through Ollama.

Model specs are Ollama model names, optionally prefixed: `ollama:llama3.2:1b` or `llama3.2:1b`.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

import requests

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
if not OLLAMA_URL.startswith("http"):
    OLLAMA_URL = "http://" + OLLAMA_URL

THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.S)


def strip_thinking(text: str) -> str:
    return THINK_RE.sub("", text).strip()


@dataclass
class GenResult:
    text: str
    prompt_tokens: int
    output_tokens: int
    ttft_s: float  # time to first generated token (includes prompt processing)
    total_s: float
    prompt_eval_s: float | None = None  # exact prompt processing time, if backend reports it
    gen_s: float | None = None  # decode time for output tokens
    thinking: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def prompt_tps(self) -> float | None:
        t = self.prompt_eval_s if self.prompt_eval_s else self.ttft_s
        return self.prompt_tokens / t if t and self.prompt_tokens else None

    @property
    def gen_tps(self) -> float | None:
        if self.gen_s and self.output_tokens:
            return self.output_tokens / self.gen_s
        decode = self.total_s - self.ttft_s
        return (self.output_tokens - 1) / decode if decode > 0 and self.output_tokens > 1 else None


class Backend:
    kind = "?"

    def __init__(self, model: str, think: bool = False):
        self.model = model
        self.think = think

    @property
    def label(self) -> str:
        return f"{self.kind}:{self.model}"

    def load(self) -> float:
        """Load model into memory. Returns load time in seconds."""
        raise NotImplementedError

    def unload(self) -> None:
        raise NotImplementedError

    def generate(self, messages: list[dict], max_tokens: int = 256, temperature: float = 0.0,
                 num_ctx: int | None = None, seed: int = 0) -> GenResult:
        raise NotImplementedError

    def info(self) -> dict:
        return {}


def ollama_name(spec: str) -> str:
    """Model spec -> Ollama model name."""
    return spec[len("ollama:"):] if spec.startswith("ollama:") else spec


def make_backend(spec: str, think: bool = False, **kw) -> Backend:
    return OllamaBackend(ollama_name(spec), think=think, **kw)


# --------------------------------------------------------------------------- Ollama

class OllamaBackend(Backend):
    kind = "ollama"

    def __init__(self, model: str, think: bool = False, keep_alive: str = "10m", num_ctx: int | None = None, **_):
        super().__init__(model, think)
        self.keep_alive = keep_alive
        self.num_ctx = num_ctx  # always sent, so Ollama never silently allocates its (huge) default context
        self._send_think = True  # flipped off if the server/model rejects the field

    def _post(self, path, payload, stream=False, timeout=3600):
        r = requests.post(f"{OLLAMA_URL}{path}", json=payload, stream=stream, timeout=timeout)
        if r.status_code >= 400:
            raise RuntimeError(f"Ollama {path} {r.status_code}: {r.text[:300]}")
        return r

    def ensure_pulled(self):
        if not is_pulled(self.model):
            print(f"  pulling {self.model} via Ollama ...")
            pull(self.model)

    def load(self) -> float:
        self.ensure_pulled()
        t0 = time.perf_counter()
        payload = {"model": self.model, "prompt": "", "keep_alive": self.keep_alive}
        if self.num_ctx:
            payload["options"] = {"num_ctx": self.num_ctx}
        r = self._post("/api/generate", payload)
        wall = time.perf_counter() - t0
        ld = r.json().get("load_duration")
        return ld / 1e9 if ld else wall

    def unload(self) -> None:
        try:
            self._post("/api/generate", {"model": self.model, "prompt": "", "keep_alive": 0}, timeout=120)
        except Exception:
            pass
        # wait until it is actually gone from /api/ps so VRAM baselines are clean
        for _ in range(50):
            if not any(m.get("name") == self.model or m.get("model") == self.model for m in ollama_ps()):
                return
            time.sleep(0.2)

    def generate(self, messages, max_tokens=256, temperature=0.0, num_ctx=None, seed=0) -> GenResult:
        opts = {"num_predict": max_tokens, "temperature": temperature, "seed": seed}
        num_ctx = num_ctx or self.num_ctx
        if num_ctx:
            opts["num_ctx"] = num_ctx
        payload = {"model": self.model, "messages": messages, "stream": True,
                   "options": opts, "keep_alive": self.keep_alive}
        if self._send_think:
            payload["think"] = self.think
        t0 = time.perf_counter()
        try:
            r = self._post("/api/chat", payload, stream=True)
        except RuntimeError as e:
            if "think" in str(e).lower() and self._send_think:
                self._send_think = False
                return self.generate(messages, max_tokens, temperature, num_ctx, seed)
            raise
        ttft = None
        text, thinking, final = [], [], {}
        for line in r.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            msg = chunk.get("message", {})
            if ttft is None and (msg.get("content") or msg.get("thinking")):
                ttft = time.perf_counter() - t0
            text.append(msg.get("content", ""))
            thinking.append(msg.get("thinking", "") or "")
            if chunk.get("done"):
                final = chunk
        total = time.perf_counter() - t0
        raw = "".join(text)
        return GenResult(
            text=strip_thinking(raw),
            thinking="".join(thinking),
            prompt_tokens=final.get("prompt_eval_count", 0),
            output_tokens=final.get("eval_count", 0),
            ttft_s=ttft if ttft is not None else total,
            total_s=total,
            prompt_eval_s=(final.get("prompt_eval_duration") or 0) / 1e9 or None,
            gen_s=(final.get("eval_duration") or 0) / 1e9 or None,
            extra={"done_reason": final.get("done_reason")},
        )

    def info(self) -> dict:
        try:
            d = self._post("/api/show", {"model": self.model}, timeout=30).json().get("details", {})
            return {"family": d.get("family"), "parameter_size": d.get("parameter_size"),
                    "quantization": d.get("quantization_level"), "format": d.get("format")}
        except Exception:
            return {}


class ServerDown(RuntimeError):
    """Ollama stopped answering. Never a model failure: callers stop and keep the item pending."""


def ollama_up() -> bool:
    try:
        return requests.get(f"{OLLAMA_URL}/api/version", timeout=3).ok
    except requests.RequestException:
        return False


def ensure_ollama(wait_s: float = 90, log=print) -> bool:
    """Make sure the Ollama server is running; start the desktop app (or `ollama serve`) if not."""
    if ollama_up():
        return True
    import shutil
    import subprocess
    candidates = [os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama app.exe"),
                  os.path.join(os.environ.get("ProgramFiles", ""), "Ollama", "ollama app.exe")]
    app = next((c for c in candidates if os.path.exists(c)), None)
    cmd = [app] if app else ([shutil.which("ollama"), "serve"] if shutil.which("ollama") else None)
    if not cmd:
        log("  Ollama is not running and could not be found to start it.")
        return False
    log("  Ollama is not running - starting it ...")
    flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    subprocess.Popen(cmd, creationflags=flags, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, close_fds=True)  # detached: outlives this benchmark
    t0 = time.time()
    while time.time() - t0 < wait_s:
        time.sleep(2)
        if ollama_up():
            log("  Ollama is back up.")
            return True
    log(f"  Ollama did not come up within {wait_s:.0f}s.")
    return False


def is_pulled(name: str) -> bool:
    names = {m["name"].lower() for m in list_ollama_models()}
    return name.lower() in names or f"{name}:latest".lower() in names


def pull(tag: str, log=print) -> None:
    """Pull a model from the Ollama library, printing sparse progress."""
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


def not_benchmarkable(name: str, entry: dict | None = None) -> str | None:
    """Why a model can't be benchmarked locally (None = fine). `entry` is its /api/tags row, if pulled."""
    tag = name.split(":", 1)[1] if ":" in name else ""
    if (entry or {}).get("remote_host") or (entry or {}).get("remote_model") or re.search(r"(^|-)cloud($|-)", tag):
        return "cloud model: runs on Ollama's servers, not your GPU"
    caps = (entry or {}).get("capabilities")
    if caps is not None and "completion" not in caps:
        return f"can't answer prompts ({', '.join(caps) or 'no capabilities'}, e.g. an embedding model)"
    return None


def ollama_entry(name: str) -> dict | None:
    low = name.lower()
    return next((m for m in list_ollama_models() if m["name"].lower() in (low, f"{low}:latest")), None)


def list_ollama_models() -> list[dict]:
    try:
        return requests.get(f"{OLLAMA_URL}/api/tags", timeout=10).json().get("models", [])
    except Exception:
        return []


def ollama_ps() -> list[dict]:
    try:
        return requests.get(f"{OLLAMA_URL}/api/ps", timeout=10).json().get("models", [])
    except Exception:
        return []


def ollama_version() -> str | None:
    try:
        return requests.get(f"{OLLAMA_URL}/api/version", timeout=5).json().get("version")
    except Exception:
        return None
