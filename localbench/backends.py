"""Model backends: Ollama (HTTP API) and Hugging Face transformers (in-process).

Model spec strings:
    ollama:llama3.2:1b          -> Ollama
    hf:Qwen/Qwen2.5-1.5B-Instruct -> transformers
    hf.co/user/repo-GGUF:Q4_K_M -> Ollama (Ollama pulls GGUF straight from HF)
    llama3.2:1b                 -> no prefix and no "/" -> Ollama
    Qwen/Qwen2.5-1.5B-Instruct  -> no prefix with "/" -> transformers
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


def make_backend(spec: str, think: bool = False, **kw) -> Backend:
    if spec.startswith("ollama:"):
        return OllamaBackend(spec[len("ollama:"):], think=think, **kw)
    if spec.startswith("hf:"):
        return HFBackend(spec[len("hf:"):], think=think, **kw)
    if spec.startswith("hf.co/") or "/" not in spec:
        return OllamaBackend(spec, think=think, **kw)
    return HFBackend(spec, think=think, **kw)


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
        names = {m["name"] for m in list_ollama_models()}
        if self.model not in names and f"{self.model}:latest" not in names:
            print(f"  pulling {self.model} via Ollama ...")
            r = self._post("/api/pull", {"model": self.model, "stream": True}, stream=True)
            last = None
            for line in r.iter_lines():
                if line:
                    st = json.loads(line).get("status")
                    if st != last:
                        print("   ", st)
                        last = st

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


# --------------------------------------------------------------------------- Hugging Face

class HFBackend(Backend):
    kind = "hf"

    def __init__(self, model: str, think: bool = False, dtype: str = "auto",
                 quant: str | None = None, **_):
        super().__init__(model, think)
        self.dtype = dtype
        self.quant = quant  # None | "8bit" | "4bit" (needs bitsandbytes)
        self.tok = None
        self.lm = None

    def load(self) -> float:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            print("  WARNING: CUDA not available, HF model will run on CPU (slow).")
        t0 = time.perf_counter()
        self.tok = AutoTokenizer.from_pretrained(self.model)
        kw = {"device_map": "cuda" if torch.cuda.is_available() else "cpu"}
        if self.quant:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=self.quant == "4bit", load_in_8bit=self.quant == "8bit",
                bnb_4bit_compute_dtype=torch.bfloat16)
        else:
            kw["dtype"] = self.dtype if self.dtype == "auto" else getattr(torch, self.dtype)
        self.lm = AutoModelForCausalLM.from_pretrained(self.model, **kw)
        self.lm.eval()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter() - t0

    def unload(self) -> None:
        import gc

        import torch
        self.lm = None
        self.tok = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _encode(self, messages):
        if self.tok.chat_template:
            text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=self.think)
        else:  # base model: plain concatenation
            text = "\n\n".join(m["content"] for m in messages) + "\n"
        return self.tok(text, return_tensors="pt", add_special_tokens=not self.tok.chat_template)

    def generate(self, messages, max_tokens=256, temperature=0.0, num_ctx=None, seed=0) -> GenResult:
        import torch
        from transformers.generation.streamers import BaseStreamer

        if self.lm is None:
            self.load()
        enc = self._encode(messages).to(self.lm.device)
        n_prompt = enc["input_ids"].shape[1]

        class _Timer(BaseStreamer):
            def __init__(s):
                s.calls = 0
                s.first = None

            def put(s, value):
                s.calls += 1
                if s.calls == 2 and s.first is None:  # call 1 is the prompt itself
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    s.first = time.perf_counter()

            def end(s):
                pass

        timer = _Timer()
        gen_kw = dict(max_new_tokens=max_tokens, streamer=timer,
                      pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        if temperature > 0:
            torch.manual_seed(seed)
            gen_kw.update(do_sample=True, temperature=temperature)
        else:
            gen_kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.lm.generate(**enc, **gen_kw)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total = time.perf_counter() - t0
        new = out[0, n_prompt:]
        raw = self.tok.decode(new, skip_special_tokens=True)
        ttft = (timer.first - t0) if timer.first else total
        m = re.search(r"<think>(.*?)(</think>|$)", raw, re.S)
        return GenResult(text=strip_thinking(raw), thinking=m.group(1) if m else "",
                         prompt_tokens=n_prompt, output_tokens=int(new.shape[0]),
                         ttft_s=ttft, total_s=total)

    def info(self) -> dict:
        d = {"dtype": self.dtype, "quantization": self.quant or "none"}
        if self.lm is not None:
            d["parameters_b"] = round(sum(p.numel() for p in self.lm.parameters()) / 1e9, 3)
            d["dtype"] = str(next(self.lm.parameters()).dtype).replace("torch.", "")
        return d
