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
_active_url = [OLLAMA_URL]  # switched to the private benchmark server while one is running (see server.py)


def url() -> str:
    """Base URL of the Ollama server benchmarks talk to."""
    return _active_url[0]


def use_url(u: str | None) -> None:
    _active_url[0] = u or OLLAMA_URL

THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.S)

# --------------------------------------------------------------------------- tool calls
# The Agent category sends tools the way Ocadox does. A model whose template has no tool support is
# refused by Ollama ("does not support tools"); like Ocadox, it then gets the tools described in its
# instructions and writes each call as a JSON block, so the score reflects what it does in the app.

TEXT_TOOLS_INTRO = (
    "\n\nTOOLS. This model calls tools by writing them out. To use one, reply with a JSON object in a "
    "```tool block, like this:\n\n```tool\n{\"name\": \"fs_read_file\", \"arguments\": {\"path\": \"src/main.py\"}}\n```\n\n"
    "Use the exact tool names below and put every argument inside \"arguments\". When the task is done, "
    "reply normally, with no tool block.\n\nThe tools (arguments marked * are required):\n")


def no_tools_error(message: str) -> bool:
    m = message.lower()
    return "does not support tools" in m or "doesn't support tools" in m or "tools are not supported" in m


def text_tools_prompt(tools: list[dict]) -> str:
    lines = []
    for t in tools:
        f = t.get("function", {})
        req = set(f.get("parameters", {}).get("required", []))
        params = ", ".join(f"{k}{'*' if k in req else ''}: {v.get('type', 'any')}"
                           for k, v in f.get("parameters", {}).get("properties", {}).items())
        desc = " ".join(f.get("description", "").split()).split(". ")[0][:200]
        lines.append(f"- {f.get('name')}({params}): {desc}")
    return TEXT_TOOLS_INTRO + "\n".join(lines) + "\n"


def as_text_mode(messages: list[dict], tools: list[dict]) -> list[dict]:
    """Messages for a model without tool calling: tools in the system prompt, results as user turns,
    earlier calls written out as tool blocks."""
    out = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            out.append({"role": "system", "content": m.get("content", "") + text_tools_prompt(tools)})
        elif role == "tool":
            out.append({"role": "user", "content": "[Tool result] " + m.get("content", "")})
        elif role == "assistant" and m.get("tool_calls"):
            calls = "\n".join("```tool\n" + json.dumps({"name": c["function"]["name"], "arguments": c["function"].get("arguments", {})})
                              + "\n```" for c in m["tool_calls"])
            out.append({"role": "assistant", "content": (m.get("content") or "") + ("\n" if m.get("content") else "") + calls})
        else:
            out.append(m)
    if not any(m.get("role") == "system" for m in out):
        out.insert(0, {"role": "system", "content": text_tools_prompt(tools)})
    return out


def _json_objects(text: str) -> list[str]:
    """Brace-balanced JSON objects in free text (string-aware)."""
    out, depth, start, in_str, esc = [], 0, 0, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                out.append(text[start:i + 1])
    return out


def parse_text_tool_calls(text: str, known: list[str] | None = None) -> list[dict]:
    """Tool calls a model wrote as text, read the way an agent app reads them (Ocadox's agent.rs):
    JSON {"name": ..., "arguments": {...}} (or "parameters"); the same written loosely (single quotes,
    trailing commas, True/None); and, given the offered tool names, Llama 3.1's <function=name>{...}</function>
    and Python-style name(key=value, ...) as Gemma and Llama 3.2 are trained to write."""
    if known and "<function=" in text:  # tag-style first: a value can hold JSON that isn't a call
        tagged = _function_tags(text, known)
        if tagged:
            return tagged
    objects = _json_objects(text)
    for loads in (json.loads, _loose):
        calls = []
        for raw in objects:
            try:
                v = loads(raw)
            except (ValueError, SyntaxError):
                continue
            if not isinstance(v, dict):
                continue
            fn = v.get("function") if isinstance(v.get("function"), dict) else v
            name = v.get("function") if isinstance(v.get("function"), str) else fn.get("name")
            if isinstance(name, str) and name and (known is None or loads is json.loads or name in known):
                args = fn.get("arguments", fn.get("parameters", v.get("arguments", v.get("parameters", {}))))
                calls.append({"name": name, "arguments": _as_args(args)})
        if calls:
            return calls
    if not known:
        return []
    return _function_tags(text, known) or _python_style(text, known)


def _loose(raw: str):
    """A Python literal, with JSON's true/false/null allowed too."""
    import ast
    tree = ast.parse(raw.strip(), mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in ("true", "false", "null"):
            node.__class__, node.value = ast.Constant, {"true": True, "false": False, "null": None}[node.id]
    return ast.literal_eval(tree)


def _function_tags(text: str, known: list[str]) -> list[dict]:
    calls = []
    for m in re.finditer(r"<function=\s*\"?([\w.-]+)\"?\s*>(.*?)(?=</function>|<function=|$)", text, re.S):
        if m.group(1) in known:
            body = m.group(2).strip()
            if "<parameter=" in body:  # Qwen 3.5+/Qwen3-Coder: <parameter=key>\nvalue\n</parameter>
                args = {k.strip("\"' "): re.sub(r"^\r?\n|\r?\n$", "", v)
                        for k, v in re.findall(r"<parameter=([^>]+)>(.*?)(?=</parameter>|<parameter=|$)", m.group(2), re.S)}
                calls.append({"name": m.group(1), "arguments": args})
                continue
            try:
                args = json.loads(body) if body else {}
            except ValueError:
                try:
                    args = _loose(body)
                except (ValueError, SyntaxError):
                    args = {}
            calls.append({"name": m.group(1), "arguments": _as_args(args)})
    return calls


_NAMESPACES = ("default_api.", "functions.", "function.", "tools.", "tool.", "api.")


def _python_style(text: str, known: list[str]) -> list[dict]:
    """name(key=value, ...) with an offered tool's exact name and keyword arguments only, not after
    `def` and not as someone's method, so code in an answer isn't taken for a call."""
    import ast
    calls = []
    for m in re.finditer(r"(?<![\w.])((?:%s)?)([A-Za-z_]\w*)\(" % "|".join(re.escape(n) for n in _NAMESPACES), text):
        name = m.group(2)
        if name not in known or text[:m.start()].endswith("def "):
            continue
        call = None
        for close in (i for i, ch in enumerate(text[m.end():], m.end()) if ch == ")"):
            try:  # the shortest stretch up to a ")" that parses as one call
                node = ast.parse(name + text[m.end() - 1:close + 1], mode="eval").body
            except SyntaxError:
                continue
            call = node if isinstance(node, ast.Call) else None
            break
        if call is None or call.args:
            continue
        try:
            args = {k.arg: _loose(ast.unparse(k.value)) for k in call.keywords if k.arg}
        except (ValueError, SyntaxError):
            continue
        calls.append({"name": name, "arguments": args})
    return calls


def _as_args(args) -> dict:
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return {}
    return args if isinstance(args, dict) else {}


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
                 num_ctx: int | None = None, seed: int = 0, tools: list[dict] | None = None) -> GenResult:
        """With `tools`, the result's extra has `tool_calls` ([{"name", "arguments"}]) and `text_tools`."""
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
        self._text_tools = False  # set when the model refuses native tools (see TEXT_TOOLS_INTRO)

    def _post(self, path, payload, stream=False, timeout=3600):
        r = requests.post(f"{url()}{path}", json=payload, stream=stream, timeout=timeout)
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

    def generate(self, messages, max_tokens=256, temperature=0.0, num_ctx=None, seed=0, tools=None) -> GenResult:
        opts = {"num_predict": max_tokens, "temperature": temperature, "seed": seed}
        num_ctx = num_ctx or self.num_ctx
        if num_ctx:
            opts["num_ctx"] = num_ctx
        text_tools = bool(tools) and self._text_tools
        sent = as_text_mode(messages, tools) if text_tools else messages
        payload = {"model": self.model, "messages": sent, "stream": True,
                   "options": opts, "keep_alive": self.keep_alive}
        if tools and not text_tools:
            payload["tools"] = tools
        if self._send_think:
            payload["think"] = self.think
        t0 = time.perf_counter()
        try:
            r = self._post("/api/chat", payload, stream=True)
        except RuntimeError as e:
            if tools and not text_tools and no_tools_error(str(e)):
                self._text_tools = True  # remembered for this model's remaining items
                return self.generate(messages, max_tokens, temperature, num_ctx, seed, tools)
            if "think" in str(e).lower() and self._send_think:
                self._send_think = False
                return self.generate(messages, max_tokens, temperature, num_ctx, seed, tools)
            raise
        ttft = None
        text, thinking, final, calls = [], [], {}, []
        for line in r.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            msg = chunk.get("message", {})
            if ttft is None and (msg.get("content") or msg.get("thinking")):
                ttft = time.perf_counter() - t0
            text.append(msg.get("content", ""))
            thinking.append(msg.get("thinking", "") or "")
            for c in msg.get("tool_calls") or []:
                fn = c.get("function", {})
                calls.append({"name": fn.get("name", ""), "arguments": _as_args(fn.get("arguments", {}))})
            if chunk.get("done"):
                final = chunk
        total = time.perf_counter() - t0
        raw = "".join(text)
        extra = {"done_reason": final.get("done_reason")}
        if tools:
            if not calls:  # text mode, or a model that wrote its call as text anyway
                calls = parse_text_tool_calls(strip_thinking(raw), [t.get("function", {}).get("name") for t in tools])
            extra.update(tool_calls=calls, text_tools=text_tools)
        return GenResult(
            text=strip_thinking(raw),
            thinking="".join(thinking),
            prompt_tokens=final.get("prompt_eval_count", 0),
            output_tokens=final.get("eval_count", 0),
            ttft_s=ttft if ttft is not None else total,
            total_s=total,
            prompt_eval_s=(final.get("prompt_eval_duration") or 0) / 1e9 or None,
            gen_s=(final.get("eval_duration") or 0) / 1e9 or None,
            extra=extra,
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
        return requests.get(f"{url()}/api/version", timeout=3).ok
    except requests.RequestException:
        return False


def ensure_ollama(wait_s: float = 90, log=print) -> bool:
    """Make sure the Ollama server is running; start the desktop app (or `ollama serve`) if not.
    While LocalBench's private benchmark server is in use, that server is restarted instead."""
    if ollama_up():
        return True
    from .server import PRIVATE
    if PRIVATE.active:
        return PRIVATE.restart(log=log)
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


PULL_STALL_S = 60  # no new bytes for this long = a stalled piece; re-requesting gets it a fresh connection
PULL_ATTEMPTS = 5
PULL_PROGRESS_S = 15  # print download progress at least this often


def pull(tag: str, log=print, base: str | None = None) -> None:
    """Pull a model from the Ollama library, printing sparse progress. `base` overrides the server.
    A stalled or dropped download is retried; Ollama resumes from the parts it already has."""
    for attempt in range(1, PULL_ATTEMPTS + 1):
        try:
            if _pull_once(tag, log, base):
                return
            why = "the download ended before it finished"
        except (requests.exceptions.RequestException, RuntimeError, ValueError) as e:
            why = str(e)[:200] or type(e).__name__
        if attempt == PULL_ATTEMPTS:
            raise RuntimeError(f"download of {tag} failed {PULL_ATTEMPTS} times: {why}")
        log(f"    download interrupted ({why}) - resuming, attempt {attempt + 1}/{PULL_ATTEMPTS}")
        time.sleep(5 * attempt)


def _pull_once(tag: str, log, base: str | None) -> bool:
    """One pull request. True once Ollama reports success; raises on an error or a stall."""
    r = requests.post(f"{base or url()}/api/pull", json={"model": tag, "stream": True}, stream=True,
                      timeout=(10, PULL_STALL_S))  # read timeout: a silent stream raises instead of hanging
    r.raise_for_status()
    last_status = None
    done, moved = {}, time.monotonic()  # bytes per layer; when the total last grew
    shown_at, shown_bytes = time.monotonic(), None  # last progress line, so slow downloads never look frozen
    for line in r.iter_lines():
        if not line:
            continue
        d = json.loads(line)
        if d.get("error"):
            raise RuntimeError(d["error"])
        st = d.get("status", "")
        if d.get("completed") is not None:
            if d["completed"] > done.get(d.get("digest", st), -1):
                done[d.get("digest", st)] = d["completed"]
                moved = time.monotonic()
            elif time.monotonic() - moved > PULL_STALL_S:
                r.close()
                raise RuntimeError(f"no progress for {PULL_STALL_S} s")
        if d.get("total") and d.get("completed") is not None:
            now = time.monotonic()
            if shown_bytes is None or now - shown_at >= PULL_PROGRESS_S or d["completed"] >= d["total"]:
                gb, total_gb = d["completed"] / 1e9, d["total"] / 1e9
                line_ = f"    downloading {gb:.2f} / {total_gb:.2f} GB ({100 * d['completed'] / d['total']:.0f}%)"
                if shown_bytes is not None and now > shown_at:
                    mbps = max(d["completed"] - shown_bytes, 0) / 1e6 / (now - shown_at)
                    left = (d["total"] - d["completed"]) / 1e6 / mbps if mbps > 0.05 else None
                    if d["completed"] < d["total"]:
                        line_ += f"  {mbps:.1f} MB/s" + (f", ~{left / 60:.0f} min left" if left else ", no progress")
                if d["completed"] < d["total"] or shown_bytes is None or shown_bytes < d["total"]:
                    log(line_)
                shown_at, shown_bytes = now, d["completed"]
        elif st != last_status and not st.startswith("pulling "):
            log(f"    {st}")
        last_status = st
    return last_status == "success"


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
        return requests.get(f"{url()}/api/tags", timeout=10).json().get("models", [])
    except Exception:
        return []


def ollama_ps() -> list[dict]:
    try:
        return requests.get(f"{url()}/api/ps", timeout=10).json().get("models", [])
    except Exception:
        return []


def ollama_version() -> str | None:
    try:
        return requests.get(f"{url()}/api/version", timeout=5).json().get("version")
    except Exception:
        return None
