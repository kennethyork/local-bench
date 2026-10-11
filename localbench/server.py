"""LocalBench's own Ollama server, used only while benchmarking.

Your normal Ollama (port 11434, the desktop app, Ocadox, ...) is left exactly as configured. For a
run, LocalBench starts a second `ollama serve` on port 11435 that shares the same models folder but
can answer several questions at once (OLLAMA_NUM_PARALLEL). The slot count is chosen per model from
what fits in VRAM, so a server restart between models is normal. It's stopped when the run ends.
"""
from __future__ import annotations

import atexit
import os
import shutil
import subprocess
import threading
import time

import requests

from .backends import OLLAMA_URL, use_url
from .util import RESULTS

PORT = 11435
URL = f"http://127.0.0.1:{PORT}"


def _ollama_exe() -> str | None:
    found = shutil.which("ollama")
    if found:
        return found
    cand = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe")
    return cand if os.path.exists(cand) else None


def models_dir() -> str | None:
    """Where your Ollama keeps models: OLLAMA_MODELS from this process or your user settings."""
    if os.environ.get("OLLAMA_MODELS"):
        return os.environ["OLLAMA_MODELS"]
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            return winreg.QueryValueEx(k, "OLLAMA_MODELS")[0]
    except Exception:
        return None


def kill_orphan_runners(log=print) -> float:
    """Stop Ollama model runners whose `ollama serve` parent is gone (e.g. the server was killed).
    Ollama can never reach them again, but they keep holding VRAM. Returns how many were stopped."""
    import psutil
    stopped = 0
    for p in psutil.process_iter(["pid", "name", "ppid", "exe"]):
        try:
            name = (p.info["name"] or "").lower()
            exe = (p.info["exe"] or "").lower()
            if name != "llama-server.exe" or "ollama" not in exe:
                continue
            if psutil.pid_exists(p.info["ppid"]) and "ollama" in psutil.Process(p.info["ppid"]).name().lower():
                continue  # parent server alive: in use
            p.kill()
            stopped += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if stopped:
        log(f"Stopped {stopped} orphaned Ollama runner(s) that were still holding VRAM.")
    return stopped


class PrivateServer:
    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.parallel = 1
        self.active = False
        self._lock = threading.Lock()

    def _up(self) -> bool:
        try:
            return requests.get(f"{URL}/api/version", timeout=2).ok
        except requests.RequestException:
            return False

    def start(self, parallel: int = 1, log=print) -> bool:
        """(Re)start the private server with `parallel` slots and point LocalBench at it."""
        with self._lock:
            if self.active and self.parallel == parallel and self.proc and self.proc.poll() is None and self._up():
                return True
            self._stop_proc()
            exe = _ollama_exe()
            if not exe:
                log("  ollama.exe not found - using your normal Ollama server (one request at a time)")
                self.active = False
                use_url(OLLAMA_URL)
                return False
            env = dict(os.environ)
            env["OLLAMA_HOST"] = f"127.0.0.1:{PORT}"
            env["OLLAMA_NUM_PARALLEL"] = str(parallel)
            env["OLLAMA_MAX_LOADED_MODELS"] = "1"
            md = models_dir()
            if md:
                env["OLLAMA_MODELS"] = md
            RESULTS.mkdir(exist_ok=True)
            logf = open(RESULTS / "ollama-private.log", "ab")
            self.proc = subprocess.Popen([exe, "serve"], env=env, stdin=subprocess.DEVNULL, stdout=logf,
                                         stderr=subprocess.STDOUT,
                                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            for _ in range(60):
                time.sleep(0.5)
                if self._up():
                    self.parallel, self.active = parallel, True
                    use_url(URL)
                    atexit.register(self.stop)
                    return True
                if self.proc.poll() is not None:
                    break
            log("  private Ollama server didn't start - using your normal Ollama (one request at a time)")
            self._stop_proc()
            self.active = False
            use_url(OLLAMA_URL)
            return False

    def restart(self, log=print) -> bool:
        if self._up():
            return True
        log("  restarting LocalBench's Ollama server ...")
        self.active = False
        return self.start(self.parallel, log)

    def _stop_proc(self):
        if self.proc and self.proc.poll() is None:
            try:  # unload first so VRAM is released cleanly
                for m in requests.get(f"{URL}/api/ps", timeout=3).json().get("models", []):
                    requests.post(f"{URL}/api/generate", json={"model": m["name"], "keep_alive": 0}, timeout=30)
            except requests.RequestException:
                pass
            # end the whole tree: `ollama serve` runs models in child runner processes
            subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"], capture_output=True,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def stop(self):
        with self._lock:
            self._stop_proc()
            self.active = False
            use_url(OLLAMA_URL)


PRIVATE = PrivateServer()
