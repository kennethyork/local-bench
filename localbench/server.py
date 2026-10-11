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


def other_run_active() -> bool:
    """Is another LocalBench run going? Any other Python process from this project's .venv counts;
    one we can't inspect (e.g. an administrator run) is assumed to be one, to stay on the safe side."""
    import psutil
    venv = os.path.normcase(os.path.dirname(os.path.dirname(os.path.abspath(__import__("sys").executable))))
    me = os.getpid()
    family = {me, os.getppid()}
    for p in psutil.process_iter(["pid", "name"]):
        if p.info["pid"] in family or "python" not in (p.info["name"] or "").lower():
            continue
        try:
            exe = os.path.normcase(p.exe())
            if exe.startswith(venv):
                cmd = " ".join(p.cmdline())
                if "localbench" in cmd and "unload" not in cmd:
                    return True
        except psutil.AccessDenied:
            return True
        except psutil.NoSuchProcess:
            continue
    return False


def kill_stale_private(log=print) -> bool:
    """Stop a LocalBench private server left behind by a closed window, so it can't hold the port and
    a model. Never touches it while any other LocalBench run is active (that run may be using it)."""
    import psutil
    if other_run_active():
        return False
    try:
        conns = psutil.net_connections(kind="tcp")
    except (psutil.AccessDenied, OSError):
        return False
    for c in conns:
        if c.laddr and c.laddr.port == PORT and c.status == psutil.CONN_LISTEN and c.pid:
            if PRIVATE.proc and PRIVATE.proc.pid == c.pid:
                return False  # ours, in use
            try:
                if "ollama" not in psutil.Process(c.pid).name().lower():
                    return False
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return False
            r = subprocess.run(["taskkill", "/PID", str(c.pid), "/T", "/F"], capture_output=True,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if r.returncode == 0:
                log("Stopped a leftover LocalBench Ollama server from an earlier run.")
                return True
            log("A leftover LocalBench Ollama server is still running, but it couldn't be stopped from here "
                "(it was started as administrator). Run this as administrator, or restart the PC.")
            return False
    return False


def unload_all(log=print) -> int:
    """Unload every model from your normal Ollama and LocalBench's private server (if running)."""
    n = 0
    for base in (OLLAMA_URL, URL):
        try:
            models = requests.get(f"{base}/api/ps", timeout=3).json().get("models", [])
        except requests.RequestException:
            continue
        for m in models:
            try:
                requests.post(f"{base}/api/generate", json={"model": m["name"], "keep_alive": 0}, timeout=60)
                log(f"  unloaded {m['name']}")
                n += 1
            except requests.RequestException:
                pass
    return n


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
            if self._up():  # the port is answering but it isn't ours: never adopt a server we didn't start
                log(f"  port {PORT} is in use by another Ollama server - using your normal Ollama "
                    "(one question at a time) for this run")
                self.active = False
                use_url(OLLAMA_URL)
                return False
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
