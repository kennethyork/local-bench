"""Run model-written Python against tests in a separate, isolated interpreter.

Not a security boundary: code runs as you, but in a throwaway temp dir, with `python -I`
(no user site-packages, no cwd on sys.path), a scrubbed environment and a hard timeout.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

_ENV_KEEP = ("SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "TEMP", "TMP", "PATH", "COMSPEC")


def run_python(code: str, timeout: float = 10.0) -> tuple[bool, str]:
    """Returns (passed, short_error). Passed means exit code 0 within the timeout."""
    env = {k: os.environ[k] for k in _ENV_KEEP if k in os.environ}
    env["PYTHONIOENCODING"] = "utf-8"
    with tempfile.TemporaryDirectory(prefix="lb_") as d:
        path = os.path.join(d, "prog.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            p = subprocess.run([sys.executable, "-I", path], cwd=d, env=env, capture_output=True,
                               text=True, encoding="utf-8", errors="replace", timeout=timeout,
                               stdin=subprocess.DEVNULL,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except subprocess.TimeoutExpired:
            return False, "timeout"
    if p.returncode == 0:
        return True, ""
    err = (p.stderr or p.stdout).strip().splitlines()
    return False, (err[-1] if err else f"exit {p.returncode}")[:300]
