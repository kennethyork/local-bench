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


def run_python_output(code: str, timeout: float = 10.0) -> tuple[int | None, str, str]:
    """Run `code`; returns (exit code or None on timeout, stdout, stderr)."""
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
            return None, "", "timeout"
    return p.returncode, p.stdout or "", p.stderr or ""


def run_python(code: str, timeout: float = 10.0) -> tuple[bool, str]:
    """Returns (passed, short_error). Passed means exit code 0 within the timeout."""
    rc, out, err = run_python_output(code, timeout)
    if rc is None:
        return False, "timeout"
    if rc == 0:
        return True, ""
    lines = (err or out).strip().splitlines()
    return False, (lines[-1] if lines else f"exit {rc}")[:300]


def run_programs(files: dict[str, str], commands: list[list[str]], timeout: float = 60.0) -> tuple[int | None, str, str]:
    """Write `files` to a temp dir and run `commands` there in order, stopping at the first failure.

    Returns (exit code of the last command run, or None on timeout, stdout, stderr). Used for the Rust
    and TypeScript tasks: a compiler or Node gets the same stripped environment as the Python sandbox,
    plus what it needs to find its toolchain."""
    env = {k: os.environ[k] for k in _ENV_KEEP + _TOOLCHAIN_KEEP if k in os.environ}
    with tempfile.TemporaryDirectory(prefix="lb_") as d:
        for name, text in files.items():
            with open(os.path.join(d, name), "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
        rc, out, err = 0, "", ""
        for cmd in commands:
            try:
                p = subprocess.run(cmd, cwd=d, env=env, capture_output=True, text=True, encoding="utf-8",
                                   errors="replace", timeout=timeout, stdin=subprocess.DEVNULL,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            except subprocess.TimeoutExpired:
                return None, out, "timeout"
            except OSError as e:  # the tool vanished between the availability check and now
                return 127, out, str(e)
            rc, out, err = p.returncode, p.stdout or "", p.stderr or ""
            if rc != 0:
                break
    return rc, out, err


# rustc and cargo find their toolchain through these (rustup on every OS); node needs none of them.
_TOOLCHAIN_KEEP = ("HOME", "USERPROFILE", "RUSTUP_HOME", "CARGO_HOME", "RUSTUP_TOOLCHAIN", "LANG")
