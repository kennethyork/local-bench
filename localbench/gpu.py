"""GPU memory/utilization sampling via NVML (works for any process, incl. Ollama)."""
from __future__ import annotations

import threading
import time

try:
    import pynvml

    pynvml.nvmlInit()
    _NVML = True
except Exception:  # no NVIDIA GPU / driver / nvidia-ml-py
    _NVML = False


def available() -> bool:
    return _NVML


def gpu_info(index: int = 0) -> dict:
    if not _NVML:
        return {}
    h = pynvml.nvmlDeviceGetHandleByIndex(index)
    name = pynvml.nvmlDeviceGetName(h)
    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
    return {
        "name": name.decode() if isinstance(name, bytes) else name,
        "vram_total_mb": mem.total // 2**20,
        "driver": _s(pynvml.nvmlSystemGetDriverVersion()),
    }


def used_mb(index: int = 0) -> float:
    if not _NVML:
        return 0.0
    h = pynvml.nvmlDeviceGetHandleByIndex(index)
    return pynvml.nvmlDeviceGetMemoryInfo(h).used / 2**20


def temperature_c(index: int = 0) -> int | None:
    if not _NVML:
        return None
    try:
        return pynvml.nvmlDeviceGetTemperature(pynvml.nvmlDeviceGetHandleByIndex(index), pynvml.NVML_TEMPERATURE_GPU)
    except Exception:
        return None


def power_w(index: int = 0) -> float | None:
    if not _NVML:
        return None
    try:
        return pynvml.nvmlDeviceGetPowerUsage(pynvml.nvmlDeviceGetHandleByIndex(index)) / 1000
    except Exception:
        return None


def _s(v):
    return v.decode() if isinstance(v, bytes) else v


class GpuSampler:
    """Background thread recording peak VRAM, avg utilization and power while active.

    with GpuSampler() as s:
        ...work...
    s.peak_mb, s.avg_util, s.avg_power_w
    """

    def __init__(self, index: int = 0, interval: float = 0.05):
        self.index = index
        self.interval = interval
        self.peak_mb = 0.0
        self.peak_temp_c: int | None = None
        self.peak_power_w: float | None = None
        self.samples_util: list[float] = []
        self.samples_power: list[float] = []
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def __enter__(self):
        if _NVML:
            self._h = pynvml.nvmlDeviceGetHandleByIndex(self.index)
            self._t = threading.Thread(target=self._run, daemon=True)
            self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._t:
            self._t.join()

    def _run(self):
        while not self._stop.is_set():
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._h)
                self.peak_mb = max(self.peak_mb, mem.used / 2**20)
                self.samples_util.append(pynvml.nvmlDeviceGetUtilizationRates(self._h).gpu)
                w = pynvml.nvmlDeviceGetPowerUsage(self._h) / 1000
                self.samples_power.append(w)
                self.peak_power_w = max(self.peak_power_w or 0.0, w)
                t = pynvml.nvmlDeviceGetTemperature(self._h, pynvml.NVML_TEMPERATURE_GPU)
                self.peak_temp_c = max(self.peak_temp_c or 0, t)
            except Exception:
                pass
            time.sleep(self.interval)

    @property
    def avg_util(self) -> float | None:
        return sum(self.samples_util) / len(self.samples_util) if self.samples_util else None

    @property
    def avg_power_w(self) -> float | None:
        return sum(self.samples_power) / len(self.samples_power) if self.samples_power else None
