"""Quiet-mode governor, ported from Ocadox (src-tauri/src/governor.rs + power.rs).

Same profiles and the same maths: a tokens/sec budget, a temperature ceiling with an
approach band, and a whole-machine power budget, where the stricter pressure wins.

How it applies to benchmarking:
  * Hardware power cap (the lever that actually quiets the card — Ollama generates ahead
    at full power no matter how slowly a client reads). Applied for the whole run via NVML,
    restored afterwards. Needs an administrator shell on Windows; otherwise it degrades to
    "unavailable" and the run continues uncapped, exactly like Ocadox.
  * Pacing as a duty cycle between requests: after each item the governor idles until
    cumulative output tokens / elapsed time is back inside the (thermally/power-adjusted)
    budget, so the GPU spends real time cool instead of pinned.
  * Thermal back-off: before each request, if the GPU is past the ceiling, wait for it to
    come back into the approach band (bounded, so it can never look like a hang).
  * The speed test is never paced (that would corrupt the latency numbers) — it only runs
    under the profile's power cap, which is exactly the speed/noise trade-off worth measuring.
"""
from __future__ import annotations

import atexit
import threading
import time
from dataclasses import dataclass, replace

from . import gpu

try:
    import pynvml
except Exception:  # pragma: no cover
    pynvml = None

APPROACH_BAND = 5
FAN_BAND = 10  # fan %: start easing off this far below the fan ceiling
FLOOR = 0.15


@dataclass(frozen=True)
class PowerBudget:
    total_watts: float
    non_gpu_watts: float

    def gpu_allowance(self) -> float:
        # A budget below the non-GPU estimate would hand the GPU zero watts.
        return max(self.total_watts - self.non_gpu_watts, 50.0)


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    tokens_per_second: float | None  # None = no fixed-rate pacing
    temperature_ceiling_c: int | None
    keep_alive_secs: int | None
    power_budget: PowerBudget | None
    adaptive: bool = False  # full speed; rest only in proportion to thermal/power pressure
    fan_ceiling_pct: int | None = None  # keep the GPU fans at or below this speed (noise, not heat)
    # The fan watchdog may raise the power cap up to this (clamped to the card's maximum) while fans are quiet.
    boost_cap_watts: float | None = None
    never_waits: bool = False  # no rests or waits at all; noise is handled only by the fan watchdog's power cap

    @property
    def gpu_power_cap_watts(self) -> float | None:
        return self.power_budget.gpu_allowance() if self.power_budget else None


PROFILES = {
    # LocalBench default: as fast as it can go without loud fans. Never rests or waits; it starts at a 170 W
    # power cap and a fan watchdog moves the cap (100 W .. the card's maximum) to hold the fans at 55%.
    # Only the 80C hard stop (every profile) can pause it. Power caps need an administrator terminal.
    "quiet": ProfileSpec("quiet", None, None, 300, PowerBudget(320.0, 150.0), adaptive=True, fan_ceiling_pct=55,
                         boost_cap_watts=450.0, never_waits=True),
    # full speed while cool, backs off only under heat/power pressure
    "adaptive": ProfileSpec("adaptive", None, 75, 300, PowerBudget(450.0, 150.0), adaptive=True),
    # the Ocadox presets, unchanged
    "silent": ProfileSpec("silent", 18.0, 65, 120, PowerBudget(300.0, 150.0)),
    "balanced": ProfileSpec("balanced", 45.0, 75, 300, PowerBudget(450.0, 150.0)),
    "turbo": ProfileSpec("turbo", None, None, 600, None),
}
DEFAULT_PROFILE = "quiet"


def rest_for_load(work_s: float, load: float) -> float:
    """Adaptive duty cycle: after `work_s` seconds of generation at permitted `load` (0..1],
    rest so that work / (work + rest) == load. Full load -> no rest; 0.5 -> rest as long as it worked."""
    if load >= 1.0 or work_s <= 0:
        return 0.0
    return work_s * (1.0 / max(load, FLOOR) - 1.0)


def load_for_temperature(current_c: int, ceiling_c: int) -> float:
    """1.0 well below the ceiling, eases to 0.5 at it, keeps hardening past it — never 0."""
    if current_c + APPROACH_BAND < ceiling_c:
        return 1.0
    if current_c <= ceiling_c:
        band_start = max(ceiling_c - APPROACH_BAND, 0)
        progress = (current_c - band_start) / APPROACH_BAND
        return min(max(1.0 - 0.5 * progress, 0.5), 1.0)
    past = current_c - ceiling_c
    return min(max(0.5 / (1.0 + past / 4.0), FLOOR), 0.5)


def load_for_fan(current_pct: int, ceiling_pct: int) -> float:
    """Like load_for_temperature, on fan speed: 1.0 well below the ceiling, 0.5 at it, harder past it."""
    if current_pct + FAN_BAND < ceiling_pct:
        return 1.0
    if current_pct <= ceiling_pct:
        return min(max(1.0 - 0.5 * (current_pct - (ceiling_pct - FAN_BAND)) / FAN_BAND, 0.5), 1.0)
    return min(max(0.5 / (1.0 + (current_pct - ceiling_pct) / 8.0), FLOOR), 0.5)


def load_for_power(gpu_watts: float, allowance_watts: float) -> float:
    if gpu_watts <= allowance_watts or gpu_watts <= 0.0:
        return 1.0
    return min(max(allowance_watts / gpu_watts, FLOOR), 1.0)


class Pacer:
    def __init__(self, spec: ProfileSpec):
        self.spec = spec
        self.started = time.perf_counter()
        self.emitted = 0
        self.thermal_load = 1.0
        self.power_load = 1.0
        self.fan_load = 1.0

    def load(self) -> float:
        return min(self.thermal_load, self.power_load, self.fan_load)

    def effective_rate(self) -> float | None:
        if self.spec.tokens_per_second is None:
            return None
        return max(self.spec.tokens_per_second * self.load(), 0.05)

    def observe_temperature(self, current_c: int | None):
        if self.spec.temperature_ceiling_c is None:
            self.thermal_load = 1.0
        elif current_c is not None:  # a missing sensor keeps the last known load
            self.thermal_load = load_for_temperature(current_c, self.spec.temperature_ceiling_c)

    def observe_fan(self, pct: int | None):
        if self.spec.fan_ceiling_pct is None:
            self.fan_load = 1.0
        elif pct is not None:
            self.fan_load = load_for_fan(pct, self.spec.fan_ceiling_pct)

    def observe_power(self, gpu_watts: float | None):
        if self.spec.power_budget is None:
            self.power_load = 1.0
        elif gpu_watts is not None:
            self.power_load = load_for_power(gpu_watts, self.spec.power_budget.gpu_allowance())

    def ideal_elapsed(self) -> float:
        rate = self.effective_rate()
        return 0.0 if rate is None else self.emitted / rate

    def delay_after_tokens(self, n: int) -> float:
        self.emitted += n
        return max(self.ideal_elapsed() - (time.perf_counter() - self.started), 0.0)


# --------------------------------------------------------------------------- hardware cap

def clamp_cap_mw(requested_watts: float, min_mw: int, max_mw: int) -> int:
    requested = max(int(round(requested_watts * 1000)), 0)
    return min(max(requested, min_mw), max(max_mw, min_mw))


class PowerCap:
    """Context manager: cap GPU 0 at `watts` for the duration, restore the original limit after.
    Degrades to outcome='unavailable: …' without admin rights instead of failing the run."""

    _lock = threading.Lock()

    def __init__(self, watts: float | None, index: int = 0):
        self.watts, self.index = watts, index
        self.outcome = "not requested"
        self.applied_watts: float | None = None
        self.current_watts: float | None = None  # what's set right now (the fan watchdog may lower it)
        self.min_watts: float | None = None
        self._original_mw: int | None = None

    def __enter__(self):
        if self.watts is None:
            return self
        if pynvml is None or not gpu.available():
            self.outcome = "unavailable: no NVML"
            return self
        with self._lock:
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(self.index)
                original = pynvml.nvmlDeviceGetPowerManagementLimit(h)
                lo, hi = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(h)
                target = clamp_cap_mw(self.watts, lo, hi)
                pynvml.nvmlDeviceSetPowerManagementLimit(h, target)
                self._original_mw = original
                self.applied_watts = self.current_watts = target / 1000
                self.min_watts = lo / 1000
                self.outcome = f"applied {self.applied_watts:.0f} W (was {original / 1000:.0f} W)"
                atexit.register(self._restore)  # survive Ctrl-C / crashes
            except pynvml.NVMLError as e:
                msg = str(e)
                if "Permission" in msg:
                    msg = "needs an administrator terminal to set a GPU power limit"
                self.outcome = f"unavailable: {msg}"
        return self

    def __exit__(self, *exc):
        self._restore()

    def set(self, watts: float) -> float | None:
        """Change the held cap (only while one is applied). Returns the watts now set, or None."""
        with self._lock:
            if self._original_mw is None:
                return None
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(self.index)
                lo, hi = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(h)
                target = clamp_cap_mw(watts, lo, hi)
                pynvml.nvmlDeviceSetPowerManagementLimit(h, target)
                self.current_watts = target / 1000
            except Exception:
                return None
            return self.current_watts

    def _restore(self):
        with self._lock:
            if self._original_mw is None:
                return
            try:
                pynvml.nvmlDeviceSetPowerManagementLimit(pynvml.nvmlDeviceGetHandleByIndex(self.index),
                                                         self._original_mw)
            except Exception as e:  # pragma: no cover
                print(f"WARNING: failed to restore GPU power limit: {e}")
            self._original_mw = None


def cap_status(index: int = 0) -> dict:
    """Board limits and whether this process may change them (probe = re-set current value)."""
    if pynvml is None or not gpu.available():
        return {"can_set": False}
    h = pynvml.nvmlDeviceGetHandleByIndex(index)
    lo, hi = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(h)
    cur = pynvml.nvmlDeviceGetPowerManagementLimit(h)
    try:
        pynvml.nvmlDeviceSetPowerManagementLimit(h, cur)
        can = True
    except Exception:
        can = False
    return {"min_watts": lo / 1000, "max_watts": hi / 1000, "current_watts": cur / 1000,
            "default_watts": pynvml.nvmlDeviceGetPowerManagementDefaultLimit(h) / 1000, "can_set": can}


class FanWatch:
    """Closed loop on fan speed while a power cap is held. Rests between questions can't help during a
    long answer, so every couple of seconds: fans over the ceiling -> lower the GPU power limit a step
    (never below the card's minimum); fans back under ceiling - FAN_BAND -> raise it again, slowly,
    towards the profile's cap. Fans react to heat with a lag, hence the spaced-out steps."""

    STEP_DOWN_W, DOWN_EVERY_S = 15.0, 6.0
    STEP_UP_W, UP_EVERY_S = 10.0, 20.0

    def __init__(self, cap: PowerCap, ceiling_pct: int, base_watts: float, read_fan=None, log=print,
                 clock=time.monotonic, interval: float = 2.0):
        self.cap, self.ceiling, self.base = cap, ceiling_pct, base_watts
        self.read_fan = read_fan or gpu.fan_percent
        self.log, self.clock, self.interval = log, clock, interval
        self.lowest_watts: float | None = None
        self.steps_down = 0
        self._last_change = -1e9
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def step(self):
        f, cur, now = self.read_fan(), self.cap.current_watts, self.clock()
        if f is None or cur is None:
            return
        floor = self.cap.min_watts or 100.0
        if f > self.ceiling and cur > floor and now - self._last_change >= self.DOWN_EVERY_S:
            new = self.cap.set(max(floor, cur - self.STEP_DOWN_W))
            if new is not None:
                self._last_change, self.steps_down = now, self.steps_down + 1
                self.lowest_watts = min(self.lowest_watts or new, new)
                self.log(f"                     fans {f}% (limit {self.ceiling}%) - GPU power limit {cur:.0f} -> {new:.0f} W")
        elif f <= self.ceiling - FAN_BAND and cur < self.base and now - self._last_change >= self.UP_EVERY_S:
            new = self.cap.set(min(self.base, cur + self.STEP_UP_W))
            if new is not None:
                self._last_change = now
                self.log(f"                     fans {f}% - GPU power limit back up to {new:.0f} W")

    def _run(self):
        while not self._stop.wait(self.interval):
            try:
                self.step()
            except Exception:
                pass

    def start(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def stop(self):
        self._stop.set()
        if self._t:
            self._t.join()


# --------------------------------------------------------------------------- run-level governor

class Governor:
    """Drives a Pacer from live readings between requests and keeps the quiet-mode scoreboard."""

    MAX_COOLDOWN_S = 60.0
    MAX_FAN_WAIT_S = 120.0
    SAFETY_COOLDOWN_S = 180.0

    def __init__(self, profile: str = "turbo", pace: bool = True, max_temp_c: int | None = 80):
        self.spec = PROFILES[profile]
        self.pace = pace
        self.max_temp_c = max_temp_c  # hard stop that applies in EVERY profile, turbo included
        # what pacing acts on: nothing at all for never_waits profiles (their power budget is only the start cap)
        self.limits = replace(self.spec, power_budget=None, fan_ceiling_pct=None) if self.spec.never_waits             else self.spec
        self.pacer = Pacer(self.limits)
        self.cap = PowerCap(self.spec.gpu_power_cap_watts)
        self.fan_watch: FanWatch | None = None
        self.paced_s = 0.0
        self.cooldown_s = 0.0
        self.throttle_events = 0
        self.safety_pauses = 0

    @property
    def name(self) -> str:
        return self.spec.name

    def __enter__(self):
        self.cap.__enter__()
        self.pacer = Pacer(self.limits)
        if self.spec.fan_ceiling_pct is not None and self.cap.applied_watts is not None:
            self.fan_watch = FanWatch(self.cap, self.spec.fan_ceiling_pct,
                                      max(self.cap.applied_watts, self.spec.boost_cap_watts or 0)).start()
        return self

    def __exit__(self, *exc):
        if self.fan_watch:
            self.fan_watch.stop()
        self.cap.__exit__(*exc)

    def _observe(self) -> tuple[int | None, float | None]:
        t, w = gpu.temperature_c(), gpu.power_w()
        self.pacer.observe_temperature(t)
        self.pacer.observe_power(w)
        self.pacer.observe_fan(gpu.fan_percent())
        return t, w

    def _safety_stop(self):
        """Profile-independent guard: at max_temp, pause until 10C below it; abort if it won't cool."""
        t = gpu.temperature_c()
        if self.max_temp_c is None or t is None or t < self.max_temp_c:
            return
        print(f"  SAFETY: GPU at {t}C (limit {self.max_temp_c}C) - pausing to cool")
        t0 = time.perf_counter()
        while t is not None and t > self.max_temp_c - 10:
            if time.perf_counter() - t0 > self.SAFETY_COOLDOWN_S:
                raise RuntimeError(f"GPU stayed above {self.max_temp_c - 10}C for "
                                   f"{self.SAFETY_COOLDOWN_S:.0f}s - aborting this model to protect the card")
            time.sleep(2.0)
            t = gpu.temperature_c()
        self.safety_pauses += 1
        self.cooldown_s += time.perf_counter() - t0

    def _fans_too_loud(self, f: int | None) -> bool:
        """Fans past the ceiling, and (if a watchdog runs) it has already cut power as far as it can."""
        ceiling = self.spec.fan_ceiling_pct
        if self.spec.never_waits or ceiling is None or f is None or f <= ceiling:
            return False
        if self.fan_watch and (self.cap.current_watts or 0) > (self.cap.min_watts or 100.0) + 0.5:
            return False  # the watchdog is still lowering the power limit; let it work
        return True

    def _wait_for_fans(self):
        """Quiet profiles: if the fans are past their ceiling, let them spin down before the next request."""
        ceiling = self.spec.fan_ceiling_pct
        f = gpu.fan_percent()
        if not self._fans_too_loud(f):
            return
        print(f"                     fans at {f}% (quiet limit {ceiling}%) - waiting for them to slow down")
        self.throttle_events += 1
        t0 = time.perf_counter()
        while f is not None and f > ceiling - FAN_BAND and time.perf_counter() - t0 < self.MAX_FAN_WAIT_S:
            time.sleep(2.0)
            f = gpu.fan_percent()
        self.cooldown_s += time.perf_counter() - t0

    def continuous(self) -> bool:
        """Can work run back-to-back until needs_rest() says otherwise? (Not for fixed-rate pacing.)"""
        return not self.pace or self.spec.adaptive or self.spec.tokens_per_second is None

    def needs_rest(self) -> bool:
        """Any pressure that before_request/after_request would act on? Reads sensors, never sleeps."""
        t, _ = self._observe()
        if t is not None and self.max_temp_c is not None and t >= self.max_temp_c:
            return True
        if self._fans_too_loud(gpu.fan_percent()):
            return True
        if t is not None and self.limits.temperature_ceiling_c is not None and t > self.limits.temperature_ceiling_c:
            return True
        return self.pace and self.spec.adaptive and self.pacer.load() < 1.0

    def before_request(self):
        """Hard safety stop, then profile thermal back-off: past the ceiling, wait (bounded) for the band."""
        self._safety_stop()
        self._wait_for_fans()
        ceiling = self.limits.temperature_ceiling_c
        if ceiling is None:
            return
        t, _ = self._observe()
        if t is None or t <= ceiling:
            return
        self.throttle_events += 1
        t0 = time.perf_counter()
        while t is not None and t > ceiling - APPROACH_BAND and time.perf_counter() - t0 < self.MAX_COOLDOWN_S:
            time.sleep(1.0)
            t, _ = self._observe()
        self.cooldown_s += time.perf_counter() - t0

    def after_request(self, output_tokens: int, work_s: float = 0.0, on_rest=None) -> float:
        """Duty-cycle pacing after a request. Fixed-rate profiles idle until cumulative throughput is
        back inside their budget; adaptive rests only in proportion to thermal/power pressure.
        Calls on_rest(seconds) before a noticeable pause so the console doesn't look frozen."""
        self._observe()
        if not self.pace:
            return 0.0
        if self.spec.adaptive:
            delay = rest_for_load(work_s, self.pacer.load())
        else:
            delay = self.pacer.delay_after_tokens(output_tokens)
        if delay > 0:
            if self.pacer.load() < 1.0:
                self.throttle_events += 1
            if on_rest and delay >= 2:
                on_rest(delay)
            time.sleep(delay)
            self.paced_s += delay
        return delay

    def summary(self) -> dict:
        return {"profile": self.spec.name, "target_tps": self.spec.tokens_per_second,
                "temp_ceiling_c": self.spec.temperature_ceiling_c, "fan_ceiling_pct": self.spec.fan_ceiling_pct,
                "gpu_cap_requested_w": self.spec.gpu_power_cap_watts, "cap": self.cap.outcome,
                "cap_applied_w": self.cap.applied_watts,
                "fan_watch_lowest_w": self.fan_watch.lowest_watts if self.fan_watch else None, "paced_s": round(self.paced_s, 1),
                "cooldown_s": round(self.cooldown_s, 1), "throttle_events": self.throttle_events,
                "max_temp_c": self.max_temp_c, "safety_pauses": self.safety_pauses}
