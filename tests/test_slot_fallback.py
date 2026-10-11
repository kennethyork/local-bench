"""A model that spills out of VRAM with several questions at once is retried with fewer, not failed."""
import argparse
import unittest
from unittest import mock

from localbench import cli


class FakeBackend:
    label, model, kind = "ollama:m:1b", "m:1b", "ollama"

    def load(self):
        return 0.1

    def unload(self):
        pass

    def info(self):
        return {}


class FakeGov:
    cap = mock.Mock(outcome="not requested")

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *e):
        pass

    def summary(self):
        return {}


class SlotFallback(unittest.TestCase):
    def bench(self, spills):
        """spills: how many loads in a row spill. Returns (slots each load ran with, parallel used, saved?)"""
        a = argparse.Namespace(think=False, num_ctx=8192, parallel=8, vram_headroom=0.92, limit=40, max_tokens=None,
                               governor="quiet", max_temp=80)
        state = {"slots": None, "loads": [], "saved": None}

        def server(slots, a):
            state["slots"] = slots
            return slots

        def check(b):
            state["loads"].append(state["slots"])
            if len(state["loads"]) <= spills:
                raise RuntimeError("does not fit this GPU: 1.0 GB spilled to CPU RAM")

        def run_cat(*args, parallel, **kw):
            return {"title": "x", "score": 1, "ci95": 0, "wins": 1, "n": 1, "elapsed_s": 1, "peak_temp_c": 50,
                    "peak_power_w": 100, "parallel": parallel}

        with mock.patch.object(cli, "_backend", lambda *x, **k: FakeBackend()), \
                mock.patch.object(cli.fit, "parallel_slots", lambda *x, **k: 8), \
                mock.patch.object(cli, "_server", server), \
                mock.patch.object(cli.perf, "check_on_gpu", check), \
                mock.patch.object(cli, "Governor", FakeGov), \
                mock.patch.object(cli, "run_category", run_cat), \
                mock.patch.object(cli, "save", lambda kind, label, d: state.update(saved=d) or mock.Mock(name="p")):
            cli._bench("ollama:m:1b", a, ["codegen"])
        return state["loads"], state["saved"]

    def test_fits_first_time(self):
        loads, saved = self.bench(0)
        self.assertEqual(loads, [8])
        self.assertEqual(saved["settings"]["parallel"], 8)

    def test_halves_until_it_fits(self):
        loads, saved = self.bench(2)
        self.assertEqual(loads, [8, 4, 2])
        self.assertEqual(saved["settings"]["parallel"], 2)

    def test_gives_up_only_at_one_slot(self):
        loads, saved = self.bench(99)
        self.assertEqual(loads, [8, 4, 2, 1])
        self.assertIsNone(saved)  # stopped with no results -> nothing saved, model reported as not fitting


if __name__ == "__main__":
    unittest.main()
