"""run_category keeps every slot busy, keeps records in question order, and rests with nothing in flight."""
import random
import threading
import time
import unittest

from localbench.backends import GenResult
from localbench.categories import Category, Grade, Item, run_category


class FakeBackend:
    kind, think, label = "fake", False, "fake:model"

    def __init__(self):
        self.lock = threading.Lock()
        self.inflight = self.max_inflight = 0
        self.inflight_during_rest = 0

    def generate(self, messages, max_tokens=256, temperature=0.0, num_ctx=None, seed=0):
        with self.lock:
            self.inflight += 1
            self.max_inflight = max(self.max_inflight, self.inflight)
        time.sleep(random.uniform(0.005, 0.03))
        with self.lock:
            self.inflight -= 1
        return GenResult(messages[-1]["content"], 5, 7, 0.001, 0.02)


class Echo(Category):
    name, title = "echo", "Echo"

    def load(self):
        return [Item(f"q{i}", [{"role": "user", "content": str(i)}]) for i in range(23)]

    def grade(self, item, text):
        return Grade(1.0 if text == item.id[1:] else 0.0)


class FakeGovernor:
    name = "fake"

    def __init__(self, backend, continuous=True, rest_every=5):
        self.backend, self._continuous, self.rest_every = backend, continuous, rest_every
        self.checks = self.rests = 0

    def continuous(self):
        return self._continuous

    def needs_rest(self):
        self.checks += 1
        return self.checks % self.rest_every == 0

    def before_request(self):
        pass

    def after_request(self, tokens, work_s, on_rest=None):
        self.rests += 1
        self.backend.inflight_during_rest = max(self.backend.inflight_during_rest, self.backend.inflight)
        return 0.0


class Runner(unittest.TestCase):
    def run_it(self, parallel, governor=None, backend=None):
        b = backend or FakeBackend()
        r = run_category(b, Echo(), None, log=lambda *_: None, governor=governor, parallel=parallel)
        return b, r

    def test_records_in_order_and_all_correct(self):
        for parallel in (1, 3, 8):
            b, r = self.run_it(parallel)
            self.assertEqual([x["id"] for x in r["records"]], [it.id for it in Echo().items(None)])
            self.assertEqual(r["score"], 100.0)
            self.assertLessEqual(b.max_inflight, parallel)

    def test_slots_are_kept_full(self):
        b, _ = self.run_it(4)
        self.assertEqual(b.max_inflight, 4)

    def test_rests_only_with_nothing_in_flight(self):
        b = FakeBackend()
        gov = FakeGovernor(b, rest_every=3)
        _, r = self.run_it(4, gov, b)
        self.assertGreater(gov.rests, 1)
        self.assertEqual(b.inflight_during_rest, 0)
        self.assertEqual(len(r["records"]), 23)

    def test_fixed_rate_profiles_rest_every_batch(self):
        b = FakeBackend()
        gov = FakeGovernor(b, continuous=False, rest_every=10**9)
        self.run_it(4, gov, b)
        self.assertGreaterEqual(gov.rests, 23 // 4)  # paced after every 4 answers, plus once at the end
        self.assertEqual(b.inflight_during_rest, 0)


if __name__ == "__main__":
    unittest.main()
