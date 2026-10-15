"""Library runs: your own models are never deleted; downloads are deleted once tested, kept if a run stops."""
import argparse
import unittest
from unittest import mock

from localbench import library


class FakeOllama:
    def __init__(self, installed):
        self.installed = set(installed)
        self.deleted, self.pulled = [], []

    def list(self):
        return [{"name": n, "digest": "d" + n} for n in self.installed]

    def pull(self, tag, log=None):
        self.pulled.append(tag)
        self.installed.add(tag)

    def delete(self, tag, log=None):
        self.deleted.append(tag)
        self.installed.discard(tag)
        return True


class LibraryKeep(unittest.TestCase):
    def run_lib(self, state, installed, queue, bench, done=()):
        o = FakeOllama(installed)
        done = set(done)
        a = argparse.Namespace(vram_headroom=0.92, refresh=False, family=None, max_models=None, dry_run=False,
                               keep=False, redo=False, retry_failed=False, no_prefetch=True, no_speed=True,
                               need_ctx=8192, publish=False)
        q = [{"tag": t, "digest": "q" + t, "size_gb": 1.0, "prefits": True, "inputs": "Text"} for t in queue]

        def bench_fn(label):
            bench(label)
            done.add(label)

        fit_ok = mock.Mock(fits=True, need_mb=1000, line=lambda: "fits")
        with mock.patch.object(library, "load_state", lambda: state), \
                mock.patch.object(library, "save_state", lambda st: None), \
                mock.patch.object(library, "ensure_ollama", lambda log=None: True), \
                mock.patch.object(library, "list_ollama_models", o.list), \
                mock.patch.object(library, "is_pulled", lambda t: t in o.installed), \
                mock.patch.object(library, "pull", o.pull), \
                mock.patch.object(library, "delete", o.delete), \
                mock.patch.object(library, "build_queue", lambda *x, **k: q), \
                mock.patch.object(library, "not_benchmarkable", lambda *x: None), \
                mock.patch.object(library, "ollama_entry", lambda t: {}), \
                mock.patch.object(library, "ollama_up", lambda: True), \
                mock.patch.object(library, "models_dir_free_gb", lambda: 999.0), \
                mock.patch.object(library.fit, "gpu_budget_mb", lambda h: 20000), \
                mock.patch.object(library.fit, "free_budget_mb", lambda h: 20000), \
                mock.patch.object(library.fit, "estimate", lambda *x: fit_ok):
            try:
                library._run_library(a, ["codegen"], speed_fn=lambda l: True, bench_fn=bench_fn,
                                     done_fn=lambda l: l in done, rebuild_fn=lambda: None, publish_fn=lambda: None,
                                     log=lambda *x: None)
            except KeyboardInterrupt:
                pass
        return o, state

    def state(self, **kw):
        return {"models": {}, "catalog": {"x": {}}, "preinstalled": ["mine:7b"], "to_delete": [], **kw}

    def test_tested_downloads_are_deleted_your_models_never(self):
        o, st = self.run_lib(self.state(), ["mine:7b"], ["mine:7b", "new:7b"], bench=lambda l: None)
        self.assertEqual(o.pulled, ["new:7b"])
        self.assertEqual(o.deleted, ["new:7b"])
        self.assertEqual(st["ours"], [])

    def test_a_stopped_run_keeps_the_download_and_the_next_run_uses_it(self):
        def stop(label):
            raise KeyboardInterrupt
        o, st = self.run_lib(self.state(), ["mine:7b"], ["new:7b"], bench=stop)
        self.assertEqual((o.pulled, o.deleted), (["new:7b"], []))
        self.assertEqual(st["ours"], ["new:7b"])
        # next run: already downloaded -> no second download, and deleted after its test (it isn't yours)
        o2, st2 = self.run_lib(st, ["mine:7b", "new:7b"], ["new:7b"], bench=lambda l: None)
        self.assertEqual(o2.pulled, [])
        self.assertEqual(o2.deleted, ["new:7b"])

    def test_untested_leftovers_are_kept_tested_ones_cleaned(self):
        st = self.state(to_delete=["waiting:7b", "finished:7b"])
        o, st = self.run_lib(st, ["mine:7b", "waiting:7b", "finished:7b"], ["waiting:7b"], bench=lambda l: None,
                             done={"ollama:finished:7b"})
        self.assertIn("finished:7b", o.deleted)            # tested earlier, deletion had failed: cleaned up
        self.assertEqual(o.pulled, [])                     # waiting:7b reused, not downloaded again
        self.assertIn("waiting:7b", o.deleted)             # ...and deleted once tested
        self.assertNotIn("mine:7b", o.deleted)


if __name__ == "__main__":
    unittest.main()
