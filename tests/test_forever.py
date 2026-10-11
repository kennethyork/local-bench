"""Continuous benchmarking (library --forever): survives errors, refreshes the catalog, stops on Ctrl+C."""
import unittest
from unittest import mock

from localbench import cli


class Forever(unittest.TestCase):
    def run_library(self, forever, outcomes):
        calls, refreshed, idles = [], [], []

        def fake_run(a, *args, **kw):
            calls.append(1)
            refreshed.append(a.refresh)
            outcome = outcomes[len(calls) - 1]
            if outcome is not None:
                raise outcome

        argv = ["library", "--parallel", "1"] + (["--forever"] if forever else [])
        with mock.patch("localbench.library.run_library", fake_run), \
                mock.patch.object(cli, "_idle", lambda s: idles.append(s)), \
                mock.patch.object(cli, "_report", lambda a: None), \
                mock.patch.object(cli, "kill_stale_private", lambda: None), \
                mock.patch.object(cli, "kill_orphan_runners", lambda: None), \
                mock.patch.object(cli.PRIVATE, "stop", lambda *a, **k: None):
            cli.main(argv)
        return calls, refreshed, idles

    def test_single_pass_without_forever(self):
        calls, _, idles = self.run_library(False, [None])
        self.assertEqual((len(calls), idles), (1, []))

    def test_keeps_going_refreshes_and_recovers(self):
        calls, refreshed, idles = self.run_library(True, [None, RuntimeError("boom"), None, KeyboardInterrupt()])
        self.assertEqual(len(calls), 4)
        self.assertEqual(refreshed, [False, True, True, True])  # later passes re-read ollama.com
        self.assertEqual(idles, [6 * 3600, 600, 6 * 3600])     # caught up / error retry / caught up

    def test_errors_are_raised_without_forever(self):
        with self.assertRaises(RuntimeError):
            self.run_library(False, [RuntimeError("boom")])


if __name__ == "__main__":
    unittest.main()
