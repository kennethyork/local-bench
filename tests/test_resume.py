"""Once a model is benchmarked it stays done; partial models only finish what's missing."""
import unittest
from unittest import mock

from localbench import cli

CATS = ["debugging", "codegen", "security"]


def fake(results):
    return mock.patch.object(cli, "load_all", return_value=results)


class Resume(unittest.TestCase):
    def test_never_run_needs_everything(self):
        with fake([]):
            self.assertEqual(cli.missing_parts("ollama:m:1b", CATS, True), (True, CATS))

    def test_done_with_any_settings_is_done(self):
        results = [{"kind": "perf", "model": "ollama:m:1b"},
                   {"kind": "bench", "model": "ollama:m:1b", "settings": {"limit": 10},  # fewer items than default
                    "categories": [{"category": c} for c in CATS]}]
        with fake(results):
            self.assertTrue(cli.already_done("ollama:m:1b", CATS, True, 40))

    def test_partial_model_only_finishes_missing_parts(self):
        results = [{"kind": "perf", "model": "ollama:m:1b"},
                   {"kind": "bench", "model": "ollama:m:1b", "categories": [{"category": "debugging"}]}]
        with fake(results):
            self.assertEqual(cli.missing_parts("ollama:m:1b", CATS, True), (False, ["codegen", "security"]))

    def test_categories_split_across_runs_add_up(self):
        results = [{"kind": "perf", "model": "ollama:m:1b"},
                   {"kind": "bench", "model": "ollama:m:1b", "categories": [{"category": "debugging"}]},
                   {"kind": "bench", "model": "ollama:m:1b", "categories": [{"category": "codegen"},
                                                                          {"category": "security"}]}]
        with fake(results):
            self.assertTrue(cli.already_done("ollama:m:1b", CATS, True))

    def test_other_models_dont_count(self):
        with fake([{"kind": "perf", "model": "ollama:other:7b"}]):
            self.assertEqual(cli.missing_parts("ollama:m:1b", CATS, True), (True, CATS))

    def test_redo_runs_everything_even_when_done(self):
        results = [{"kind": "perf", "model": "ollama:m:1b"},
                   {"kind": "bench", "model": "ollama:m:1b", "categories": [{"category": c} for c in CATS]}]
        a = mock.Mock(redo=True, no_speed=False)
        with fake(results), mock.patch.object(cli, "_bench") as bench, mock.patch.object(cli, "_speed",
                                                                                         return_value=True) as speed:
            cli._speed_part("ollama:m:1b", a, CATS)
            cli._bench_part("ollama:m:1b", a, CATS)
        speed.assert_called_once()
        bench.assert_called_once_with("ollama:m:1b", a, CATS)

    def test_without_redo_done_parts_are_skipped(self):
        results = [{"kind": "perf", "model": "ollama:m:1b"},
                   {"kind": "bench", "model": "ollama:m:1b", "categories": [{"category": "debugging"}]}]
        a = mock.Mock(redo=False, no_speed=False)
        with fake(results), mock.patch.object(cli, "_bench") as bench, mock.patch.object(cli, "_speed") as speed:
            self.assertTrue(cli._speed_part("ollama:m:1b", a, CATS))
            cli._bench_part("ollama:m:1b", a, CATS)
        speed.assert_not_called()
        bench.assert_called_once_with("ollama:m:1b", a, ["codegen", "security"])


if __name__ == "__main__":
    unittest.main()
