"""`localbench compare`: before/after per category, paired item by item."""
import io
import unittest
from contextlib import redirect_stdout
from unittest import mock

from localbench import cli
from localbench.categories import PRESETS, registry


def cat(title, scores):
    recs = [{"id": f"i{n}", "score": s} for n, s in enumerate(scores)]
    return {"title": title, "score": 100 * sum(scores) / len(scores), "records": recs}


CATS = {
    "ollama:qwen3:14b": {"stack": cat("Stack", [0, 0, 1, 0.5] * 5), "agent": cat("Agent", [1, 0] * 10),
                         "codegen": cat("Code", [1] * 20), "ui": cat("UI", [1])},
    "ollama:ocadox-coder": {"stack": cat("Stack", [1, 1, 1, 0.5] * 5), "agent": cat("Agent", [1, 0] * 10),
                            "codegen": cat("Code", [1] * 10 + [0] * 10)},
}


class Compare(unittest.TestCase):
    def test_paired_changes(self):
        rows = {r["category"]: r for r in cli.compare(CATS, "ollama:qwen3:14b", "ollama:ocadox-coder")}
        self.assertEqual(set(rows), {"stack", "agent", "codegen"})  # only shared categories
        self.assertAlmostEqual(rows["stack"]["delta"], 50.0)
        self.assertTrue(rows["stack"]["significant"])
        self.assertEqual((rows["stack"]["fixed"], rows["stack"]["broke"]), (10, 0))
        self.assertEqual(rows["agent"]["delta"], 0)
        self.assertFalse(rows["agent"]["significant"])
        self.assertTrue(rows["codegen"]["significant"] and rows["codegen"]["delta"] < 0)
        self.assertEqual(rows["codegen"]["broke"], 10)

    def test_command_output(self):
        out = io.StringIO()
        with mock.patch("localbench.report.collect", return_value={"cats": CATS}), redirect_stdout(out):
            cli.main(["compare", "-m", "qwen3:14b", "-m", "ocadox-coder", "-c", "training"])
        text = out.getvalue()
        self.assertIn("+50.0", text)
        self.assertIn("Got worse: Code", text)
        self.assertNotIn("UI", text)

    def test_unknown_model_explains_what_to_run(self):
        with mock.patch("localbench.report.collect", return_value={"cats": CATS}), self.assertRaises(SystemExit) as e:
            cli.main(["compare", "-m", "qwen3:14b", "-m", "nope"])
        self.assertIn("-c training", str(e.exception.code))

    def test_training_preset_is_registered(self):
        self.assertTrue(set(PRESETS["training"]) <= set(registry()))


if __name__ == "__main__":
    unittest.main()
