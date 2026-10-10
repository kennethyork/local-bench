"""Which Ollama models count as benchmarkable on this machine."""
import unittest

from localbench.backends import not_benchmarkable


class Filters(unittest.TestCase):
    def test_local_chat_models_pass(self):
        self.assertIsNone(not_benchmarkable("llama3.2:1b", {"capabilities": ["completion", "tools"]}))
        self.assertIsNone(not_benchmarkable("qwen3.8:latest", {"capabilities": ["completion", "vision", "thinking"]}))

    def test_embedding_models_are_skipped(self):
        why = not_benchmarkable("nomic-embed-text:latest", {"capabilities": ["embedding"]})
        self.assertIn("can't answer prompts", why)

    def test_cloud_models_are_skipped(self):
        self.assertIn("cloud", not_benchmarkable("gpt-oss:120b-cloud", None))
        self.assertIn("cloud", not_benchmarkable("qwen3-coder:480b-cloud", {"capabilities": ["completion"]}))
        self.assertIn("cloud", not_benchmarkable("glm-4.6:cloud", None))
        self.assertIn("cloud", not_benchmarkable("x:latest", {"remote_host": "https://ollama.com:443"}))

    def test_cloud_in_a_family_name_is_not_a_cloud_tag(self):
        self.assertIsNone(not_benchmarkable("cloudy-llm:7b", {"capabilities": ["completion"]}))

    def test_unknown_capabilities_are_allowed(self):
        # older Ollama versions don't report capabilities; don't block those
        self.assertIsNone(not_benchmarkable("somemodel:7b", {}))
        self.assertIsNone(not_benchmarkable("not-pulled-yet:3b", None))


if __name__ == "__main__":
    unittest.main()
