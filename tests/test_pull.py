"""Model downloads: a dropped, stalled or unfinished pull is resumed instead of hanging or failing the model."""
import json
import unittest
from unittest import mock

import requests

from localbench import backends


class FakeResponse:
    def __init__(self, lines, raise_after=None):
        self.lines, self.raise_after = lines, raise_after

    def raise_for_status(self):
        pass

    def close(self):
        pass

    def iter_lines(self):
        for line in self.lines:
            yield json.dumps(line).encode()
        if self.raise_after:
            raise self.raise_after


OK = [{"status": "pulling manifest"}, {"status": "pulling abc", "digest": "abc", "total": 10, "completed": 5},
      {"status": "pulling abc", "digest": "abc", "total": 10, "completed": 10}, {"status": "success"}]
CUT = [{"status": "pulling abc", "digest": "abc", "total": 10, "completed": 5}]


class Pull(unittest.TestCase):
    def run_pull(self, responses):
        calls = []

        def post(*a, **kw):
            calls.append(kw.get("timeout"))
            r = responses[len(calls) - 1]
            if isinstance(r, Exception):
                raise r
            return r

        with mock.patch.object(backends.requests, "post", post), mock.patch.object(backends.time, "sleep", lambda s: None):
            backends.pull("m:7b", log=lambda *_: None)
        return calls

    def test_clean_download(self):
        calls = self.run_pull([FakeResponse(OK)])
        self.assertEqual(len(calls), 1)
        self.assertIsNotNone(calls[0][1])  # has a read timeout, so a silent stream can't hang forever

    def test_resumes_after_drop_stall_and_early_end(self):
        calls = self.run_pull([FakeResponse(CUT, requests.exceptions.ChunkedEncodingError("reset")),
                               FakeResponse(CUT, requests.exceptions.ReadTimeout("stalled")),
                               requests.exceptions.ConnectionError("refused"),
                               FakeResponse(CUT),  # stream ended without "success"
                               FakeResponse(OK)])
        self.assertEqual(len(calls), 5)

    def test_gives_up_after_five_attempts(self):
        with self.assertRaises(RuntimeError) as e:
            self.run_pull([FakeResponse(CUT)] * 5)
        self.assertIn("5 times", str(e.exception))

    def test_no_progress_counts_as_a_stall(self):
        frozen = [{"status": "pulling abc", "digest": "abc", "total": 10, "completed": 5}] * 3
        t = iter([0.0, 1.0, 400.0, 401.0] + [500.0] * 50)
        with mock.patch.object(backends.time, "monotonic", lambda: next(t)), \
                mock.patch.object(backends.requests, "post", lambda *a, **k: FakeResponse(frozen)):
            with self.assertRaises(RuntimeError) as e:
                backends._pull_once("m:7b", lambda *_: None, None)
        self.assertIn("no progress", str(e.exception))


if __name__ == "__main__":
    unittest.main()
