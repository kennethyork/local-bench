from ..hfdata import rows
from ..sandbox import run_python
from ..textutil import extract_python
from . import Category, Grade, Item

SYSTEM = "You are an expert Python engineer who fixes bugs precisely."


class Debugging(Category):
    name = "debugging"
    title = "Debugging"
    description = "Find and fix a subtle bug in a Python function, given example tests."
    method = "HumanEvalFix (bigcode/humanevalpack, 164 bugs). Fixed function must pass the hidden test suite."
    max_tokens = 1024

    def load(self):
        items = []
        for r in rows("bigcode/humanevalpack", "python", "test"):
            buggy = r["declaration"] + r["buggy_solution"]
            prompt = (f"This Python function has a bug.\n\n```python\n{buggy.strip()}\n```\n\n"
                      f"It should satisfy these tests:\n\n```python\n{r['example_test'].strip()}\n```\n\n"
                      f"Fix the bug. Reply with the complete corrected function `{r['entry_point']}` "
                      "in a single ```python code block.")
            items.append(Item(r["task_id"].replace("Python/", "HumanEvalFix/"),
                              [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                              {"entry": r["entry_point"], "imports": r.get("import") or "",
                               "setup": r.get("test_setup") or "", "test": r["test"],
                               "bug_type": r.get("bug_type")}))
        return items

    def grade(self, item, text):
        p = item.payload
        code = extract_python(text, p["entry"])
        ok, err = run_python("\n".join([p["imports"], p["setup"], code, "", p["test"]]))
        return Grade(1.0 if ok else 0.0, err, {"bug_type": p["bug_type"]})

    def summarize(self, records):
        by = {}
        for r in records:
            b = by.setdefault(r.get("bug_type") or "?", [0, 0])
            b[0] += r["score"]
            b[1] += 1
        return {"by_bug_type": {k: round(100 * v[0] / v[1], 1) for k, v in by.items()}}
