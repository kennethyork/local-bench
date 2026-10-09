import re

from ..hfdata import rows
from ..sandbox import run_python
from ..textutil import extract_python
from . import Category, Grade, Item

SYSTEM = "You are an expert Python engineer. Write correct, self-contained code."


class CodeGen(Category):
    name = "codegen"
    title = "Code Gen"
    description = "Implement a function from a short natural-language spec."
    method = "MBPP sanitized test split (google-research-datasets/mbpp, 257 tasks). Code must pass all asserts."
    max_tokens = 1024

    def load(self):
        items = []
        for r in rows("google-research-datasets/mbpp", "sanitized", "test"):
            tests = list(r["test_list"])
            m = re.search(r"assert\s+(?:not\s+)?\(?\s*(?:set\()?\s*(?:math\.isclose\()?\s*([A-Za-z_]\w*)\s*\(", tests[0])
            fn = m.group(1) if m else None
            prompt = (f"{r['prompt'].strip()}\n\nYour code must pass this test:\n\n```python\n{tests[0]}\n```\n\n"
                      "Reply with the complete implementation (including any imports) in a single "
                      "```python code block. Do not include the tests.")
            items.append(Item(f"MBPP/{r['task_id']}",
                              [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                              {"fn": fn, "imports": "\n".join(r["test_imports"] or []), "tests": tests}))
        return items

    def grade(self, item, text):
        p = item.payload
        code = extract_python(text, p["fn"])
        ok, err = run_python("\n".join([p["imports"], code, "", *p["tests"]]))
        return Grade(1.0 if ok else 0.0, err)
