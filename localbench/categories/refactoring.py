"""Refactoring: working-but-ugly code in, clean code out, behaviour must be preserved.

Input = a HumanEval canonical solution with its docstring removed and every local variable
renamed to v1, v2, ... (verified to still pass its tests). The model must clean it up.
Score: 0 if the hidden tests fail; otherwise 0.5 + 0.5 * (fraction of quality checks met).
"""
import ast
import builtins

from ..hfdata import rows
from ..sandbox import run_python
from ..textutil import extract_python
from . import Category, Grade, Item

SYSTEM = "You are a senior Python engineer doing careful, behaviour-preserving refactors."
SHORT_OK = {"i", "j", "k", "_", "n", "x"}
_COMPOUND = (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.AsyncFor, ast.AsyncWith)


def _func(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _obfuscate(src: str, entry: str) -> str | None:
    tree = ast.parse(src)
    fn = _func(tree, entry)
    if fn is None:
        return None
    arg_names = {a.arg for a in ast.walk(fn) if isinstance(a, ast.arg)}
    declared = set()
    for n in ast.walk(fn):
        if isinstance(n, (ast.Global, ast.Nonlocal)):
            declared |= set(n.names)
    stored = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id not in stored:
            stored.append(n.id)
    mapping = {name: f"v{i + 1}" for i, name in enumerate(
        x for x in stored if x not in arg_names and x not in declared and not hasattr(builtins, x))}
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and n.id in mapping:
            n.id = mapping[n.id]
    if fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(getattr(fn.body[0], "value", None), ast.Constant) \
            and isinstance(fn.body[0].value.value, str):
        fn.body = fn.body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def _depth(node, d=0):
    best = d
    for child in ast.iter_child_nodes(node):
        best = max(best, _depth(child, d + 1 if isinstance(child, _COMPOUND) else d))
    return best


class Refactoring(Category):
    name = "refactoring"
    title = "Refactoring"
    description = "Clean up obfuscated working code without changing behaviour."
    method = ("HumanEval solutions (bigcode/humanevalpack) with locals renamed v1..vN and docstrings stripped. "
              "Hidden tests must still pass (else 0); then +credit for descriptive names, a docstring, "
              "full type hints and no deeper nesting.")
    max_tokens = 1536

    def load(self):
        items = []
        for r in rows("bigcode/humanevalpack", "python", "test"):
            src = r["declaration"] + r["canonical_solution"]
            try:
                ugly = _obfuscate(src, r["entry_point"])
            except SyntaxError:
                ugly = None
            if not ugly or ugly.count("v1") == 0:
                continue  # nothing meaningful to refactor
            prompt = (f"Refactor this working Python code for readability and maintainability.\n\n"
                      f"```python\n{ugly}\n```\n\nRequirements:\n"
                      f"- Keep the function name `{r['entry_point']}` and its parameters exactly as they are.\n"
                      "- Behaviour must stay identical for every input.\n"
                      "- Use descriptive variable names, add a docstring, and add type hints for all "
                      "parameters and the return value.\n"
                      "- Keep it simple; don't add nesting.\n\n"
                      "Reply with the complete refactored code in a single ```python code block.")
            items.append(Item(r["task_id"].replace("Python/", "Refactor/"),
                              [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                              {"entry": r["entry_point"], "imports": r.get("import") or "",
                               "setup": r.get("test_setup") or "", "test": r["test"], "ugly": ugly,
                               "depth": _depth(_func(ast.parse(ugly), r["entry_point"]))}))
        return items

    def items(self, limit):
        out = []
        for it in super().items(None):
            if limit and len(out) >= limit:
                break
            p = it.payload  # the obfuscated input itself must still pass, or the item is unfair
            if run_python("\n".join([p["imports"], p["setup"], p["ugly"], "", p["test"]]))[0]:
                out.append(it)
        return out

    def grade(self, item, text):
        p = item.payload
        code = extract_python(text, p["entry"])
        ok, err = run_python("\n".join([p["imports"], p["setup"], code, "", p["test"]]))
        if not ok:
            return Grade(0.0, f"behaviour changed: {err}", {"checks": {}})
        try:
            fn = _func(ast.parse(code), p["entry"])
        except SyntaxError:
            fn = None
        if fn is None:
            return Grade(0.5, "tests pass; function not found for quality checks", {"checks": {}})
        args = [a for a in fn.args.args + fn.args.kwonlyargs]
        names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        names |= {a.arg for a in ast.walk(fn) if isinstance(a, ast.arg)}
        bad = [n for n in names if (len(n) < 3 and n not in SHORT_OK) or (n[0] == "v" and n[1:].isdigit())]
        checks = {
            "descriptive_names": len(bad) == 0,
            "docstring": ast.get_docstring(fn) is not None,
            "type_hints": fn.returns is not None and all(a.annotation is not None for a in args),
            "no_deeper_nesting": _depth(fn) <= p["depth"],
        }
        score = 0.5 + 0.5 * sum(checks.values()) / len(checks)
        missing = [k for k, v in checks.items() if not v]
        return Grade(score, ("missing: " + ", ".join(missing)) if missing else "", {"checks": checks})

    def summarize(self, records):
        keys = ["descriptive_names", "docstring", "type_hints", "no_deeper_nesting"]
        n = len(records) or 1
        return {"preserved_pct": round(100 * sum(r["score"] > 0 for r in records) / n, 1),
                **{k + "_pct": round(100 * sum(r.get("checks", {}).get(k, False) for r in records) / n, 1)
                   for k in keys}}

