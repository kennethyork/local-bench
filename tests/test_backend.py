"""The back-end category's tasks are solvable and graded fairly: reference solutions score 100%,
plausible wrong answers don't, and the SQL data has no ties that would make an answer ambiguous."""
import sqlite3
import unittest

from localbench.categories.backend import DATA, SCHEMA, SQL_TASKS, BackEnd

REFERENCE = {
    "todo_api": '''
import re
class TodoAPI:
    def __init__(self):
        self.todos, self.next_id = {}, 1
    def handle(self, method, path, body=None):
        if path == "/todos":
            if method == "GET":
                return 200, list(self.todos.values())
            if method == "POST":
                t = (body or {}).get("title") if isinstance(body, dict) else None
                if not isinstance(t, str) or not t:
                    return 400, {"error": "title required"}
                todo = {"id": self.next_id, "title": t, "done": False}
                self.todos[self.next_id] = todo
                self.next_id += 1
                return 201, todo
            return 405, {"error": "method not allowed"}
        m = re.fullmatch(r"/todos/(\\d+)", path)
        if not m:
            return 404, {"error": "not found"}
        if method not in ("GET", "PATCH", "DELETE"):
            return 405, {"error": "method not allowed"}
        todo = self.todos.get(int(m.group(1)))
        if todo is None:
            return 404, {"error": "not found"}
        if method == "GET":
            return 200, todo
        if method == "DELETE":
            del self.todos[todo["id"]]
            return 204, None
        if not isinstance(body, dict) or not isinstance(body.get("done"), bool):
            return 400, {"error": "done must be a bool"}
        todo["done"] = body["done"]
        return 200, todo
''',
    "router": '''
class Router:
    def __init__(self):
        self.routes = []
    def add(self, method, pattern, handler):
        self.routes.append((method, pattern.split("/"), handler))
    def dispatch(self, method, path):
        parts, path_found = path.split("/"), False
        for m, pat, h in self.routes:
            if len(pat) != len(parts):
                continue
            params, ok = {}, True
            for p, s in zip(pat, parts):
                if p.startswith("{") and p.endswith("}"):
                    if not s:
                        ok = False
                        break
                    params[p[1:-1]] = s
                elif p != s:
                    ok = False
                    break
            if ok:
                path_found = True
                if m == method:
                    return 200, h(**params)
        return (405, None) if path_found else (404, None)
''',
    "paginate": '''
def paginate(items, page, per_page):
    if page < 1 or not 1 <= per_page <= 100:
        raise ValueError("bad page")
    total = len(items)
    pages = -(-total // per_page)
    return {"items": items[(page - 1) * per_page: page * per_page], "page": page, "per_page": per_page,
            "total": total, "pages": pages, "next": page + 1 if page < pages else None,
            "prev": page - 1 if page > 1 else None}
''',
    "rate_limiter": '''
class RateLimiter:
    def __init__(self, capacity, refill_per_sec, clock):
        self.capacity, self.rate, self.clock, self.b = capacity, refill_per_sec, clock, {}
    def allow(self, key):
        now = self.clock()
        tokens, last = self.b.get(key, (self.capacity, now))
        tokens = min(self.capacity, tokens + (now - last) * self.rate)
        if tokens >= 1:
            self.b[key] = (tokens - 1, now)
            return True
        self.b[key] = (tokens, now)
        return False
''',
    "signed_tokens": '''
import base64, hashlib, hmac, json
def _b(x): return base64.urlsafe_b64encode(x).decode().rstrip("=")
def _u(s): return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))
def sign(payload, secret, expires_at):
    body = _b(json.dumps({"p": payload, "exp": expires_at}, sort_keys=True).encode())
    return body + "." + _b(hmac.new(secret.encode(), body.encode(), hashlib.sha256).digest())
def verify(token, secret, now):
    try:
        body, sig = token.split(".")
        good = hmac.new(secret.encode(), body.encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(_u(sig), good):
            raise ValueError("bad signature")
        d = json.loads(_u(body))
    except ValueError:
        raise
    except Exception as e:
        raise ValueError("malformed") from e
    if now >= d["exp"]:
        raise ValueError("expired")
    return d["p"]
''',
    "ttl_cache": '''
from collections import OrderedDict
class TTLCache:
    def __init__(self, maxsize, ttl, clock):
        self.maxsize, self.ttl, self.clock, self.d = maxsize, ttl, clock, OrderedDict()
    def _purge(self):
        now = self.clock()
        for k in [k for k, (_, exp) in self.d.items() if exp <= now]:
            del self.d[k]
    def get(self, key, default=None):
        self._purge()
        if key not in self.d:
            return default
        self.d.move_to_end(key)
        return self.d[key][0]
    def set(self, key, value):
        self._purge()
        self.d[key] = (value, self.clock() + self.ttl)
        self.d.move_to_end(key)
        while len(self.d) > self.maxsize:
            self.d.popitem(last=False)
    def __len__(self):
        self._purge()
        return len(self.d)
''',
    "retry": '''
def retry(fn, attempts, base_delay, sleep):
    for i in range(attempts):
        try:
            return fn()
        except TransientError:
            if i == attempts - 1:
                raise
            sleep(base_delay * 2 ** i)
''',
    "log_summary": '''
import math, re
LINE = re.compile(r'^\\S+ \\S+ \\S+ \\[[^\\]]+\\] "\\S+ (\\S+) [^"]*" (\\d{3}) \\d+ (\\d+)ms$')
def summarize(lines):
    r = {"requests": 0, "malformed": 0, "errors": 0, "by_path": {}, "p95_ms": None}
    times = []
    for line in lines:
        if not line.strip():
            continue
        m = LINE.match(line.strip())
        if not m:
            r["malformed"] += 1
            continue
        path = m.group(1).split("?")[0]
        r["requests"] += 1
        r["errors"] += int(m.group(2)) >= 500
        r["by_path"][path] = r["by_path"].get(path, 0) + 1
        times.append(int(m.group(3)))
    if times:
        r["p95_ms"] = sorted(times)[math.ceil(0.95 * len(times)) - 1]
    return r
''',
}


def answer(code, lang="python"):
    return f"```{lang}\n{code}\n```"


class BackEndCategory(unittest.TestCase):
    cat = BackEnd()
    items = {it.id: it for it in cat.load()}

    def test_every_python_task_has_a_reference_scoring_100(self):
        for it in self.items.values():
            if it.payload["kind"] != "py":
                continue
            g = self.cat.grade(it, answer(REFERENCE[it.payload["id"]]))
            self.assertEqual(g.score, 1.0, f"{it.id}: {g.detail}")

    def test_every_reference_query_scores_100(self):
        for it in self.items.values():
            if it.payload["kind"] == "sql":
                g = self.cat.grade(it, answer(it.payload["ref"], "sql"))
                self.assertEqual(g.score, 1.0, f"{it.id}: {g.detail}")

    def test_reference_queries_return_rows_without_ties(self):
        con = sqlite3.connect(":memory:")
        con.executescript(SCHEMA + DATA)
        for sid, _, ref, _ in SQL_TASKS:
            rows = con.execute(ref).fetchall()
            self.assertTrue(rows, sid)
        units = con.execute("""SELECT p.category, SUM(oi.qty) u FROM products p JOIN order_items oi ON oi.product_id = p.id
            JOIN orders o ON o.id = oi.order_id AND o.status = 'paid' GROUP BY p.id""").fetchall()
        for cat in {c for c, _ in units}:
            top = sorted((u for c, u in units if c == cat), reverse=True)
            self.assertNotEqual(top[0], top[1] if len(top) > 1 else None, f"tie for top product in {cat}")

    def test_wrong_sql_scores_zero_and_wrong_order_half(self):
        it = self.items["sql/revenue_by_country"]
        self.assertEqual(self.cat.grade(it, answer("SELECT country, 0 FROM customers", "sql")).score, 0.0)
        self.assertEqual(self.cat.grade(it, answer("SELEC nonsense", "sql")).score, 0.0)
        flipped = it.payload["ref"].replace("ORDER BY revenue DESC", "ORDER BY revenue ASC")
        self.assertEqual(self.cat.grade(it, answer(flipped, "sql")).score, 0.5)

    def test_partial_and_broken_python(self):
        it = self.items["api/retry"]
        no_backoff = "def retry(fn, attempts, base_delay, sleep):\n    for i in range(attempts):\n        try:\n" \
                     "            return fn()\n        except TransientError:\n            if i == attempts - 1:\n" \
                     "                raise\n            sleep(base_delay)\n"
        g = self.cat.grade(it, answer(no_backoff))
        self.assertTrue(0 < g.score < 1, g)
        self.assertEqual(self.cat.grade(it, "I can't help with that.").score, 0.0)
        self.assertEqual(self.cat.grade(it, answer("while True:\n    pass")).score, 0.0)  # timeout


if __name__ == "__main__":
    unittest.main()
