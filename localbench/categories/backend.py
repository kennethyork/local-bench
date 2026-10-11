"""Back end: SQL against a small shop database, plus API handlers and service building blocks.

SQL answers are run in SQLite and compared with a reference query's rows (order matters only where
the task asks for one). Python answers run against unit tests; each test counts, so partial
solutions earn partial credit. Everything runs in the sandbox (separate interpreter, temp dir, timeout).
"""
import json
import re

from ..sandbox import run_python_output
from ..textutil import code_blocks, extract_python
from . import Category, Grade, Item

SYSTEM = "You are a senior back-end engineer. You write correct, production-quality code."

# --------------------------------------------------------------------------- SQL

SCHEMA = """
CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT NOT NULL, country TEXT NOT NULL, signup_date TEXT NOT NULL);
CREATE TABLE products  (id INTEGER PRIMARY KEY, name TEXT NOT NULL, category TEXT NOT NULL);
CREATE TABLE orders    (id INTEGER PRIMARY KEY, customer_id INTEGER NOT NULL REFERENCES customers(id),
                        order_date TEXT NOT NULL, status TEXT NOT NULL CHECK (status IN ('paid','refunded','pending')));
CREATE TABLE order_items (order_id INTEGER NOT NULL REFERENCES orders(id), product_id INTEGER NOT NULL REFERENCES products(id),
                          qty INTEGER NOT NULL, unit_price REAL NOT NULL);
"""
DATA = """
INSERT INTO customers VALUES (1,'Ada','UK','2025-11-02'),(2,'Bo','US','2025-12-15'),(3,'Chen','US','2026-01-20'),
  (4,'Dara','DE','2026-02-01'),(5,'Eli','UK','2026-02-11'),(6,'Fay','DE','2026-03-05');
INSERT INTO products VALUES (1,'Laptop','electronics'),(2,'Mouse','electronics'),(3,'Desk','furniture'),
  (4,'Chair','furniture'),(5,'Lamp','furniture');
INSERT INTO orders VALUES (1,1,'2026-01-05','paid'),(2,1,'2026-02-10','paid'),(3,2,'2026-01-22','refunded'),
  (4,2,'2026-03-03','paid'),(5,3,'2026-02-14','paid'),(6,3,'2026-03-18','pending'),(7,4,'2026-03-01','paid'),
  (8,5,'2026-03-09','paid'),(9,5,'2026-03-30','refunded'),(10,1,'2026-03-25','paid');
INSERT INTO order_items VALUES (1,1,1,1200.0),(1,2,2,25.0),(2,3,1,350.0),(3,1,1,1150.0),(4,2,3,24.5),(4,5,2,40.0),
  (5,4,4,150.0),(5,2,1,25.0),(6,3,2,340.0),(7,1,1,1180.0),(7,4,2,145.0),(8,5,1,42.0),(8,2,1,25.0),
  (9,4,1,150.0),(10,2,4,23.0),(10,5,2,39.0);
"""

# (id, task, reference query, order matters)
SQL_TASKS = [
    ("revenue_by_country",
     "Total revenue per country from paid orders only (revenue = qty * unit_price). Columns: country, revenue "
     "(rounded to 2 decimals). Highest revenue first.",
     """SELECT c.country, ROUND(SUM(oi.qty * oi.unit_price), 2) AS revenue FROM customers c
        JOIN orders o ON o.customer_id = c.id AND o.status = 'paid' JOIN order_items oi ON oi.order_id = o.id
        GROUP BY c.country ORDER BY revenue DESC""", True),
    ("never_ordered",
     "Customers who have never placed an order of any status. Column: name. Sorted by name.",
     "SELECT name FROM customers WHERE id NOT IN (SELECT customer_id FROM orders) ORDER BY name", True),
    ("top_product_per_category",
     "For each product category, the product with the most units sold in paid orders. "
     "Columns: category, product (the product's name), units. Sorted by category.",
     """WITH u AS (SELECT p.category, p.name AS product, SUM(oi.qty) AS units FROM products p
          JOIN order_items oi ON oi.product_id = p.id JOIN orders o ON o.id = oi.order_id AND o.status = 'paid'
          GROUP BY p.id),
        r AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY category ORDER BY units DESC) AS rn FROM u)
        SELECT category, product, units FROM r WHERE rn = 1 ORDER BY category""", True),
    ("monthly_revenue",
     "Paid revenue per calendar month. Columns: month (text 'YYYY-MM'), revenue (rounded to 2 decimals). "
     "Sorted by month.",
     """SELECT substr(o.order_date, 1, 7) AS month, ROUND(SUM(oi.qty * oi.unit_price), 2) AS revenue
        FROM orders o JOIN order_items oi ON oi.order_id = o.id WHERE o.status = 'paid'
        GROUP BY month ORDER BY month""", True),
    ("above_average_spenders",
     "Customers whose total paid spend is greater than the average total paid spend of customers who have at "
     "least one paid order. Columns: name, spend (rounded to 2 decimals). Highest spend first.",
     """WITH s AS (SELECT o.customer_id, SUM(oi.qty * oi.unit_price) AS spend FROM orders o
          JOIN order_items oi ON oi.order_id = o.id WHERE o.status = 'paid' GROUP BY o.customer_id)
        SELECT c.name, ROUND(s.spend, 2) AS spend FROM s JOIN customers c ON c.id = s.customer_id
        WHERE s.spend > (SELECT AVG(spend) FROM s) ORDER BY s.spend DESC""", True),
    ("status_counts",
     "For every customer (including those with no orders): name and the number of their paid, refunded and "
     "pending orders. Columns: name, paid, refunded, pending (0 when none). Sorted by name.",
     """SELECT c.name, COALESCE(SUM(o.status = 'paid'), 0) AS paid, COALESCE(SUM(o.status = 'refunded'), 0) AS refunded,
        COALESCE(SUM(o.status = 'pending'), 0) AS pending FROM customers c LEFT JOIN orders o ON o.customer_id = c.id
        GROUP BY c.id ORDER BY c.name""", True),
]

SQL_PROMPT = """You are working with this SQLite database:

```sql
{schema}
```

Write one SQLite query for this task:

{task}

Reply with the query in a single ```sql code block."""

SQL_CHECK = """import json, sqlite3
con = sqlite3.connect(":memory:")
con.executescript({schema} + {data})
norm = lambda rows: [tuple(round(v, 2) if isinstance(v, float) else v for v in r) for r in rows]
ref = norm(con.execute({ref}).fetchall())
try:
    got = norm(con.execute({query}).fetchall())
except Exception as e:
    print("__LB__" + json.dumps({{"score": 0, "detail": "SQL error: " + str(e)[:200]}}))
    raise SystemExit
if got == ref:
    res = {{"score": 1, "detail": ""}}
elif {ordered} and sorted(map(repr, got)) == sorted(map(repr, ref)):
    res = {{"score": 0.5, "detail": "right rows, wrong order"}}
else:
    res = {{"score": 0, "detail": "expected " + repr(ref[:4])[:150] + ", got " + repr(got[:4])[:150]}}
print("__LB__" + json.dumps(res))
"""

# --------------------------------------------------------------------------- Python services

PY_TASKS = [
    ("todo_api", "TodoAPI",
     """Implement an in-memory REST API for to-do items as a class `TodoAPI` with a method
`handle(self, method, path, body=None)` that returns a tuple `(status_code, response_body)`.

- `GET /todos` -> 200, a list of all todos in creation order.
- `POST /todos` with body `{"title": <non-empty str>}` -> 201, the created todo
  `{"id": <int>, "title": <str>, "done": False}`. Ids start at 1 and increase by 1. A missing, empty or
  non-string title -> 400 with body `{"error": <message>}`.
- `GET /todos/<id>` -> 200 with the todo, or 404 `{"error": ...}` if it doesn't exist.
- `PATCH /todos/<id>` with body `{"done": <bool>}` -> 200 with the updated todo; 400 if `done` is missing or
  not a bool; 404 if the todo doesn't exist.
- `DELETE /todos/<id>` -> 204 with body `None`; 404 if it doesn't exist.
- A path that isn't `/todos` or `/todos/<integer id>` -> 404. A known path with an unsupported method -> 405.""",
     ["a = TodoAPI(); assert a.handle('GET', '/todos') == (200, [])",
      "a = TodoAPI(); s, b = a.handle('POST', '/todos', {'title': 'x'}); assert s == 201 and b == {'id': 1, 'title': 'x', 'done': False}",
      "a = TodoAPI(); a.handle('POST', '/todos', {'title': 'x'}); s, b = a.handle('POST', '/todos', {'title': 'y'}); assert b['id'] == 2",
      "a = TodoAPI(); assert a.handle('POST', '/todos', {})[0] == 400 and a.handle('POST', '/todos', {'title': ''})[0] == 400",
      "a = TodoAPI(); assert a.handle('POST', '/todos', {'title': 5})[0] == 400 and a.handle('POST', '/todos', None)[0] == 400",
      "a = TodoAPI(); a.handle('POST', '/todos', {'title': 'x'}); assert a.handle('GET', '/todos/1') == (200, {'id': 1, 'title': 'x', 'done': False})",
      "a = TodoAPI(); s, b = a.handle('GET', '/todos/9'); assert s == 404 and 'error' in b",
      "a = TodoAPI(); a.handle('POST', '/todos', {'title': 'x'}); assert a.handle('PATCH', '/todos/1', {'done': True}) == (200, {'id': 1, 'title': 'x', 'done': True})",
      "a = TodoAPI(); a.handle('POST', '/todos', {'title': 'x'}); assert a.handle('PATCH', '/todos/1', {'done': 'yes'})[0] == 400",
      "a = TodoAPI(); a.handle('POST', '/todos', {'title': 'x'}); assert a.handle('DELETE', '/todos/1') == (204, None) and a.handle('GET', '/todos/1')[0] == 404",
      "a = TodoAPI(); assert a.handle('DELETE', '/todos/3')[0] == 404 and a.handle('PATCH', '/todos/3', {'done': True})[0] == 404",
      "a = TodoAPI(); assert a.handle('GET', '/users')[0] == 404 and a.handle('GET', '/todos/abc')[0] == 404",
      "a = TodoAPI(); assert a.handle('PUT', '/todos')[0] == 405 and a.handle('POST', '/todos/1', {'title': 'x'})[0] == 405",
      "a = TodoAPI(); a.handle('POST', '/todos', {'title': 'a'}); a.handle('POST', '/todos', {'title': 'b'}); assert [t['title'] for t in a.handle('GET', '/todos')[1]] == ['a', 'b']"]),
    ("router", "Router",
     """Implement a URL router as a class `Router`:

- `add(self, method, pattern, handler)`: register `handler` for an HTTP method and a path pattern such as
  `/users/{id}/posts/{post_id}`. `{name}` matches exactly one non-empty path segment (no `/`).
- `dispatch(self, method, path)`: find the route whose pattern matches the whole path and return
  `(200, handler(**params))`, where params maps each `{name}` to the matched segment (a str).
  If no pattern matches the path -> `(404, None)`. If a pattern matches but not for this method -> `(405, None)`.
  Static segments must match exactly; a trailing slash is not ignored.""",
     ["r = Router(); r.add('GET', '/health', lambda: 'ok'); assert r.dispatch('GET', '/health') == (200, 'ok')",
      "r = Router(); r.add('GET', '/users/{id}', lambda id: id); assert r.dispatch('GET', '/users/42') == (200, '42')",
      "r = Router(); r.add('GET', '/users/{uid}/posts/{pid}', lambda uid, pid: (uid, pid)); assert r.dispatch('GET', '/users/7/posts/x9') == (200, ('7', 'x9'))",
      "r = Router(); r.add('GET', '/users/{id}', lambda id: id); assert r.dispatch('GET', '/users') == (404, None)",
      "r = Router(); r.add('GET', '/users/{id}', lambda id: id); assert r.dispatch('GET', '/users/1/extra') == (404, None)",
      "r = Router(); r.add('GET', '/users/{id}', lambda id: id); assert r.dispatch('GET', '/users/') == (404, None)",
      "r = Router(); r.add('GET', '/users/{id}', lambda id: id); assert r.dispatch('DELETE', '/users/1') == (405, None)",
      "r = Router(); r.add('GET', '/a/{x}', lambda x: 'get'); r.add('POST', '/a/{x}', lambda x: 'post'); assert r.dispatch('POST', '/a/1') == (200, 'post')",
      "r = Router(); r.add('GET', '/files/{name}', lambda name: name); assert r.dispatch('GET', '/files/a.txt') == (200, 'a.txt')",
      "r = Router(); r.add('GET', '/health', lambda: 'ok'); assert r.dispatch('GET', '/healthz') == (404, None) and r.dispatch('GET', '/health/') == (404, None)"]),
    ("paginate", "paginate",
     """Implement `paginate(items, page, per_page)` for a list API. Return a dict:
`{"items": <the slice for this page>, "page": page, "per_page": per_page, "total": len(items),
"pages": <number of pages, 0 when there are no items>, "next": <next page number or None>,
"prev": <previous page number or None>}`.
Raise `ValueError` if `page < 1`, or if `per_page` is not between 1 and 100 inclusive. A page past the end is
not an error: it returns an empty `items` list (with `next` None and `prev` = page - 1).""",
     ["r = paginate(list(range(10)), 1, 3); assert r['items'] == [0, 1, 2] and r['pages'] == 4 and r['total'] == 10",
      "r = paginate(list(range(10)), 1, 3); assert r['next'] == 2 and r['prev'] is None",
      "r = paginate(list(range(10)), 4, 3); assert r['items'] == [9] and r['next'] is None and r['prev'] == 3",
      "r = paginate([], 1, 10); assert r['items'] == [] and r['pages'] == 0 and r['next'] is None and r['prev'] is None",
      "r = paginate(list(range(5)), 9, 2); assert r['items'] == [] and r['next'] is None and r['prev'] == 8",
      "r = paginate(list(range(6)), 2, 3); assert r['pages'] == 2 and r['next'] is None and r['page'] == 2 and r['per_page'] == 3",
      "for args in [(0, 10), (1, 0), (1, 101), (-1, 5)]:\n    try:\n        paginate([1, 2], *args); raise AssertionError('no ValueError for %r' % (args,))\n    except ValueError:\n        pass",
      "r = paginate(list(range(5)), 1, 100); assert r['items'] == [0, 1, 2, 3, 4] and r['pages'] == 1 and r['next'] is None"]),
    ("rate_limiter", "RateLimiter",
     """Implement a per-key token-bucket rate limiter as a class `RateLimiter(capacity, refill_per_sec, clock)`.
`clock` is a zero-argument function returning the current time in seconds (float); never call time.time().
`allow(self, key)` returns True and consumes one token if the bucket for `key` has at least one token,
otherwise False. Each key starts with a full bucket of `capacity` tokens. Buckets refill continuously at
`refill_per_sec` tokens per second, never above `capacity`. Keys are independent.""",
     ["t = [0.0]; r = RateLimiter(3, 1.0, lambda: t[0]); assert [r.allow('a') for _ in range(4)] == [True, True, True, False]",
      "t = [0.0]; r = RateLimiter(2, 1.0, lambda: t[0]); r.allow('a'); r.allow('a'); assert r.allow('b') is True",
      "t = [0.0]; r = RateLimiter(2, 1.0, lambda: t[0]); r.allow('a'); r.allow('a'); t[0] = 1.0; assert r.allow('a') is True and r.allow('a') is False",
      "t = [0.0]; r = RateLimiter(2, 2.0, lambda: t[0]); r.allow('a'); r.allow('a'); t[0] = 0.25; assert r.allow('a') is False; t[0] = 0.5; assert r.allow('a') is True",
      "t = [0.0]; r = RateLimiter(3, 1.0, lambda: t[0]); t[0] = 100.0; assert [r.allow('a') for _ in range(4)] == [True, True, True, False]",
      "t = [0.0]; r = RateLimiter(1, 0.5, lambda: t[0]); assert r.allow('a') and not r.allow('a'); t[0] = 1.9; assert not r.allow('a'); t[0] = 2.0; assert r.allow('a')"]),
    ("signed_tokens", "sign",
     """Implement stateless signed tokens (like a minimal JWT) with the standard library only:

- `sign(payload, secret, expires_at)`: `payload` is a JSON-serialisable dict, `secret` a str, `expires_at` an
  int Unix timestamp. Return a URL-safe string token carrying the payload and expiry, signed with
  HMAC-SHA256 using `secret`.
- `verify(token, secret, now)`: return the original payload dict if the signature is valid and `now < expires_at`.
  Raise `ValueError` for a bad signature, a wrong secret, an expired token or a malformed token.
  Compare signatures in constant time.""",
     ["t = sign({'sub': 'u1', 'role': 'admin'}, 's3cret', 1000); assert verify(t, 's3cret', 999) == {'sub': 'u1', 'role': 'admin'}",
      "t = sign({'sub': 'u1'}, 's3cret', 1000); assert isinstance(t, str) and all(c.isalnum() or c in '-_.=' for c in t)",
      "t = sign({'sub': 'u1'}, 's3cret', 1000)\ntry:\n    verify(t, 'other', 1); raise AssertionError('accepted wrong secret')\nexcept ValueError:\n    pass",
      "t = sign({'sub': 'u1'}, 's3cret', 1000)\ntry:\n    verify(t, 's3cret', 1000); raise AssertionError('accepted expired token')\nexcept ValueError:\n    pass",
      "t = sign({'sub': 'u1'}, 's3cret', 1000)\nfor i in range(len(t)):\n    c = 'A' if t[i] != 'A' else 'B'\n    bad = t[:i] + c + t[i + 1:]\n    try:\n        r = verify(bad, 's3cret', 1)\n    except ValueError:\n        continue\n    assert r == {'sub': 'u1'}, 'tampered token accepted with a different payload'",
      "for bad in ['', 'abc', 'a.b.c.d', '...']:\n    try:\n        verify(bad, 's3cret', 1); raise AssertionError('accepted ' + repr(bad))\n    except ValueError:\n        pass",
      "a = sign({'sub': 'u1'}, 's3cret', 1000); b = sign({'sub': 'u2'}, 's3cret', 1000); assert a != b and verify(b, 's3cret', 5) == {'sub': 'u2'}"]),
    ("ttl_cache", "TTLCache",
     """Implement `TTLCache(maxsize, ttl, clock)`, an in-memory cache. `clock` is a zero-argument function
returning the current time in seconds; never call time.time().
- `set(self, key, value)`: store the value (replacing any existing one, which also resets its expiry).
- `get(self, key, default=None)`: return the value, or `default` if missing or expired. Entries expire
  `ttl` seconds after they were set (an entry set at t is gone at t + ttl).
- When storing a new key would exceed `maxsize`, evict the least recently used entry, where both `get` hits
  and `set` count as a use. Expired entries never count towards the size.
- `__len__` returns the number of entries that haven't expired.""",
     ["t = [0]; c = TTLCache(2, 10, lambda: t[0]); c.set('a', 1); assert c.get('a') == 1 and c.get('zz', 'd') == 'd'",
      "t = [0]; c = TTLCache(2, 10, lambda: t[0]); c.set('a', 1); t[0] = 9.9; assert c.get('a') == 1; t[0] = 10; assert c.get('a') is None",
      "t = [0]; c = TTLCache(2, 10, lambda: t[0]); c.set('a', 1); c.set('b', 2); c.set('c', 3); assert c.get('a') is None and c.get('b') == 2 and c.get('c') == 3",
      "t = [0]; c = TTLCache(2, 10, lambda: t[0]); c.set('a', 1); c.set('b', 2); c.get('a'); c.set('c', 3); assert c.get('a') == 1 and c.get('b') is None",
      "t = [0]; c = TTLCache(2, 10, lambda: t[0]); c.set('a', 1); t[0] = 5; c.set('a', 2); t[0] = 12; assert c.get('a') == 2",
      "t = [0]; c = TTLCache(3, 10, lambda: t[0]); c.set('a', 1); c.set('b', 2); t[0] = 11; c.set('c', 3); assert len(c) == 1",
      "t = [0]; c = TTLCache(2, 10, lambda: t[0]); c.set('a', 1); t[0] = 11; c.set('b', 2); c.set('c', 3); assert c.get('b') == 2 and c.get('c') == 3"]),
    ("retry", "retry",
     """Implement `retry(fn, attempts, base_delay, sleep)` for calling flaky services.
`TransientError` is already defined (`class TransientError(Exception)`), so don't redefine it.
Call `fn()` and return its result. If it raises `TransientError`, call `sleep(delay)` and try again, with delays
`base_delay`, `base_delay * 2`, `base_delay * 4`, ... (exponential backoff), up to `attempts` calls in total.
If the last attempt also fails, re-raise that last `TransientError`. Any other exception is re-raised
immediately without retrying. Never sleep after the final attempt. Use only the given `sleep` function.""",
     ["s = []; assert retry(lambda: 5, 3, 1, s.append) == 5 and s == []",
      "n = [0]\ndef f():\n    n[0] += 1\n    if n[0] < 3:\n        raise TransientError()\n    return 'ok'\ns = []; assert retry(f, 5, 0.5, s.append) == 'ok' and s == [0.5, 1.0] and n[0] == 3",
      "n = [0]\ndef f():\n    n[0] += 1\n    raise TransientError(str(n[0]))\ns = []\ntry:\n    retry(f, 3, 1, s.append); raise AssertionError('no raise')\nexcept TransientError as e:\n    assert str(e) == '3' and s == [1, 2] and n[0] == 3",
      "n = [0]\ndef f():\n    n[0] += 1\n    raise KeyError('x')\ns = []\ntry:\n    retry(f, 4, 1, s.append); raise AssertionError('no raise')\nexcept KeyError:\n    assert n[0] == 1 and s == []",
      "n = [0]\ndef f():\n    n[0] += 1\n    raise TransientError()\ns = []\ntry:\n    retry(f, 1, 1, s.append)\nexcept TransientError:\n    assert n[0] == 1 and s == []"]),
    ("log_summary", "summarize",
     """Implement `summarize(lines)` for web-server access logs. Each well-formed line looks like:
`203.0.113.9 - - [11/Oct/2026:13:55:36 +0000] "GET /api/users?page=2 HTTP/1.1" 200 512 34ms`
(IP, timestamp, quoted request line, status code, response bytes, response time in ms).
Return a dict:
- `"requests"`: number of well-formed lines,
- `"malformed"`: number of other non-empty lines (ignore blank lines),
- `"errors"`: well-formed lines with status >= 500,
- `"by_path"`: dict mapping path (without the query string) -> count,
- `"p95_ms"`: the 95th-percentile response time using the nearest-rank method (the value at rank
  ceil(0.95 * n) in the sorted list, 1-based), or None if there are no requests.""",
     ["assert summarize([])['p95_ms'] is None and summarize([])['requests'] == 0",
      "L = ['1.1.1.1 - - [11/Oct/2026:13:55:36 +0000] \"GET /a?x=1 HTTP/1.1\" 200 5 10ms', '1.1.1.1 - - [11/Oct/2026:13:55:37 +0000] \"POST /a HTTP/1.1\" 503 0 20ms']\nr = summarize(L); assert r['requests'] == 2 and r['errors'] == 1 and r['by_path'] == {'/a': 2}",
      "L = ['garbage line', '', '1.1.1.1 - - [11/Oct/2026:13:55:36 +0000] \"GET /b HTTP/1.1\" 404 5 7ms']\nr = summarize(L); assert r['malformed'] == 1 and r['requests'] == 1 and r['errors'] == 0",
      "L = ['1.1.1.1 - - [11/Oct/2026:13:55:36 +0000] \"GET /p HTTP/1.1\" 200 5 %dms' % ms for ms in range(1, 101)]\nassert summarize(L)['p95_ms'] == 95",
      "L = ['1.1.1.1 - - [11/Oct/2026:13:55:36 +0000] \"GET /p HTTP/1.1\" 200 5 %dms' % ms for ms in (50, 10, 30)]\nassert summarize(L)['p95_ms'] == 50",
      "L = ['1.1.1.1 - - [11/Oct/2026:13:55:36 +0000] \"GET /x HTTP/1.1\" 500 5 1ms', '1.1.1.1 - - [11/Oct/2026:13:55:36 +0000] \"GET /y?q HTTP/1.1\" 200 5 1ms', '1.1.1.1 - - [11/Oct/2026:13:55:36 +0000] \"GET /x HTTP/1.1\" 200 5 1ms']\nassert summarize(L)['by_path'] == {'/x': 2, '/y': 1}"]),
]

PY_PROMPT = """{spec}

Reply with the complete implementation (standard library only, including any imports) in a single
```python code block. Do not include tests or example usage."""

PRELUDE = {"retry": "class TransientError(Exception):\n    pass\n"}

PY_CHECK = """import json as __lb_json
__lb_tests = {tests}
__lb_ok, __lb_fail = 0, []
for __lb_i, __lb_t in enumerate(__lb_tests):
    try:
        exec(__lb_t, dict(globals()))
        __lb_ok += 1
    except BaseException as __lb_e:
        __lb_fail.append("test %d: %s %s" % (__lb_i + 1, type(__lb_e).__name__, str(__lb_e)[:80]))
print("__LB__" + __lb_json.dumps({{"score": __lb_ok / len(__lb_tests), "detail": "; ".join(__lb_fail[:3])}}))
"""


def _result(rc, out, err) -> Grade:
    m = re.search(r"__LB__(\{.*\})", out)
    if not m:
        if rc is None:
            return Grade(0.0, "timeout")
        lines = (err or out).strip().splitlines()
        return Grade(0.0, (lines[-1] if lines else f"exit {rc}")[:300])
    r = json.loads(m.group(1))
    return Grade(float(r["score"]), r["detail"])


class BackEnd(Category):
    name = "backend"
    title = "Back End"
    description = "SQL queries, REST API handlers and service building blocks (routing, rate limiting, caching, auth tokens)."
    method = (f"{len(SQL_TASKS)} SQL tasks on a shop database (rows must match a reference query) and "
              f"{len(PY_TASKS)} API/service tasks graded by unit tests (partial credit per test).")
    max_tokens = 2048
    uses_limit = False

    def load(self):
        items = []
        for sid, task, ref, ordered in SQL_TASKS:
            prompt = SQL_PROMPT.format(schema=SCHEMA.strip(), task=task)
            items.append(Item(f"sql/{sid}", [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                              {"kind": "sql", "ref": ref, "ordered": ordered}))
        for pid, symbol, spec, tests in PY_TASKS:
            items.append(Item(f"api/{pid}", [{"role": "system", "content": SYSTEM},
                                             {"role": "user", "content": PY_PROMPT.format(spec=spec)}],
                              {"kind": "py", "id": pid, "symbol": symbol, "tests": tests}))
        return items

    def grade(self, item, text):
        p = item.payload
        if p["kind"] == "sql":
            blocks = code_blocks(text, "sql")
            query = (max(blocks, key=len) if blocks else text).strip()
            src = SQL_CHECK.format(schema=json.dumps(SCHEMA), data=json.dumps(DATA), ref=json.dumps(p["ref"]),
                                   query=json.dumps(query), ordered=p["ordered"])
            return _result(*run_python_output(src))
        code = extract_python(text, p["symbol"] if p["symbol"][0].islower() else None)
        src = PRELUDE.get(p["id"], "") + code + "\n\n" + PY_CHECK.format(tests=repr(p["tests"]))
        return _result(*run_python_output(src, timeout=15))
