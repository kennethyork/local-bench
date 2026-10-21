"""Stack: Rust and TypeScript tasks from a Tauri desktop app's code base, compiled and run against tests.

This is the category to watch when fine-tuning a model for one stack (Tauri 2 with a Rust core and a
React/TypeScript front end). Every task is the kind of small, self-contained piece such an app is made
of: version checks for the updater, an LRU cache, line-ending-safe edits, safe workspace paths, a
priority job queue, shell-style argument splitting, and on the front end byte and duration formatting,
a tab reducer, settings merging, an event emitter and model-list grouping.

Rust answers are built with `rustc --test` and TypeScript answers run on Node (types stripped). Each test
counts, so a partial answer earns partial credit; an answer that doesn't compile scores 0. A language
whose toolchain isn't installed is skipped rather than scored, and the report says so.
"""
import json
import re
import shutil
import subprocess
import sys

from ..sandbox import run_programs
from ..textutil import code_blocks
from . import Category, Grade, Item

SYSTEM = ("You are a senior engineer on a Tauri 2 desktop app: a Rust core and a React + TypeScript front end. "
          "You write correct, idiomatic, production-quality code.")

RUST_PROMPT = """Write this in Rust (edition 2021, standard library only - no crates):

{task}

Reply with the complete code in a single ```rust block. Do not include a `main` function or tests."""

TS_PROMPT = """Write this in TypeScript (no imports, no libraries; it runs on Node with the types stripped, so
don't use `enum` or `namespace`):

{task}

Reply with the complete code in a single ```ts block. Export nothing and don't include tests."""

# --------------------------------------------------------------------------- Rust
# (id, task, tests). Each test is the body of a #[test] fn; the answer is in scope (`use super::*`).

RUST_TASKS = [
    ("semver",
     """Two functions for an auto-updater:

`pub fn parse_version(s: &str) -> Option<(u64, u64, u64)>` parses "MAJOR.MINOR.PATCH" with an optional leading
`v` (so "v0.10.11" works). Surrounding whitespace is ignored. Anything else - missing or extra parts, empty
parts, non-digits, a pre-release suffix like "1.2.3-beta" - is None.

`pub fn is_newer(candidate: &str, current: &str) -> bool` is true only when both parse and candidate is a
strictly higher version than current (compare numerically: 0.10.0 is newer than 0.9.9).""",
     ['assert_eq!(parse_version("1.2.3"), Some((1, 2, 3)));',
      'assert_eq!(parse_version(" v0.10.11 "), Some((0, 10, 11)));',
      'assert_eq!(parse_version("1.2"), None); assert_eq!(parse_version("1.2.3.4"), None);',
      'assert_eq!(parse_version("1..3"), None); assert_eq!(parse_version("1.2.x"), None); assert_eq!(parse_version(""), None);',
      'assert_eq!(parse_version("1.2.3-beta"), None); assert_eq!(parse_version("V1.2.3"), None);',
      'assert!(is_newer("0.10.0", "0.9.9")); assert!(is_newer("v1.0.0", "0.99.99"));',
      'assert!(!is_newer("1.2.3", "1.2.3")); assert!(!is_newer("1.2.2", "1.2.3"));',
      'assert!(!is_newer("garbage", "1.0.0")); assert!(!is_newer("2.0.0", "garbage"));']),
    ("lru",
     """A least-recently-used cache:

```
pub struct Lru<V> { /* your fields */ }
impl<V> Lru<V> {
    pub fn new(capacity: usize) -> Self;
    pub fn get(&mut self, key: &str) -> Option<&V>;   // a hit makes the key most recently used
    pub fn put(&mut self, key: &str, value: V);       // insert or replace; replacing also makes it most recent
    pub fn len(&self) -> usize;
}
```

When a put would go over capacity, evict the least recently used entry first. A capacity of 0 stores
nothing. V has no trait bounds (don't require Clone).""",
     ['let mut c = Lru::new(2); c.put("a", 1); c.put("b", 2); assert_eq!(c.get("a"), Some(&1)); assert_eq!(c.len(), 2);',
      'let mut c = Lru::new(2); c.put("a", 1); c.put("b", 2); c.put("c", 3); assert_eq!(c.get("a"), None); assert_eq!(c.get("c"), Some(&3)); assert_eq!(c.len(), 2);',
      'let mut c = Lru::new(2); c.put("a", 1); c.put("b", 2); c.get("a"); c.put("c", 3); assert_eq!(c.get("b"), None); assert_eq!(c.get("a"), Some(&1));',
      'let mut c = Lru::new(2); c.put("a", 1); c.put("b", 2); c.put("a", 10); c.put("c", 3); assert_eq!(c.get("b"), None); assert_eq!(c.get("a"), Some(&10)); assert_eq!(c.len(), 2);',
      'let mut c: Lru<String> = Lru::new(0); c.put("a", "x".to_string()); assert_eq!(c.get("a"), None); assert_eq!(c.len(), 0);',
      'struct NoClone(u8); let mut c = Lru::new(1); c.put("a", NoClone(1)); c.put("b", NoClone(2)); assert!(c.get("a").is_none()); assert_eq!(c.get("b").map(|v| v.0), Some(2));',
      'let mut c = Lru::new(3); for (i, k) in ["a", "b", "c", "d", "e"].iter().enumerate() { c.put(k, i); } assert_eq!(c.len(), 3); assert!(c.get("b").is_none()); assert_eq!(c.get("e"), Some(&4));']),
    ("edit_eol",
     """An agent edits files by sending the exact text to find and its replacement, always with `\\n` line
endings - but files on Windows often use `\\r\\n`.

`pub fn replace_once(text: &str, find: &str, replace: &str) -> Result<String, String>`

Replace exactly one occurrence of `find` in `text`. Match `find` against the text regardless of whether the
text uses `\\n` or `\\r\\n`. The result must keep the text's own line endings: if the text uses `\\r\\n`, newlines in
`replace` become `\\r\\n` too. Return Err when `find` is empty, not present, or present more than once (the
error message should say which: contain "not found" or "more than once"/"empty").""",
     ['assert_eq!(replace_once("a\\nb\\nc\\n", "b\\n", "B\\n").unwrap(), "a\\nB\\nc\\n");',
      'assert_eq!(replace_once("a\\r\\nb\\r\\nc\\r\\n", "a\\nb", "x\\ny").unwrap(), "x\\r\\ny\\r\\nc\\r\\n");',
      'assert_eq!(replace_once("fn a() {}\\r\\n", "a()", "b()").unwrap(), "fn b() {}\\r\\n");',
      'let e = replace_once("abc", "z", "y").unwrap_err(); assert!(e.contains("not found"), "{e}");',
      'let e = replace_once("x x", "x", "y").unwrap_err(); assert!(e.contains("more than once"), "{e}");',
      'assert!(replace_once("abc", "", "y").is_err());',
      'assert_eq!(replace_once("one\\r\\ntwo", "one\\ntwo", "three").unwrap(), "three");',
      'assert_eq!(replace_once("héllo wörld", "wörld", "world").unwrap(), "héllo world");']),
    ("safe_join",
     """Agents pass workspace-relative paths; they must never escape the workspace.

`pub fn safe_join(root: &std::path::Path, rel: &str) -> Option<std::path::PathBuf>`

Join `rel` onto `root` lexically (don't touch the file system). Accept both `/` and `\\` as separators. Drop
`.` and empty segments, and let `..` remove the previous segment. Return None if `rel` is absolute (starts
with `/` or `\\`, or has a drive letter like `C:`), or if a `..` would climb above `root`. An empty `rel` or
"." is the root itself.""",
     ['use std::path::Path; assert_eq!(safe_join(Path::new("/w"), "src/main.rs"), Some(Path::new("/w").join("src").join("main.rs")));',
      'use std::path::Path; assert_eq!(safe_join(Path::new("/w"), "src\\\\lib.rs"), Some(Path::new("/w").join("src").join("lib.rs")));',
      'use std::path::Path; assert_eq!(safe_join(Path::new("/w"), "./a//b/../c"), Some(Path::new("/w").join("a").join("c")));',
      'use std::path::Path; assert_eq!(safe_join(Path::new("/w"), "../etc/passwd"), None); assert_eq!(safe_join(Path::new("/w"), "a/../../b"), None);',
      'use std::path::Path; assert_eq!(safe_join(Path::new("/w"), "/etc/passwd"), None); assert_eq!(safe_join(Path::new("/w"), "\\\\x"), None);',
      'use std::path::Path; assert_eq!(safe_join(Path::new("/w"), "C:\\\\Windows"), None); assert_eq!(safe_join(Path::new("/w"), "c:x"), None);',
      'use std::path::Path; assert_eq!(safe_join(Path::new("/w"), ""), Some(Path::new("/w").to_path_buf())); assert_eq!(safe_join(Path::new("/w"), "."), Some(Path::new("/w").to_path_buf()));',
      'use std::path::Path; assert_eq!(safe_join(Path::new("/w"), "a/.."), Some(Path::new("/w").to_path_buf()));']),
    ("job_queue",
     """A GPU job queue:

```
pub struct JobQueue { /* your fields */ }
impl JobQueue {
    pub fn new() -> Self;
    pub fn push(&mut self, name: &str, priority: u8) -> u64;  // returns a new id: 1, 2, 3, ...
    pub fn next(&mut self) -> Option<(u64, String)>;          // removes and returns the next job
    pub fn cancel(&mut self, id: u64) -> bool;                // true if a waiting job was removed
    pub fn len(&self) -> usize;
}
```

`next` takes the highest priority first; jobs of equal priority come out in the order they were pushed.
Ids are never reused, even after a job is taken or cancelled.""",
     ['let mut q = JobQueue::new(); assert_eq!(q.push("a", 1), 1); assert_eq!(q.push("b", 1), 2); assert_eq!(q.next(), Some((1, "a".into()))); assert_eq!(q.next(), Some((2, "b".into()))); assert_eq!(q.next(), None);',
      'let mut q = JobQueue::new(); q.push("low", 1); q.push("high", 9); q.push("mid", 5); assert_eq!(q.next().unwrap().1, "high"); assert_eq!(q.next().unwrap().1, "mid"); assert_eq!(q.next().unwrap().1, "low");',
      'let mut q = JobQueue::new(); q.push("a", 5); q.push("b", 9); q.push("c", 5); q.push("d", 9); let order: Vec<String> = std::iter::from_fn(|| q.next().map(|j| j.1)).collect(); assert_eq!(order, ["b", "d", "a", "c"]);',
      'let mut q = JobQueue::new(); let a = q.push("a", 1); q.push("b", 1); assert!(q.cancel(a)); assert!(!q.cancel(a)); assert_eq!(q.len(), 1); assert_eq!(q.next().unwrap().1, "b");',
      'let mut q = JobQueue::new(); assert!(!q.cancel(42)); let a = q.push("a", 1); q.next(); assert!(!q.cancel(a)); assert_eq!(q.push("b", 1), 2);',
      'let mut q = JobQueue::new(); for i in 0..1000u32 { q.push(&i.to_string(), (i % 3) as u8); } assert_eq!(q.len(), 1000); assert_eq!(q.next().unwrap().1, "2"); assert_eq!(q.next().unwrap().1, "5");']),
    ("split_args",
     """Split a command line into arguments, shell-style:

`pub fn split_args(line: &str) -> Result<Vec<String>, String>`

Arguments are separated by runs of whitespace. Double quotes and single quotes group text (the quotes
themselves are removed), and quoted text can join unquoted text in one argument (`a"b c"d` is `ab cd`).
Inside double quotes, and outside quotes, a backslash escapes the next character; inside single quotes a
backslash is literal. `""` is an empty argument. An unclosed quote, or a trailing lone backslash, is an Err.""",
     ['assert_eq!(split_args("cargo test --release").unwrap(), ["cargo", "test", "--release"]);',
      'assert_eq!(split_args("  a \\t  b  ").unwrap(), ["a", "b"]); assert!(split_args("").unwrap().is_empty());',
      'assert_eq!(split_args(r#"git commit -m "fix the bug""#).unwrap(), ["git", "commit", "-m", "fix the bug"]);',
      'assert_eq!(split_args(r#"a"b c"d"#).unwrap(), ["ab cd"]); assert_eq!(split_args(r#"x "" y"#).unwrap(), ["x", "", "y"]);',
      'assert_eq!(split_args(r#"echo \'it\\s\' "say \\"hi\\"""#).unwrap(), ["echo", r"it\\s", r#"say "hi""#]);',
      'assert_eq!(split_args(r"a\\ b c").unwrap(), ["a b", "c"]);',
      'assert!(split_args(r#"echo "open"#).is_err()); assert!(split_args("echo \'open").is_err()); assert!(split_args("trailing\\\\").is_err());']),
]

# --------------------------------------------------------------------------- TypeScript
# (id, task, tests). Each test is a statement; `eq(actual, expected)` is a deep-equality assertion.

TS_TASKS = [
    ("format_bytes",
     """`function formatBytes(n: number): string` for download sizes. Use 1024 steps and the units B, KB, MB,
GB, TB. Under 1024 bytes show a whole number ("0 B", "1023 B"); from KB up show one decimal ("1.0 KB",
"1.5 MB"), rounding to the nearest tenth. If rounding would show 1024.0 of a unit, use the next unit instead
("1.0 MB" for 1048575). Stop at TB ("2048.0 TB"). Negative or non-finite input returns "0 B".""",
     ['eq(formatBytes(0), "0 B")', 'eq(formatBytes(1023), "1023 B")', 'eq(formatBytes(1024), "1.0 KB")',
      'eq(formatBytes(1536 * 1024), "1.5 MB")', 'eq(formatBytes(8.3 * 1024 ** 3), "8.3 GB")',
      'eq(formatBytes(1048575), "1.0 MB")', 'eq(formatBytes(2048 * 1024 ** 4), "2048.0 TB")',
      'eq(formatBytes(-5), "0 B"); eq(formatBytes(NaN), "0 B"); eq(formatBytes(Infinity), "0 B")']),
    ("parse_duration",
     """`function parseDuration(s: string): number | null` returns seconds for strings like "90s", "2m",
"1h30m", "1h 30m 15s", "45" (a bare number is seconds). Units are h, m, s, each at most once and in that order;
whitespace between parts is allowed; case-insensitive. Numbers are non-negative integers. Return null for
anything else ("", "1x", "30m1h", "1h1h", "-5", "1.5h").""",
     ['eq(parseDuration("90s"), 90)', 'eq(parseDuration("2m"), 120)', 'eq(parseDuration("1h30m"), 5400)',
      'eq(parseDuration("1h 30m 15s"), 5415)', 'eq(parseDuration("45"), 45); eq(parseDuration("0"), 0)',
      'eq(parseDuration("2H 5M"), 7500)',
      'for (const bad of ["", "1x", "30m1h", "1h1h", "-5", "1.5h", "h", " "]) eq(parseDuration(bad), null)']),
    ("tabs_reducer",
     """Editor tabs, as a pure reducer:

```
type Tab = { id: string; title: string };
type State = { tabs: Tab[]; active: string | null };
type Action =
  | { type: "open"; tab: Tab }          // add at the end and activate; if the id is already open, just activate it
  | { type: "close"; id: string }       // remove; if it was active, activate the tab now at its position
                                        //   (the one that was to its right), else the new last tab, else null
  | { type: "move"; from: number; to: number };  // move the tab at index `from` to index `to`
function reduce(state: State, action: Action): State
```

Never mutate the input state or its arrays. Unknown ids and out-of-range indexes return the state unchanged
(the same object).""",
     ['const s = reduce({ tabs: [], active: null }, { type: "open", tab: { id: "a", title: "A" } }); eq(s, { tabs: [{ id: "a", title: "A" }], active: "a" })',
      'const s0 = { tabs: [{ id: "a", title: "A" }, { id: "b", title: "B" }], active: "b" }; eq(reduce(s0, { type: "open", tab: { id: "a", title: "A2" } }), { tabs: s0.tabs, active: "a" })',
      'const s0 = { tabs: [{ id: "a", title: "A" }, { id: "b", title: "B" }, { id: "c", title: "C" }], active: "b" }; eq(reduce(s0, { type: "close", id: "b" }).active, "c")',
      'const s0 = { tabs: [{ id: "a", title: "A" }, { id: "b", title: "B" }], active: "b" }; eq(reduce(s0, { type: "close", id: "b" }), { tabs: [{ id: "a", title: "A" }], active: "a" })',
      'const s0 = { tabs: [{ id: "a", title: "A" }, { id: "b", title: "B" }], active: "a" }; eq(reduce(s0, { type: "close", id: "b" }).active, "a"); eq(reduce({ tabs: [{ id: "a", title: "A" }], active: "a" }, { type: "close", id: "a" }), { tabs: [], active: null })',
      'const s0 = { tabs: [{ id: "a", title: "A" }, { id: "b", title: "B" }, { id: "c", title: "C" }], active: "a" }; eq(reduce(s0, { type: "move", from: 0, to: 2 }).tabs.map(t => t.id), ["b", "c", "a"])',
      'const s0 = Object.freeze({ tabs: Object.freeze([Object.freeze({ id: "a", title: "A" })]), active: "a" }); reduce(s0, { type: "open", tab: { id: "b", title: "B" } }); reduce(s0, { type: "close", id: "a" }); reduce(s0, { type: "move", from: 0, to: 0 }); eq(s0.tabs.length, 1)',
      'const s0 = { tabs: [{ id: "a", title: "A" }], active: "a" }; if (reduce(s0, { type: "close", id: "zz" }) !== s0) throw new Error("unknown id should return the same state"); if (reduce(s0, { type: "move", from: 0, to: 5 }) !== s0) throw new Error("bad index should return the same state")']),
    ("deep_merge",
     """Settings are saved as a partial patch over the defaults:

`function deepMerge<T>(base: T, patch: unknown): T`

Plain objects merge recursively. Arrays and every other value in the patch replace what's in base. A key
whose patch value is `undefined` is ignored; `null` is a real value and replaces. Keys only in the patch are
added. Neither argument is mutated, and the result doesn't share objects with the patch (later edits to the
patch must not change the result). A patch that isn't a plain object (null, an array, a number) is ignored,
giving a result equal to base.""",
     ['eq(deepMerge({ a: 1, b: 2 }, { b: 3 }), { a: 1, b: 3 })',
      'eq(deepMerge({ ui: { theme: "dark", font: 14 }, gpu: true }, { ui: { font: 16 } }), { ui: { theme: "dark", font: 16 }, gpu: true })',
      'eq(deepMerge({ list: [1, 2, 3] }, { list: [9] }), { list: [9] })',
      'eq(deepMerge({ a: 1, b: { c: 2 } }, { a: undefined, b: null }), { a: 1, b: null })',
      'eq(deepMerge({ a: 1 }, { z: { y: 1 } }), { a: 1, z: { y: 1 } })',
      'const base = { ui: { font: 14 } }; const patch = { ui: { font: 16 } }; const r = deepMerge(base, patch); eq(base, { ui: { font: 14 } }); eq(patch, { ui: { font: 16 } }); if (r.ui === patch.ui) throw new Error("shares the patch object")',
      'eq(deepMerge({ a: 1 }, null), { a: 1 }); eq(deepMerge({ a: 1 }, [1, 2]), { a: 1 }); eq(deepMerge({ a: 1 }, 5), { a: 1 })']),
    ("emitter",
     """A small typed event emitter:

```
function createEmitter(): {
  on(event: string, fn: (...args: any[]) => void): () => void;    // returns an unsubscribe function
  once(event: string, fn: (...args: any[]) => void): () => void;  // runs at most once; also unsubscribable
  emit(event: string, ...args: any[]): number;                    // calls listeners in the order added;
                                                                  //   returns how many were called
}
```

Subscribing the same function twice means it is called twice. Listeners added or removed while an emit is
running don't affect that emit (it uses the listeners present when it started). If a listener throws, the
others still run and emit then rethrows the first error.""",
     ['const e = createEmitter(); const got: any[] = []; e.on("x", (a, b) => got.push([a, b])); eq(e.emit("x", 1, 2), 1); eq(got, [[1, 2]])',
      'const e = createEmitter(); const got: number[] = []; e.on("x", () => got.push(1)); e.on("x", () => got.push(2)); e.emit("x"); eq(got, [1, 2]); eq(e.emit("nothing"), 0)',
      'const e = createEmitter(); let n = 0; const off = e.on("x", () => n++); e.emit("x"); off(); e.emit("x"); eq(n, 1); off()',
      'const e = createEmitter(); let n = 0; e.once("x", () => n++); e.emit("x"); e.emit("x"); eq(n, 1); let m = 0; const off = e.once("y", () => m++); off(); e.emit("y"); eq(m, 0)',
      'const e = createEmitter(); let n = 0; const f = () => n++; e.on("x", f); e.on("x", f); eq(e.emit("x"), 2); eq(n, 2)',
      'const e = createEmitter(); const got: string[] = []; e.on("x", () => { got.push("a"); e.on("x", () => got.push("late")); }); e.emit("x"); eq(got, ["a"])',
      'const e = createEmitter(); const got: string[] = []; let off2 = () => {}; e.on("x", () => { got.push("a"); off2(); }); off2 = e.on("x", () => got.push("b")); e.emit("x"); eq(got, ["a", "b"])',
      'const e = createEmitter(); let ran = false; e.on("x", () => { throw new Error("boom"); }); e.on("x", () => { ran = true; }); let caught = ""; try { e.emit("x"); } catch (err: any) { caught = err.message; } eq([ran, caught], [true, "boom"])']),
    ("group_models",
     """The model picker groups Ollama model names like "qwen3:14b":

`function groupModels(names: string[]): Record<string, string[]>`

Group by the family (the part before ":"); the value lists the tags (the part after ":"; a name without one
has the tag "latest"). Drop duplicates. Within a family, tags without a size come first, alphabetically
("latest" is just another tag); then tags that start with a size like "14b", "0.6b" or "30b-a3b", smallest
size first (compare the number, so "8b" < "14b"; ties alphabetically). Families are case-sensitive.""",
     ['eq(groupModels(["qwen3:14b", "qwen3:8b", "llama3.2"]), { qwen3: ["8b", "14b"], "llama3.2": ["latest"] })',
      'eq(groupModels(["qwen3:0.6b", "qwen3:32b", "qwen3:4b"]).qwen3, ["0.6b", "4b", "32b"])',
      'eq(groupModels(["m:latest", "m:7b", "m:instruct", "m"]).m, ["instruct", "latest", "7b"])',
      'eq(groupModels(["q:30b-a3b", "q:30b", "q:14b-q8_0", "q:14b"]).q, ["14b", "14b-q8_0", "30b", "30b-a3b"])',
      'eq(groupModels([]), {}); eq(Object.keys(groupModels(["B:1b", "b:1b"])).sort(), ["B", "b"])',
      'eq(groupModels(["hf.co/org/model:Q4_K_M", "hf.co/org/model:Q8_0"])["hf.co/org/model"], ["Q4_K_M", "Q8_0"])']),
]

TS_HARNESS_HEAD = """const __assert = (await import("node:assert")).strict;
function eq(actual: unknown, expected: unknown) { __assert.deepStrictEqual(actual, expected); }
"""
TS_HARNESS_TAIL = """
const __tests: Array<() => void> = [{tests}];
let __passed = 0;
for (const [i, t] of __tests.entries()) {{
  try {{ t(); __passed++; }} catch (e) {{ console.log("FAIL " + (i + 1) + ": " + String((e as any)?.message ?? e).split("\\n")[0]); }}
}}
console.log("LB_RESULT " + __passed + "/" + __tests.length);
"""

# --------------------------------------------------------------------------- toolchains

_TOOLS: dict[str, bool] = {}


def have(tool: str) -> bool:
    """rustc, or a Node new enough to strip TypeScript types (22.6+)."""
    if tool not in _TOOLS:
        ok = shutil.which(tool) is not None
        if ok and tool == "node":
            try:
                v = subprocess.run(["node", "--version"], capture_output=True, text=True, timeout=10).stdout
                major, minor = (int(x) for x in v.strip().lstrip("v").split(".")[:2])
                ok = (major, minor) >= (22, 6)
            except (OSError, ValueError, subprocess.SubprocessError):
                ok = False
        _TOOLS[tool] = ok
    return _TOOLS[tool]


def _code(text: str, lang: str) -> str:
    blocks = code_blocks(text, "typescript") if lang == "ts" else []
    blocks = blocks or code_blocks(text, lang)
    return max(blocks, key=len) if blocks else text


def rust_program(answer: str, tests: list[str]) -> str:
    body = "\n".join(f"    #[test]\n    fn t{i + 1:02}() {{ {t} }}" for i, t in enumerate(tests))
    return (f"#![allow(dead_code, unused_imports, unused_variables, unused_mut)]\n{answer}\n\n"
            f"#[cfg(test)]\nmod lb_tests {{\n    use super::*;\n{body}\n}}\n")


def ts_program(answer: str, tests: list[str]) -> str:
    answer = re.sub(r"^\s*export\s+(default\s+)?", "", answer, flags=re.M)  # asked not to, but harmless
    fns = ", ".join(f"() => {{ {t}; }}" for t in tests)
    return TS_HARNESS_HEAD + answer + "\n" + TS_HARNESS_TAIL.format(tests=fns)


def grade_rust(answer: str, tests: list[str]) -> Grade:
    exe = "prog.exe" if sys.platform == "win32" else "prog"
    rc, out, err = run_programs({"prog.rs": rust_program(answer, tests)},
                                [["rustc", "--edition", "2021", "--test", "-C", "debuginfo=0", "-o", exe, "prog.rs"],
                                 [f"./{exe}" if sys.platform != "win32" else exe, "lb_tests::", "--test-threads=1"]],
                                timeout=120)
    if rc is None:
        return Grade(0.0, "timeout")
    m = re.search(r"test result: \w+\. (\d+) passed; (\d+) failed", out)
    if not m:  # didn't compile
        first = next((l for l in err.splitlines() if l.startswith("error")), err.strip()[:200] or f"exit {rc}")
        return Grade(0.0, f"compile: {first[:200]}")
    passed, failed = int(m.group(1)), int(m.group(2))
    total = passed + failed or len(tests)
    return Grade(passed / len(tests), f"{passed}/{len(tests)} tests" + ("" if total == len(tests) else " (some crashed)"))


def grade_ts(answer: str, tests: list[str]) -> Grade:
    rc, out, err = run_programs({"prog.mts": ts_program(answer, tests)},
                                [["node", "--experimental-strip-types", "--no-warnings", "prog.mts"]], timeout=30)
    if rc is None:
        return Grade(0.0, "timeout")
    m = re.search(r"LB_RESULT (\d+)/(\d+)", out)
    if not m:
        lines = [l for l in err.splitlines() if l.strip()]
        detail = next((l for l in lines if "Error" in l), lines[-1] if lines else f"exit {rc}")
        return Grade(0.0, f"error: {detail.strip()[:200]}")
    passed = int(m.group(1))
    fails = [l for l in out.splitlines() if l.startswith("FAIL")]
    return Grade(passed / len(tests), f"{passed}/{len(tests)} tests" + (f"; {fails[0][:150]}" if fails else ""))


class Stack(Category):
    name = "stack"
    title = "Stack (Rust + TS)"
    description = "Tauri-app code in Rust and TypeScript, compiled and run against unit tests."
    method = (f"{len(RUST_TASKS)} Rust tasks built with rustc --test and {len(TS_TASKS)} TypeScript tasks run on "
              "Node; score = share of unit tests passed (0 if it doesn't compile)")
    max_tokens = 3072
    uses_limit = False

    def load(self):
        items = []
        langs = [("rust", RUST_TASKS, RUST_PROMPT, "rustc"), ("ts", TS_TASKS, TS_PROMPT, "node")]
        for lang, tasks, prompt, tool in langs:
            if not have(tool):
                continue
            for tid, task, tests in tasks:
                items.append(Item(f"{lang}/{tid}", [{"role": "system", "content": SYSTEM},
                                                    {"role": "user", "content": prompt.format(task=task)}],
                                  {"lang": lang, "tests": tests}))
        return items

    def grade(self, item, text):
        p = item.payload
        if p["lang"] == "rust":
            return grade_rust(_code(text, "rust"), p["tests"])
        return grade_ts(_code(text, "ts"), p["tests"])

    def summarize(self, records):
        missing = [t for t in ("rustc", "node") if not have(t)]
        return {"skipped": ", ".join(f"{t} not found" for t in missing)} if missing else {}
