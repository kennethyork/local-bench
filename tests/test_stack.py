"""The Stack category is solvable and fair: reference answers pass every test, broken or partial answers
score less, and answers that don't compile score 0. Needs rustc and Node 22.6+ (skipped otherwise)."""
import unittest

from localbench.categories import Item, registry
from localbench.categories import stack
from tests.stack_refs import RUST, TS

HAVE_RUST, HAVE_NODE = stack.have("rustc"), stack.have("node")


@unittest.skipUnless(HAVE_RUST, "rustc not installed")
class RustTasks(unittest.TestCase):
    def test_reference_answers_pass_everything(self):
        for tid, _, tests in stack.RUST_TASKS:
            with self.subTest(tid):
                g = stack.grade_rust(RUST[tid], tests)
                self.assertEqual(g.score, 1.0, g.detail)

    def test_partial_answer_gets_partial_credit(self):
        loose = RUST["semver"].replace("if p.is_empty() || !p.bytes().all(|b| b.is_ascii_digit()) { return None; }", "")
        loose = loose.replace("let s = s.strip_prefix('v').unwrap_or(s);", "let s = s.trim_start_matches(|c| c == 'v' || c == 'V');")
        g = stack.grade_rust(loose, stack.RUST_TASKS[0][2])
        self.assertTrue(0 < g.score < 1, g.detail)

    def test_no_compile_scores_zero(self):
        g = stack.grade_rust("pub fn parse_version(s: &str) -> Option<(u64, u64, u64)> { s.nope() }", stack.RUST_TASKS[0][2])
        self.assertEqual(g.score, 0.0)
        self.assertTrue(g.detail.startswith("compile"), g.detail)

    def test_models_own_tests_and_main_dont_count(self):
        extra = RUST["lru"] + "\nfn main() {}\n#[cfg(test)] mod tests { #[test] fn mine() { assert!(false); } }\n"
        self.assertEqual(stack.grade_rust(extra, stack.RUST_TASKS[1][2]).score, 1.0)

    def test_panicking_answer_fails_only_its_tests(self):
        half = RUST["job_queue"].replace("pub fn cancel(&mut self, id: u64) -> bool {", "pub fn cancel(&mut self, id: u64) -> bool { panic!(\"todo\");")
        g = stack.grade_rust(half, stack.RUST_TASKS[4][2])
        self.assertTrue(0.5 <= g.score < 1, g.detail)


@unittest.skipUnless(HAVE_NODE, "Node 22.6+ not installed")
class TsTasks(unittest.TestCase):
    def test_reference_answers_pass_everything(self):
        for tid, _, tests in stack.TS_TASKS:
            with self.subTest(tid):
                g = stack.grade_ts(TS[tid], tests)
                self.assertEqual(g.score, 1.0, g.detail)

    def test_mutating_reducer_loses_points(self):
        bad = TS["tabs_reducer"].replace("return { tabs: [...state.tabs, action.tab], active: action.tab.id };",
                                         "state.tabs.push(action.tab); state.active = action.tab.id; return state;")
        g = stack.grade_ts(bad, stack.TS_TASKS[2][2])
        self.assertTrue(g.score < 1, g.detail)

    def test_syntax_error_scores_zero(self):
        g = stack.grade_ts("function formatBytes(n: number): string { return (", stack.TS_TASKS[0][2])
        self.assertEqual(g.score, 0.0)

    def test_exports_and_typescript_fence_are_fine(self):
        cat = registry()["stack"]
        it = Item("ts/format_bytes", [], {"lang": "ts", "tests": stack.TS_TASKS[0][2]})
        text = "Here you go:\n```typescript\nexport " + TS["format_bytes"].strip() + "\n```\n"
        self.assertEqual(cat.grade(it, text).score, 1.0)

    def test_infinite_loop_times_out(self):
        g = stack.grade_ts("function formatBytes(n: number): string { while (true) {} }", stack.TS_TASKS[0][2][:1])
        self.assertEqual(g.score, 0.0)


class Loading(unittest.TestCase):
    def test_items_match_installed_toolchains(self):
        items = registry()["stack"].load()
        self.assertEqual(sum(i.id.startswith("rust/") for i in items), len(stack.RUST_TASKS) if HAVE_RUST else 0)
        self.assertEqual(sum(i.id.startswith("ts/") for i in items), len(stack.TS_TASKS) if HAVE_NODE else 0)


if __name__ == "__main__":
    unittest.main()
