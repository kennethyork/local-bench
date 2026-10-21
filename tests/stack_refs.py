"""Reference answers for the Stack category (tests check each scores 1.0)."""
RUST = {
"semver": r'''
pub fn parse_version(s: &str) -> Option<(u64, u64, u64)> {
    let s = s.trim();
    let s = s.strip_prefix('v').unwrap_or(s);
    let parts: Vec<&str> = s.split('.').collect();
    if parts.len() != 3 { return None; }
    let mut n = [0u64; 3];
    for (i, p) in parts.iter().enumerate() {
        if p.is_empty() || !p.bytes().all(|b| b.is_ascii_digit()) { return None; }
        n[i] = p.parse().ok()?;
    }
    Some((n[0], n[1], n[2]))
}
pub fn is_newer(candidate: &str, current: &str) -> bool {
    match (parse_version(candidate), parse_version(current)) { (Some(a), Some(b)) => a > b, _ => false }
}''',
"lru": r'''
use std::collections::HashMap;
pub struct Lru<V> { cap: usize, tick: u64, map: HashMap<String, (u64, V)> }
impl<V> Lru<V> {
    pub fn new(capacity: usize) -> Self { Lru { cap: capacity, tick: 0, map: HashMap::new() } }
    pub fn get(&mut self, key: &str) -> Option<&V> {
        self.tick += 1;
        let t = self.tick;
        match self.map.get_mut(key) { Some(e) => { e.0 = t; Some(&e.1) } None => None }
    }
    pub fn put(&mut self, key: &str, value: V) {
        if self.cap == 0 { return; }
        self.tick += 1;
        if !self.map.contains_key(key) && self.map.len() >= self.cap {
            let old = self.map.iter().min_by_key(|(_, e)| e.0).map(|(k, _)| k.clone()).unwrap();
            self.map.remove(&old);
        }
        self.map.insert(key.to_string(), (self.tick, value));
    }
    pub fn len(&self) -> usize { self.map.len() }
}''',
"edit_eol": r'''
pub fn replace_once(text: &str, find: &str, replace: &str) -> Result<String, String> {
    if find.is_empty() { return Err("find is empty".into()); }
    let crlf = text.contains("\r\n");
    let norm = text.replace("\r\n", "\n");
    let find = find.replace("\r\n", "\n");
    let n = norm.matches(find.as_str()).count();
    if n == 0 { return Err("not found".into()); }
    if n > 1 { return Err("found more than once".into()); }
    let out = norm.replacen(find.as_str(), &replace.replace("\r\n", "\n"), 1);
    Ok(if crlf { out.replace('\n', "\r\n") } else { out })
}''',
"safe_join": r'''
use std::path::{Path, PathBuf};
pub fn safe_join(root: &Path, rel: &str) -> Option<PathBuf> {
    if rel.starts_with('/') || rel.starts_with('\\') { return None; }
    let b = rel.as_bytes();
    if b.len() >= 2 && b[1] == b':' && b[0].is_ascii_alphabetic() { return None; }
    let mut parts: Vec<&str> = Vec::new();
    for seg in rel.split(|c| c == '/' || c == '\\') {
        match seg { "" | "." => {}, ".." => { parts.pop()?; } s => parts.push(s) }
    }
    let mut p = root.to_path_buf();
    for s in parts { p.push(s); }
    Some(p)
}''',
"job_queue": r'''
pub struct JobQueue { next_id: u64, jobs: Vec<(u8, u64, String)> }
impl JobQueue {
    pub fn new() -> Self { JobQueue { next_id: 1, jobs: Vec::new() } }
    pub fn push(&mut self, name: &str, priority: u8) -> u64 {
        let id = self.next_id; self.next_id += 1; self.jobs.push((priority, id, name.to_string())); id
    }
    pub fn next(&mut self) -> Option<(u64, String)> {
        let i = (0..self.jobs.len()).max_by(|&a, &b| self.jobs[a].0.cmp(&self.jobs[b].0).then(self.jobs[b].1.cmp(&self.jobs[a].1)))?;
        let (_, id, name) = self.jobs.remove(i); Some((id, name))
    }
    pub fn cancel(&mut self, id: u64) -> bool {
        let n = self.jobs.len(); self.jobs.retain(|j| j.1 != id); self.jobs.len() != n
    }
    pub fn len(&self) -> usize { self.jobs.len() }
}''',
"split_args": r'''
pub fn split_args(line: &str) -> Result<Vec<String>, String> {
    let mut out = Vec::new(); let mut cur = String::new(); let mut has = false;
    let mut it = line.chars();
    while let Some(c) = it.next() {
        match c {
            c if c.is_whitespace() => { if has { out.push(std::mem::take(&mut cur)); has = false; } }
            '\\' => { cur.push(it.next().ok_or("trailing backslash")?); has = true; }
            '\'' => { has = true; loop { match it.next() { Some('\'') => break, Some(c) => cur.push(c), None => return Err("unclosed quote".into()) } } }
            '"' => { has = true; loop { match it.next() {
                Some('"') => break,
                Some('\\') => cur.push(it.next().ok_or("trailing backslash")?),
                Some(c) => cur.push(c),
                None => return Err("unclosed quote".into()) } } }
            c => { cur.push(c); has = true; }
        }
    }
    if has { out.push(cur); }
    Ok(out)
}''',
}

TS = {
"format_bytes": '''
function formatBytes(n: number): string {
  if (!Number.isFinite(n) || n < 0) return "0 B";
  if (n < 1024) return `${Math.round(n)} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let v = n / 1024, i = 0;
  while (i < units.length - 1 && Math.round(v * 10) / 10 >= 1024) { v /= 1024; i++; }
  return `${v.toFixed(1)} ${units[i]}`;
}''',
"parse_duration": '''
function parseDuration(s: string): number | null {
  const t = s.trim().toLowerCase();
  if (/^\\d+$/.test(t)) return Number(t);
  const m = /^(?:(\\d+)h)?\\s*(?:(\\d+)m)?\\s*(?:(\\d+)s)?$/.exec(t);
  if (!t || !m || (!m[1] && !m[2] && !m[3])) return null;
  return Number(m[1] ?? 0) * 3600 + Number(m[2] ?? 0) * 60 + Number(m[3] ?? 0);
}''',
"tabs_reducer": '''
type Tab = { id: string; title: string };
type State = { tabs: Tab[]; active: string | null };
type Action = { type: "open"; tab: Tab } | { type: "close"; id: string } | { type: "move"; from: number; to: number };
function reduce(state: State, action: Action): State {
  if (action.type === "open") {
    if (state.tabs.some(t => t.id === action.tab.id)) return { ...state, active: action.tab.id };
    return { tabs: [...state.tabs, action.tab], active: action.tab.id };
  }
  if (action.type === "close") {
    const i = state.tabs.findIndex(t => t.id === action.id);
    if (i < 0) return state;
    const tabs = state.tabs.filter(t => t.id !== action.id);
    let active = state.active;
    if (active === action.id) active = tabs[i]?.id ?? tabs[tabs.length - 1]?.id ?? null;
    return { tabs, active };
  }
  const { from, to } = action, n = state.tabs.length;
  if (from < 0 || from >= n || to < 0 || to >= n) return state;
  const tabs = [...state.tabs];
  const [t] = tabs.splice(from, 1);
  tabs.splice(to, 0, t);
  return { ...state, tabs };
}''',
"deep_merge": '''
function isPlain(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}
function clone(v: unknown): unknown { return v === undefined ? v : structuredClone(v); }
function deepMerge<T>(base: T, patch: unknown): T {
  if (!isPlain(patch) || !isPlain(base)) return base;
  const out: Record<string, unknown> = { ...base };
  for (const [k, v] of Object.entries(patch)) {
    if (v === undefined) continue;
    out[k] = isPlain(v) && isPlain(out[k]) ? deepMerge(out[k], v) : clone(v);
  }
  return out as T;
}''',
"emitter": '''
function createEmitter() {
  const map = new Map<string, Array<{ fn: (...a: any[]) => void }>>();
  function add(event: string, fn: (...a: any[]) => void, once: boolean) {
    const entry = { fn: once ? (...a: any[]) => { off(); fn(...a); } : fn };
    const list = map.get(event) ?? [];
    list.push(entry); map.set(event, list);
    function off() { const l = map.get(event); if (l) { const i = l.indexOf(entry); if (i >= 0) l.splice(i, 1); } }
    return off;
  }
  return {
    on: (e: string, fn: (...a: any[]) => void) => add(e, fn, false),
    once: (e: string, fn: (...a: any[]) => void) => add(e, fn, true),
    emit(e: string, ...args: any[]): number {
      const list = [...(map.get(e) ?? [])];
      let first: unknown, failed = false;
      for (const l of list) { try { l.fn(...args); } catch (err) { if (!failed) { failed = true; first = err; } } }
      if (failed) throw first;
      return list.length;
    },
  };
}''',
"group_models": '''
function groupModels(names: string[]): Record<string, string[]> {
  const out: Record<string, string[]> = {};
  for (const n of names) {
    const i = n.lastIndexOf(":");
    const fam = i < 0 ? n : n.slice(0, i), tag = i < 0 ? "latest" : n.slice(i + 1);
    (out[fam] ??= []);
    if (!out[fam].includes(tag)) out[fam].push(tag);
  }
  const size = (t: string) => { const m = /^(\\d+(?:\\.\\d+)?)b(?:$|-)/i.exec(t); return m ? Number(m[1]) : null; };
  for (const f of Object.keys(out)) out[f].sort((a, b) => {
    const sa = size(a), sb = size(b);
    if (sa === null && sb === null) return a < b ? -1 : a > b ? 1 : 0;
    if (sa === null) return -1; if (sb === null) return 1;
    return sa - sb || (a < b ? -1 : a > b ? 1 : 0);
  });
  return out;
}''',
}
