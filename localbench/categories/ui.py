"""UI generation: build a self-contained HTML page from a spec.

Graded with a DOM checklist (required elements, accessibility attributes, behaviour hooks,
responsive CSS, no external dependencies). Pages are saved for the report's side-by-side gallery.
"""
import re
from html.parser import HTMLParser

from ..textutil import extract_html
from . import Category, Grade, Item

SYSTEM = "You are a senior front-end engineer. You write clean, accessible, self-contained HTML/CSS/JS."
PROMPT = """Build a single self-contained HTML page: {spec}

Hard requirements:
{reqs}
- Everything inline in one file (CSS in <style>, JS in <script>); no external URLs, fonts or libraries.
- Include <meta name="viewport" content="width=device-width, initial-scale=1">.
- Make it look polished.

Reply with the complete page in a single ```html code block."""

# (id, spec, [requirement lines shown to the model], [(check label, selector, min count)])
SPECS = [
    ("login", "a login screen for a web app called 'Northwind'.",
     ["- Email input with id=\"email\" (type=email) and password input with id=\"password\" (type=password), each with a <label>.",
      "- A submit button (type=submit).",
      "- An element with id=\"error\" for validation messages, shown when a field is empty on submit."],
     [("email input", "input#email[type=email]", 1), ("password input", "input#password[type=password]", 1),
      ("labels", "label", 2), ("submit button", "button[type=submit]", 1), ("error element", "#error", 1),
      ("submit handled in JS", "@js:submit", 1)]),
    ("pricing", "a pricing section with three plans: Starter, Pro and Enterprise.",
     ["- Each plan is an element with class=\"tier\" containing a name, a monthly price like \"$19/mo\", a feature list (<ul>) and a button.",
      "- The Pro plan also has the class \"featured\" and is visually highlighted."],
     [("3 tiers", ".tier", 3), ("featured tier", ".tier.featured", 1), ("feature lists", "ul", 3),
      ("buttons", "button", 3), ("prices shown", "@text:/mo", 1)]),
    ("todo", "a to-do list app.",
     ["- Text input id=\"new-todo\", button id=\"add-btn\", list <ul id=\"todo-list\">.",
      "- Adding creates an <li>; each item can be marked done and deleted.",
      "- Items persist in localStorage."],
     [("input", "input#new-todo", 1), ("add button", "button#add-btn", 1), ("list", "ul#todo-list", 1),
      ("event listeners", "@js:addEventListener", 1), ("localStorage", "@js:localStorage", 1),
      ("creates li", "@js:li", 1)]),
    ("dashboard", "an analytics dashboard.",
     ["- A <header> with the app name and a <nav>.",
      "- Four KPI cards, each with class=\"stat-card\", laid out with CSS grid.",
      "- A <table> of recent orders with a <thead> and at least 5 rows in <tbody>."],
     [("header", "header", 1), ("nav", "nav", 1), ("4 stat cards", ".stat-card", 4), ("table head", "thead", 1),
      ("5+ rows", "tbody tr", 5), ("CSS grid", "@css:grid", 1)]),
    ("navbar", "a responsive site header with navigation.",
     ["- A <nav> with at least 4 links.",
      "- On narrow screens (<700px) links collapse behind a button id=\"menu-toggle\" that toggles them and updates aria-expanded.",
      "- Use a CSS @media query."],
     [("nav", "nav", 1), ("4+ links", "nav a", 4), ("toggle button", "button#menu-toggle", 1),
      ("aria-expanded", "button#menu-toggle[aria-expanded]", 1), ("media query", "@css:@media", 1),
      ("toggle updates aria", "@js:aria-expanded", 1)]),
    ("modal", "a page with a button that opens a confirmation modal dialog.",
     ["- Button id=\"open-modal\" opens it; the dialog has id=\"modal\", role=\"dialog\" and aria-modal=\"true\".",
      "- A button id=\"close-modal\" closes it, and so does the Escape key."],
     [("open button", "button#open-modal", 1), ("dialog role", "#modal[role=dialog]", 1),
      ("aria-modal", "#modal[aria-modal=true]", 1), ("close button", "button#close-modal", 1),
      ("Escape key", "@js:Escape", 1)]),
    ("contact", "a contact form with client-side validation.",
     ["- <form id=\"contact\"> with a required name input (name=\"name\"), a required email input (type=email) and a <textarea name=\"message\">.",
      "- On submit, prevent the default, validate, and show the result in an element id=\"status\" with aria-live=\"polite\"."],
     [("form", "form#contact", 1), ("name required", "input[name=name][required]", 1),
      ("email required", "input[type=email][required]", 1), ("textarea", "textarea[name=message]", 1),
      ("status live region", "#status[aria-live=polite]", 1), ("preventDefault", "@js:preventDefault", 1)]),
    ("accordion", "an FAQ accordion with 4 questions.",
     ["- Each question is a <button class=\"accordion-header\" aria-expanded=\"false\"> followed by a <div class=\"accordion-panel\">.",
      "- Clicking a header toggles its panel and its aria-expanded value."],
     [("4 headers", "button.accordion-header", 4), ("aria-expanded", "button.accordion-header[aria-expanded]", 4),
      ("4 panels", ".accordion-panel", 4), ("click handling", "@js:click", 1),
      ("aria toggled in JS", "@js:aria-expanded", 1)]),
]
UNIVERSAL = [("viewport meta", "meta[name=viewport]", 1), ("has <style>", "style", 1), ("self-contained", "@noext", 1)]

_SEL = re.compile(r"^(?P<tag>[a-z0-9]+)?(?P<rest>(?:[#.][\w-]+|\[[^\]]+\])*)$", re.I)


class _Dom(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.els, self.stack, self.text, self.css, self.js = [], [], [], [], []
        self._in = None

    def handle_starttag(self, tag, attrs):
        a = {k.lower(): (v if v is not None else "") for k, v in attrs}
        el = {"tag": tag, "attrs": a, "anc": [e["tag"] for e in self.stack] + [x for e in self.stack for x in _keys(e)]}
        self.els.append(el)
        if tag not in ("meta", "link", "input", "br", "img", "hr", "source"):
            self.stack.append(el)
        if tag in ("style", "script"):
            self._in = tag

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i]["tag"] == tag:
                del self.stack[i:]
                break
        if tag in ("style", "script"):
            self._in = None

    def handle_data(self, data):
        (self.css if self._in == "style" else self.js if self._in == "script" else self.text).append(data)


def _keys(el):
    a = el["attrs"]
    return ([f"#{a['id']}"] if a.get("id") else []) + [f".{c}" for c in a.get("class", "").split()]


def _match(el, sel):
    m = _SEL.match(sel)
    if not m:
        return False
    if m["tag"] and el["tag"] != m["tag"].lower():
        return False
    a = el["attrs"]
    for part in re.findall(r"[#.][\w-]+|\[[^\]]+\]", m["rest"] or ""):
        if part[0] == "#" and a.get("id") != part[1:]:
            return False
        if part[0] == "." and part[1:] not in a.get("class", "").split():
            return False
        if part[0] == "[":
            k, _, v = part[1:-1].partition("=")
            k, v = k.strip().lower(), v.strip().strip("'\"")
            if k not in a or (v and a[k].lower() != v.lower()):
                return False
    return True


def _count(dom, selector):
    parts = selector.split()
    target, ancestors = parts[-1], parts[:-1]
    n = 0
    for el in dom.els:
        if not _match(el, target):
            continue
        if all(p in el["anc"] for p in ancestors):  # ancestors: simple tag / #id / .class
            n += 1
    return n


def check_html(html, checks):
    dom = _Dom()
    try:
        dom.feed(html)
    except Exception:
        pass
    js, css, text = "\n".join(dom.js), "\n".join(dom.css), " ".join(dom.text)
    results = {}
    for label, sel, need in checks:
        if sel.startswith("@js:"):
            ok = sel[4:] in js
        elif sel.startswith("@css:"):
            ok = sel[5:] in css
        elif sel.startswith("@text:"):
            ok = sel[6:] in text
        elif sel == "@noext":
            ok = not re.search(r"""(?:src|href)\s*=\s*["']?(?:https?:)?//""", html, re.I) and "@import url(http" not in css
        else:
            ok = _count(dom, sel) >= need
        results[label] = bool(ok)
    return results


class UIGen(Category):
    name = "ui"
    title = "UI Gen"
    description = "Build a working, accessible, self-contained web UI from a spec."
    method = (f"{len(SPECS)} UI specs (login, pricing, todo, dashboard, navbar, modal, form, accordion). "
              "Score = fraction of DOM/ARIA/JS/CSS checklist items met. Pages are viewable in the gallery.")
    # Strong models write rich pages: 4096 cut qwen3.8:27b off on all 8 specs before it reached its
    # <script>. 7168 still fits an 8K context with the prompt (~750 tokens).
    max_tokens = 7168
    uses_limit = False

    def load(self):
        return [Item(f"ui/{sid}", [{"role": "system", "content": SYSTEM},
                                   {"role": "user", "content": PROMPT.format(spec=spec, reqs="\n".join(reqs))}],
                     {"checks": checks + UNIVERSAL, "spec": spec}) for sid, spec, reqs, checks in SPECS]

    def grade(self, item, text):
        html = extract_html(text)
        res = check_html(html, item.payload["checks"])
        failed = [k for k, v in res.items() if not v]
        return Grade(sum(res.values()) / len(res), ("failed: " + ", ".join(failed)) if failed else "",
                     {"checks": res, "html": html})
