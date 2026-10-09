"""Security review: spot the vulnerability (and its CWE) — and don't cry wolf on safe code.

Curated set: vulnerable snippets with accepted CWE ids, plus safe look-alikes as controls.
Score per item: vulnerable -> 1.0 if flagged with an accepted CWE, 0.5 if flagged with a wrong
CWE, 0 if missed. Safe -> 1.0 if not flagged, 0 if flagged (false positive).
"""
import re

from ..textutil import as_bool, extract_json
from . import Category, Grade, Item

SYSTEM = "You are an application security engineer reviewing code."
PROMPT = """Review this {lang} code for security vulnerabilities.

```{lang}
{code}
```

Reply with only a JSON object:
{{"vulnerable": true or false, "cwe": "CWE-<number> of the main issue, or null", "reason": "<one sentence>"}}"""

# (id, lang, code, accepted CWE ids — empty list means the snippet is safe)
CASES = [
    ("sqli-fstring", "python", '''def get_user(db, username):
    cur = db.cursor()
    cur.execute(f"SELECT id, email FROM users WHERE name = '{username}'")
    return cur.fetchone()''', [89]),
    ("cmdi-os-system", "python", '''import os
from flask import request

@app.route("/ping")
def ping():
    host = request.args.get("host", "")
    os.system("ping -c 1 " + host)
    return "ok"''', [78, 77]),
    ("path-traversal", "python", '''import os
from flask import request, send_file

BASE = "/srv/files"

@app.route("/download")
def download():
    return send_file(os.path.join(BASE, request.args["name"]))''', [22, 23, 73]),
    ("pickle-untrusted", "python", '''import base64, pickle
from flask import request

@app.route("/session", methods=["POST"])
def load_session():
    data = base64.b64decode(request.cookies["session"])
    return str(pickle.loads(data))''', [502]),
    ("yaml-unsafe", "python", '''import yaml
from flask import request

@app.route("/config", methods=["POST"])
def upload_config():
    cfg = yaml.load(request.data, Loader=yaml.Loader)
    return {"keys": list(cfg)}''', [502, 20, 94]),
    ("hardcoded-secret", "python", '''import boto3

AWS_ACCESS_KEY_ID = "AKIAIOSFODNN7EXAMPLE"
AWS_SECRET_ACCESS_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"

def client():
    return boto3.client("s3", aws_access_key_id=AWS_ACCESS_KEY_ID,
                        aws_secret_access_key=AWS_SECRET_ACCESS_KEY)''', [798, 259, 321, 312]),
    ("md5-passwords", "python", '''import hashlib

def store_password(db, user, password):
    digest = hashlib.md5(password.encode()).hexdigest()
    db.execute("UPDATE users SET pw = ? WHERE name = ?", (digest, user))''', [916, 327, 328, 759, 760]),
    ("xss-flask", "python", '''from flask import request

@app.route("/hello")
def hello():
    name = request.args.get("name", "")
    return f"<h1>Hello, {name}!</h1>"''', [79, 80]),
    ("ssrf", "python", '''import requests
from flask import request

@app.route("/preview")
def preview():
    url = request.args["url"]
    return requests.get(url, timeout=5).text''', [918]),
    ("js-eval", "javascript", '''app.get("/calc", (req, res) => {
  const result = eval(req.query.expr);
  res.json({ result });
});''', [95, 94]),
    ("js-xss", "javascript", '''app.get("/search", (req, res) => {
  res.send(`<p>Results for ${req.query.q}</p>`);
});''', [79, 80]),
    ("node-cmdi", "javascript", '''const { exec } = require("child_process");

app.post("/convert", (req, res) => {
  exec(`convert ${req.body.filename} out.png`, (err) => {
    res.send(err ? "failed" : "done");
  });
});''', [78, 77]),
    ("weak-random-token", "python", '''import random, string

def make_reset_token():
    alphabet = string.ascii_letters + string.digits
    return "".join(random.choice(alphabet) for _ in range(32))''', [330, 338]),
    ("jwt-no-verify", "python", '''import jwt

def current_user(token):
    claims = jwt.decode(token, options={"verify_signature": False}, algorithms=["HS256"])
    return claims["sub"]''', [347, 287, 345]),
    ("tls-verify-off", "python", '''import requests

def fetch_balance(account_id, api_key):
    r = requests.get(f"https://bank.example.com/api/accounts/{account_id}",
                     headers={"Authorization": f"Bearer {api_key}"}, verify=False)
    return r.json()["balance"]''', [295]),
    ("open-redirect", "python", '''from flask import redirect, request

@app.route("/login", methods=["POST"])
def login():
    authenticate(request.form)
    return redirect(request.args.get("next", "/"))''', [601]),
    ("xxe", "python", '''from lxml import etree
from flask import request

@app.route("/import", methods=["POST"])
def import_xml():
    parser = etree.XMLParser(resolve_entities=True, no_network=False)
    doc = etree.fromstring(request.data, parser)
    return doc.findtext("title")''', [611, 776]),
    ("c-overflow", "c", '''#include <string.h>

void greet(const char *name) {
    char buf[16];
    strcpy(buf, name);
    printf("Hello %s\\n", buf);
}''', [120, 121, 787, 119, 242]),
    ("idor", "python", '''from flask_login import login_required

@app.route("/invoices/<int:invoice_id>")
@login_required
def get_invoice(invoice_id):
    invoice = Invoice.query.get_or_404(invoice_id)
    return invoice.to_json()''', [639, 862, 284, 285, 863, 200]),
    ("tar-slip", "python", '''import tarfile

def install_plugin(uploaded_path, plugins_dir):
    with tarfile.open(uploaded_path) as tar:
        tar.extractall(plugins_dir)''', [22, 23, 59]),
    # ---- safe controls
    ("safe-sql-params", "python", '''def get_user(db, username):
    cur = db.cursor()
    cur.execute("SELECT id, email FROM users WHERE name = %s", (username,))
    return cur.fetchone()''', []),
    ("safe-subprocess", "python", '''import ipaddress, subprocess
from flask import request

@app.route("/ping")
def ping():
    host = str(ipaddress.ip_address(request.args.get("host", "")))
    out = subprocess.run(["ping", "-c", "1", host], capture_output=True, text=True, timeout=5)
    return out.stdout''', []),
    ("safe-path", "python", '''from pathlib import Path
from flask import abort, request, send_file

BASE = Path("/srv/files").resolve()

@app.route("/download")
def download():
    target = (BASE / request.args["name"]).resolve()
    if not target.is_relative_to(BASE) or not target.is_file():
        abort(404)
    return send_file(target)''', []),
    ("safe-bcrypt", "python", '''import bcrypt

def store_password(db, user, password):
    digest = bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12))
    db.execute("UPDATE users SET pw = ? WHERE name = ?", (digest, user))''', []),
    ("safe-token", "python", '''import secrets

def make_reset_token():
    return secrets.token_urlsafe(32)''', []),
    ("safe-template", "python", '''from flask import render_template, request

@app.route("/hello")
def hello():
    # templates/hello.html: <h1>Hello, {{ name }}!</h1>  (Jinja autoescape on)
    return render_template("hello.html", name=request.args.get("name", ""))''', []),
    ("safe-js-dom", "javascript", '''const params = new URLSearchParams(location.search);
const el = document.getElementById("greeting");
el.textContent = `Hello, ${params.get("name") ?? "guest"}`;''', []),
    ("safe-idor", "python", '''from flask import abort
from flask_login import current_user, login_required

@app.route("/invoices/<int:invoice_id>")
@login_required
def get_invoice(invoice_id):
    invoice = Invoice.query.get_or_404(invoice_id)
    if invoice.owner_id != current_user.id:
        abort(404)
    return invoice.to_json()''', []),
]


def _cwe_num(v):
    m = re.search(r"\d+", str(v or ""))
    return int(m.group(0)) if m else None


class Security(Category):
    name = "security"
    title = "Security"
    description = "Spot real vulnerabilities and name the CWE, without flagging safe code."
    method = (f"{sum(1 for c in CASES if c[3])} vulnerable snippets (Python/JS/C, OWASP-style) + "
              f"{sum(1 for c in CASES if not c[3])} safe look-alikes. 1 = correct verdict and accepted CWE, "
              "0.5 = flagged with the wrong CWE.")
    max_tokens = 256
    uses_limit = False

    def load(self):
        return [Item(f"sec/{cid}", [{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": PROMPT.format(lang=lang, code=code)}],
                     {"cwes": cwes}) for cid, lang, code, cwes in CASES]

    def grade(self, item, text):
        cwes = item.payload["cwes"]
        obj = extract_json(text) or {}
        flagged = as_bool(obj.get("vulnerable"))
        if flagged is None:  # fall back to prose
            low = text.lower()
            flagged = bool(re.search(r"cwe-\d+", low)) or ("vulnerab" in low and "not vulnerable" not in low)
        cwe = _cwe_num(obj.get("cwe")) or _cwe_num(re.search(r"CWE-\d+", text) and re.search(r"CWE-\d+", text).group(0))
        extra = {"vulnerable": bool(cwes), "flagged": flagged, "cwe": cwe}
        if not cwes:
            return Grade(0.0 if flagged else 1.0, "false positive" if flagged else "", extra)
        if not flagged:
            return Grade(0.0, "missed", extra)
        if cwe in cwes:
            return Grade(1.0, "", extra)
        return Grade(0.5, f"wrong CWE (got {cwe}, accepted {cwes})", extra)

    def summarize(self, records):
        vul = [r for r in records if r["vulnerable"]]
        safe = [r for r in records if not r["vulnerable"]]
        return {"detection_pct": round(100 * sum(r["flagged"] for r in vul) / max(1, len(vul)), 1),
                "cwe_correct_pct": round(100 * sum(r["score"] == 1 for r in vul) / max(1, len(vul)), 1),
                "false_positive_pct": round(100 * sum(r["flagged"] for r in safe) / max(1, len(safe)), 1)}
