"""Parsing model output: code blocks, JSON, letters, numbers."""
from __future__ import annotations

import json
import re

_FENCE = re.compile(r"```[ \t]*([\w+-]*)[^\n]*\n(.*?)(?:```|$)", re.S)


def code_blocks(text: str, lang: str | None = None) -> list[str]:
    blocks = [(l.lower(), b) for l, b in _FENCE.findall(text)]
    if lang:
        pref = [b for l, b in blocks if l in (lang, lang[:2])]
        if pref:
            return pref
    return [b for _, b in blocks]


def extract_python(text: str, must_define: str | None = None) -> str:
    """Pick the python block that defines `must_define` (else the longest block, else raw text)."""
    blocks = code_blocks(text, "python")
    if must_define:
        hits = [b for b in blocks if re.search(rf"def\s+{re.escape(must_define)}\s*\(", b)]
        if hits:
            return max(hits, key=len)
    if blocks:
        return max(blocks, key=len)
    return text


def extract_html(text: str) -> str:
    blocks = code_blocks(text, "html")
    if blocks:
        return max(blocks, key=len)
    m = re.search(r"(<!doctype html.*?</html>|<html.*?</html>)", text, re.S | re.I)
    return m.group(1) if m else text


def extract_json(text: str) -> dict | None:
    for cand in code_blocks(text, "json") + [text]:
        for m in re.finditer(r"\{.*?\}", cand, re.S):  # shortest objects first
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
        m = re.search(r"\{.*\}", cand, re.S)  # then greedy (nested objects)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return None


def as_bool(v) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes", "y", "1"):
            return True
        if s in ("false", "no", "n", "0"):
            return False
    return None


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # TruthfulQA has questions with up to 13 options


def extract_letter(text: str, n_options: int) -> str | None:
    valid = LETTERS[:n_options]
    t = text.strip()
    for p in (rf"(?i)answer\s*(?:is|:)?\s*\(?\**([{valid}])\b",
              rf"^\(?\**([{valid}])\**\)?(?:[.):\s]|$)",
              rf"\b([{valid}])\b"):
        m = re.search(p, t, re.M)
        if m:
            return m.group(1).upper()
    return None


_NUM = r"-?\$?\d[\d,]*(?:\.\d+)?"


def extract_number(text: str) -> str | None:
    for p in (rf"(?i)answer\s*[:=]?\s*\**\s*({_NUM})", rf"####\s*({_NUM})", rf"\\boxed\{{({_NUM})\}}"):
        m = re.findall(p, text)
        if m:
            return norm_num(m[-1])
    m = re.findall(_NUM, text)
    return norm_num(m[-1]) if m else None


def norm_num(s: str) -> str:
    s = s.replace(",", "").replace("$", "").rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s
