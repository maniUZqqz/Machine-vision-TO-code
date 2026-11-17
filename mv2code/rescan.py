"""Second look at the text lines the first OCR pass was not sure about.

One OCR run per line is a coin toss for small Persian text: the same crop read thicker, bigger or through another model gives a different word
and often the right one.  This module re-reads only the *doubtful* lines (low confidence, a word no list knows, nothing found at all) with a few
cheap variants of the crop and keeps the reading the engine itself trusts most:

* variants (``VARIANTS``): the crop at 64 px line height with the strokes thickened (erode 5x5 / 3x3), gamma-darkened, and at 48 px gamma + erode;
  measured on real and synthetic Persian pages the thickened variants are the best single ones (68.6% of lines exact vs 62.2% for the
  default crop) and the best of them by confidence reached 74.4% (``bench/line_eval.py``, see CHANGELOG);
* choice: ``confidence + 0.03 * (other variants that read the same) + 0.1 * word-list score``, and a variant must beat the first reading by a margin
  - the first reading is never replaced by something worse;
* bounded: at most ``MAX_LINES`` lines and ``MV2CODE_RESCAN_BUDGET`` seconds (default 75); lines are taken least-confident first, so a slow machine
  still gets the worst lines fixed.  ``MV2CODE_RESCAN=off`` disables the pass.

Only the Tesseract backend has a ``rescan`` method (the other backends are single-pass by design).
"""
from __future__ import annotations

import os
import re
import time
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import lexicon, persian, review

MAX_LINES = 90
MARGIN = 0.02
_TOK = re.compile("[ء-يٱ-ەﭐ-﷿]+")
_KEY_DROP = re.compile(r"[\s‌‍.,،؛؟:!\-–—·•|/()«»\"']")


def enabled() -> bool:
    return os.environ.get("MV2CODE_RESCAN", "on").lower() not in ("off", "0", "false", "no")


def budget_seconds() -> float:
    try:
        return max(0.0, float(os.environ.get("MV2CODE_RESCAN_BUDGET", "75")))
    except ValueError:
        return 75.0


def _thicken(c: np.ndarray, k: int) -> np.ndarray:
    return cv2.erode(c, np.ones((k, k), np.uint8))


def _gamma(c: np.ndarray, g: float = 2.2) -> np.ndarray:
    return (255.0 * (c.astype(np.float32) / 255.0) ** g).astype(np.uint8)


# name, line height, transform, language set ("both" = fas+eng / the caller's choice, "fas" = Persian only)
VARIANTS = [
    ("e5", 64, lambda c: _thicken(c, 5), "both"),
    ("gam", 64, _gamma, "both"),
    ("e3", 64, lambda c: _thicken(c, 3), "both"),
    ("gam_e3", 48, lambda c: _thicken(_gamma(c), 3), "fas"),
    ("e5_fas", 64, lambda c: _thicken(c, 5), "fas"),
]


def _key(text: str) -> str:
    t = persian.normalize_line(text or "", "fa", zwnj=False, repair=False)
    return _KEY_DROP.sub("", t)


def _lex_ratio(text: str, lex) -> float:
    if lex is None:
        return 0.0
    k = u = 0
    for tok in (text or "").split():
        core = "".join(_TOK.findall(tok))
        if len(core) >= 3:
            if lex.is_known(lexicon._key(core)):
                k += len(core)
            else:
                u += len(core)
    return (k - 0.8 * u) / max(1, k + u)


def _span(r) -> float:
    """width (in crop pixels) spanned by the *word-like* tokens of a reading (a one- or two-character token is an icon fragment more often than a word)"""
    ws = getattr(r, "ws", None)
    if not ws or len(ws[0]) < 9:
        return 0.0
    keep = [w for w in ws if sum(ch.isalnum() for ch in w[4]) >= 3]       # one or two characters at an end are icon fragments more often than words
    if not keep:
        keep = [w for w in ws if sum(ch.isalnum() for ch in w[4]) >= 2 and not (w[4].isdigit() and len(w[4]) <= 2)] or list(ws)
    return float(max(w[3] + w[6] for w in keep) - min(w[3] for w in keep))


def needs_second_look(text: str, conf: float, lex) -> bool:
    if not (text or "").strip():
        return True
    if conf < 0.9:
        return True
    try:
        norm = persian.normalize_line(text, "fa", zwnj=True, repair=True, conf=conf)
        return bool(review.reasons_for(norm, conf, lex))
    except Exception:  # noqa: BLE001
        return False


_LETTER_AR = re.compile("[ء-يٱ-ۓ]")
_LATIN_TOKEN = re.compile(r"^[A-Za-z][A-Za-z0-9&+.\-]{1,11}$")
_NUM_TOKEN = re.compile(r"^[+%]?[0-9][0-9.,]*%?$")


_ACRONYM = re.compile(r"^[A-Z][A-Za-z]{1,9}$")


def _digit_token(text: str) -> bool:
    """two or more digits (any script) and nothing else: what the Persian model makes of a bold Latin acronym (``PDF`` -> ``۳۱۴``)"""
    t = text.rstrip(".")
    return len(t) >= 2 and all(ch.isdigit() for ch in t)


def _suspect_token(text: str, conf: float) -> bool:
    """a Latin word / number inside a Persian line that was read with a doubtful confidence (an icon, ``PDF`` read as ``۳۵۴``, ``100%`` as ``96100``)"""
    if conf >= 90 or _LETTER_AR.search(text):
        return False
    return any(ch.isalnum() for ch in text) or "%" in text


def fix_tokens(backend, results: list) -> list:
    """Re-read single doubtful Latin words / numbers of Persian lines on their own (English model, then a digit whitelist) - the Persian model
    turns ``PDF`` into digits and ``100%`` into ``96100``.  Only clearly better readings replace the old token."""
    out = list(results)
    for k, r in enumerate(out):
        ws, src = getattr(r, "ws", None), getattr(r, "src", None)
        if not ws or src is None or len(ws) < 2:
            continue
        if sum(len(_LETTER_AR.findall(w[4])) for w in ws) < 2:
            continue
        new, changed = list(ws), False
        H, W = src.shape[:2]
        for i, w in enumerate(ws):
            if len(w) < 9 or not (_suspect_token(w[4], w[5]) or _digit_token(w[4])):
                continue
            x0, x1 = max(0, w[3] - 6), min(W, w[3] + w[6] + 6)
            y0, y1 = max(0, w[7] - 6), min(H, w[7] + w[8] + 6)
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            c = src[y0:y1, x0:x1]
            f = max(1.0, 56.0 / c.shape[0])
            if f > 1.05:
                c = cv2.resize(c, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
            c = cv2.copyMakeBorder(c, 10, 10, 14, 14, cv2.BORDER_CONSTANT, value=255)
            try:
                a_t, a_c = backend._read_word(c, "eng", 7)
            except Exception:  # noqa: BLE001
                continue
            tok = None
            a_t = a_t.strip("-–—_ ")
            if (_LATIN_TOKEN.match(a_t) or _NUM_TOKEN.match(a_t)) and a_c >= 85 and a_c >= w[5] + 10 and a_t != w[4]:
                tok, cf = a_t, a_c
            elif _NUM_TOKEN.match(a_t) and any(ch in a_t for ch in "%+") and a_c >= 88 and a_c >= w[5] + 2 and a_t != w[4]:
                tok, cf = a_t, a_c                         # a sign (%, +) is something the Persian model cannot read: it becomes digits (``100%`` -> ``96100``)
            elif _digit_token(w[4]) and _ACRONYM.match(a_t.rstrip(".")) and (a_c >= 88 or (a_c >= 80 and a_c >= w[5] + 12)):
                tok, cf = a_t, a_c                         # digits that the English model reads as a clear acronym: the digits were a misread of it
            if tok is not None and w[4].endswith(".") and not tok.endswith("."):
                tok += "."
            if tok is not None:
                new[i] = w[:4] + (tok, max(cf, 90.0)) + w[6:]            # checked on its own: keep it out of the junk filter for Latin words at a line end
                changed = True
        if changed:
            r2 = backend._assemble(new)
            r2.src = src
            out[k] = r2
    return out


def rescan(backend, img: np.ndarray, boxes: Sequence, crops: Sequence[np.ndarray], results: list, lang: str = "auto", prepare=None) -> list:
    """Returns ``results`` with the doubtful lines re-read (a new list; the entries of lines that stayed are the same objects)."""
    if not enabled() or not results:
        return results
    results = fix_tokens(backend, results)
    lex = lexicon.load()
    todo: List[Tuple[float, int]] = []
    for k, r in enumerate(results):
        h = boxes[k][3]
        if h < 8 or boxes[k][2] < 6:
            continue
        if needs_second_look(r.text, r.conf, lex):
            # empty reads (a missing line) come after the readings that exist but are weak; bigger ink first
            todo.append(((0.0 if r.text.strip() else 1.0) + (r.conf if r.text.strip() else 1.0 - min(1.0, boxes[k][2] * boxes[k][3] / 4000.0)), k))
    if not todo:
        return results
    todo.sort()
    todo = todo[:MAX_LINES]
    t_end = time.monotonic() + budget_seconds()
    la_both = backend._lang_arg(lang)
    la_fas = "fas" if "fas" in set(backend.installed_langs()) and lang != "en" else la_both
    cands = {k: [(results[k].text, results[k].conf, "first", results[k])] for _, k in todo}
    def settled(k: int) -> bool:
        """two readings agree and the best one is confident: more variants would only cost time"""
        its = cands[k]
        if len(its) < 3:
            return False
        keys = {}
        for t, c, _n, _r in its:
            keys.setdefault(_key(t), []).append(c)
        return any(len(cs) >= 2 and max(cs) >= 0.92 for cs in keys.values())

    for name, th, fn, lset in VARIANTS:
        if lset == "fas" and la_fas == la_both and "fas" not in la_both:
            continue                                   # no Persian model: the Persian-only variants add nothing
        for _, k in todo:
            if time.monotonic() > t_end:
                break
            if settled(k):
                continue
            c = prepare(img, boxes[k], th) if prepare else crops[k]
            try:
                r = backend._read_line(fn(c), la_both if lset == "both" else la_fas)
            except Exception:  # noqa: BLE001
                continue
            if r is not None and r.text.strip():
                cands[k].append((r.text, r.conf, name, r))
        if time.monotonic() > t_end:
            break
    out = list(results)
    for _, k in todo:
        fixed = fix_tokens(backend, [it[3] for it in cands[k]])         # every candidate gets its Latin words / numbers re-read before they are compared
        cands[k] = [(f.text, f.conf, it[2], f) for it, f in zip(cands[k], fixed)]
        items = cands[k]
        if len(items) < 2:
            continue
        votes = {}
        for t, _c, _n, _r in items:
            votes[_key(t)] = votes.get(_key(t), 0) + 1

        ref = max([_span(it[3]) for it in items] + [1.0])        # the widest reading: a reading that lost a word at an end (a higher average, a shorter text) must not win

        def score(it):
            t, c, _n, r = it
            sp = _span(r)
            cov = (sp / ref) if sp > 0 else 1.0
            return c + 0.03 * (votes[_key(t)] - 1) + 0.1 * _lex_ratio(persian.normalize_line(t, "fa", repair=True, conf=c), lex) + 1.0 * min(0.0, cov - 0.93)
        first = items[0]
        best = max(items[1:], key=score)
        if score(best) > score(first) + MARGIN:
            out[k] = best[3]
    return fix_tokens(backend, out)
