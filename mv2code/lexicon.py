"""Repair Persian OCR errors with a word list and a letter-confusion model (no model files, no torch).

Small Persian text is where OCR engines slip: the dots that tell ب پ ت ث ن ی apart, ج چ ح خ, ف ق, ع غ ... shrink to a pixel and a word becomes a
non-word (``محفوظ`` -> ``محقوظ``, ``پشتیبانی`` -> ``پشنیبانی``).  Instead of trusting the engine blindly, every word that is *not* in a word list is
compared with the list words of the same length, letter by letter, with a small cost for letters that look alike at small sizes and a full cost
for any other letter; the best common word wins - only when it is close and clearly more frequent than the read word.

* word list: ``data/fa_words.tsv.gz`` (28k words with their frequency, see ``data/NOTICE.md`` for the source and licence);
* brand words (``glossary``) and words that occur three times or more on the page are never "corrected" (names are not in any dictionary);
* the corrected word also gets its standard half-space (``میکند`` -> ``می‌کند``) when the list knows that spelling.

Pure functions apart from the lazily loaded word list; no dependency besides numpy.
"""
from __future__ import annotations

import gzip
import os
import re
import threading
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

ZWNJ = "‌"
LETTERS = "ءآأؤإئابةتثجحخدذرزسشصضطظعغفقکلمنهویپچژگ"
_ID = {c: i for i, c in enumerate(LETTERS)}
_N = len(LETTERS)

# letters that differ only by their dots / a tiny stroke -> cheap to confuse at 10-14 px
_SKELETON = ["بپتثنیئ", "جچحخ", "دذ", "رزژ", "سش", "صض", "طظ", "عغ", "فق", "کگ", "هة", "اآأإ", "وؤ"]
# look-alikes across skeletons that small-text OCR really mixes up (observed on real pages)
_LOOKALIKE = [("م", "ح", 0.6), ("غ", "خ", 0.55), ("غ", "ف", 0.55), ("غ", "ق", 0.6), ("ه", "ت", 0.7), ("ل", "ا", 0.7), ("ر", "و", 0.75), ("م", "ه", 0.8),
              ("ن", "ر", 0.8), ("ی", "ر", 0.85), ("ع", "ف", 0.7), ("ک", "ل", 0.85), ("ط", "ل", 0.85), ("د", "و", 0.85), ("ع", "ه", 0.8), ("چ", "ح", 0.5)]


def _cost_matrix() -> np.ndarray:
    c = np.ones((_N, _N), np.float32)
    np.fill_diagonal(c, 0.0)
    for grp in _SKELETON:
        for a in grp:
            for b in grp:
                if a != b and a in _ID and b in _ID:
                    c[_ID[a], _ID[b]] = 0.35
    for a, b, v in _LOOKALIKE:
        if a in _ID and b in _ID:
            c[_ID[a], _ID[b]] = min(c[_ID[a], _ID[b]], v)
            c[_ID[b], _ID[a]] = min(c[_ID[b], _ID[a]], v)
    return c


COST = _cost_matrix()
_INSERTABLE = "اویهتنرلم"
DEL_COST = 0.9
INS_COST = 0.9

_TOKEN = re.compile("^([^" + LETTERS + ZWNJ + "0-9۰-۹٠-٩]*)([" + LETTERS + ZWNJ + "]+)([^" + LETTERS + ZWNJ + "0-9۰-۹٠-٩]*)$")


def _key(word: str) -> str:
    return word.replace(ZWNJ, "")


class Lexicon:
    def __init__(self, entries: Iterable[Tuple[str, float]]):
        best: Dict[str, Tuple[str, float]] = {}      # key -> (surface with half-space, zipf of that surface)
        zipf: Dict[str, float] = {}                  # key -> best zipf over spellings
        for w, z in entries:
            k = _key(w)
            if not k or any(ch not in _ID for ch in k):
                continue
            zipf[k] = max(zipf.get(k, 0.0), z)
            if k not in best or z > best[k][1]:
                best[k] = (w, z)
        self.best, self.zipf = best, zipf
        by_len: Dict[int, List[str]] = {}
        for k in zipf:
            by_len.setdefault(len(k), []).append(k)
        self._arr: Dict[int, Tuple[np.ndarray, List[str], np.ndarray]] = {}
        for n, ks in by_len.items():
            a = np.array([[_ID[ch] for ch in k] for k in ks], np.int16)
            self._arr[n] = (a, ks, np.array([zipf[k] for k in ks], np.float32))

    # ------------------------------------------------------------------ queries
    def known(self, word: str, floor: float = 2.0) -> bool:
        return self.zipf.get(_key(word), 0.0) >= floor

    def surface(self, key: str) -> str:
        return self.best[key][0] if key in self.best else key

    def near(self, key: str, extra: Optional[Dict[str, float]] = None, max_cost: float = 0.8, min_zipf: float = 3.0) -> Optional[Tuple[str, float, float]]:
        """best list word close to ``key`` -> (key, zipf, cost) or None."""
        if any(ch not in _ID for ch in key) or len(key) < 3:
            return None
        n = len(key)
        q = np.array([_ID[ch] for ch in key], np.int16)
        found: List[Tuple[float, str, float, float]] = []
        pool = self._arr.get(n)
        if pool is not None:
            a, ks, zs = pool
            cost = COST[q[None, :], a].sum(axis=1)
            ok = np.nonzero((cost <= max_cost) & (zs >= min_zipf))[0]
            for i in ok:
                found.append((float(zs[i]) - 2.2 * float(cost[i]), ks[i], float(zs[i]), float(cost[i])))
        # one letter too many / too few: only towards a really common word (rare real words and names differ from common ones by a letter all the time)
        edit_zipf = max(min_zipf, 4.3)
        if n >= 5 and DEL_COST <= max_cost:
            for i in range(n):
                k2 = key[:i] + key[i + 1:]
                z = self.zipf.get(k2, 0.0)
                if z >= edit_zipf:
                    found.append((z - 2.2 * DEL_COST, k2, z, DEL_COST))
        if n >= 4 and INS_COST <= max_cost:
            for i in range(n + 1):
                for ch in _INSERTABLE:
                    k2 = key[:i] + ch + key[i:]
                    z = self.zipf.get(k2, 0.0)
                    if z >= edit_zipf:
                        found.append((z - 2.2 * INS_COST, k2, z, INS_COST))
        for g, z in (extra or {}).items():
            if len(g) == n:
                c = float(COST[q, np.array([_ID[ch] for ch in g], np.int16)].sum())
                if c <= max_cost:
                    found.append((z - 2.2 * c, g, z, c))
        if not found:
            return None
        found.sort(reverse=True)
        _, k, z, c = found[0]
        return k, z, c


_LEX: Optional[Lexicon] = None
_LOCK = threading.Lock()


def load(path: Optional[str] = None) -> Optional[Lexicon]:
    """The bundled word list (cached).  ``MV2CODE_FA_WORDS`` may point at another ``word<TAB>zipf`` file (optionally .gz)."""
    global _LEX
    if _LEX is not None:
        return _LEX
    with _LOCK:
        if _LEX is not None:
            return _LEX
        p = path or os.environ.get("MV2CODE_FA_WORDS") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "fa_words.tsv.gz")
        try:
            op = gzip.open if p.endswith(".gz") else open
            with op(p, "rt", encoding="utf-8") as fh:
                rows = []
                for line in fh:
                    w, _, z = line.rstrip("\n").partition("\t")
                    if w:
                        rows.append((w, float(z or 3.0)))
            _LEX = Lexicon(rows)
        except (OSError, ValueError):
            _LEX = None
    return _LEX


def correct_token(tok: str, lex: Lexicon, conf: float = 0.7, glossary: Optional[Dict[str, float]] = None, protected: Optional[set] = None) -> str:
    """One whitespace-free token: returns the repaired token (unchanged when it is fine, not Persian, or too far from any common word)."""
    m = _TOKEN.match(tok)
    if not m:
        return tok
    pre, core, post = m.groups()
    key = _key(core)
    if len(key) < 3 or key in (protected or ()) or key in (glossary or {}):
        return tok
    z = lex.zipf.get(key, 0.0)
    if z >= 2.0:
        # known word: only restore the standard half-space spelling when the list prefers it clearly
        surf = lex.surface(key)
        if surf != core and ZWNJ in surf and ZWNJ not in core and lex.best[key][1] >= z - 0.01 and _key(surf) == key:
            return pre + surf + post
        return tok
    n = len(key)
    # brand words first: an unknown word that is one or two look-alike letters away from a brand word is that brand word (any length)
    if glossary:
        qa = np.array([_ID[ch] for ch in key if ch in _ID], np.int16)
        if len(qa) == n:
            best_g = None
            for g in glossary:
                if len(g) == n and all(ch in _ID for ch in g):
                    c = float(COST[qa, np.array([_ID[ch] for ch in g], np.int16)].sum())
                    if c <= 1.0 and (best_g is None or c < best_g[0]):
                        best_g = (c, g)
            if best_g is not None:
                return pre + best_g[1] + post
    strict = conf >= 0.9
    # how much "difference" a word of this length may have: short words are one dot away from a dozen other words, so they are barely touched
    limit = 0.0 if n <= 3 else 0.75 if n == 4 else 0.85 if n <= 6 else 1.1
    near = lex.near(key, None, max_cost=limit - (0.15 if strict else 0.0), min_zipf=3.5 if strict else 3.0)
    if near is None:
        return tok
    k2, z2, cost = near
    if z2 < z + 1.2:
        return tok
    return pre + lex.surface(k2) + post


def correct_text(text: str, conf: float = 0.7, glossary: Optional[Iterable[str]] = None, protected: Optional[set] = None, lex: Optional[Lexicon] = None) -> str:
    lex = lex or load()
    if lex is None or not text:
        return text
    g: Dict[str, float] = {}
    for w in glossary or ():
        for part in re.split(r"\s+", str(w).strip()):
            k = _key(part.strip(".,:;،؛؟!()«»\"'"))
            if len(k) >= 2 and all(ch in _ID for ch in k):
                g[k] = 6.0
    return " ".join(correct_token(t, lex, conf, g, protected) for t in text.split(" "))


def repeated_words(texts: Iterable[str]) -> set:
    """Words that appear three times or more on the page: names and jargon repeated by the page itself, never 'repaired'."""
    cnt: Dict[str, int] = {}
    for t in texts:
        seen = set()
        for tok in t.split():
            m = _TOKEN.match(tok)
            if m:
                k = _key(m.group(2))
                if len(k) >= 3 and k not in seen:
                    seen.add(k)
                    cnt[k] = cnt.get(k, 0) + 1
    return {k for k, n in cnt.items() if n >= 3}
