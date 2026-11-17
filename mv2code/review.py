"""Which recognised lines should a person look at again?

OCR is never perfect on small Persian text.  Instead of pretending, the converter marks the lines it is not sure about (``TL.check`` /
``report["review"]`` / ``"review": true`` on a text node of the tree) so a UI can highlight them ("check this line").

A line is flagged when

* the OCR engine itself was not confident (``conf`` below ``LOW_CONF``), or
* it contains a Persian word of 3+ letters that no word list knows (after the repair step), or
* it contains a token that cannot be a word (Persian letters glued to digits/Latin letters, a stray single character at an end ...).

Brand words (``glossary``) and words the page repeats three times are not "unknown".  Text that did not come from OCR (``ocr_lines``, e.g.
read from a page's DOM) is never flagged.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Optional

from . import lexicon, persian

LOW_CONF = 0.80
MAX_ITEMS = 60

_PERSIAN_WORD = re.compile("^[^" + lexicon.LETTERS + lexicon.ZWNJ + "]*([" + lexicon.LETTERS + lexicon.ZWNJ + "]{3,})[^" + lexicon.LETTERS + lexicon.ZWNJ + "]*$")
_MIXED = re.compile("(?:[؀-ۿ][A-Za-z0-9۰-۹]|[A-Za-z0-9۰-۹][؀-ۿ])")


def reasons_for(text: str, conf: float, lex: Optional[lexicon.Lexicon], glossary: Optional[Iterable[str]] = None, protected: Optional[set] = None) -> List[str]:
    out: List[str] = []
    if conf < LOW_CONF:
        out.append("conf")
    if lex is not None and persian.has_arabic_script(text):
        g = {lexicon._key(p) for w in (glossary or ()) for p in str(w).split()}
        unknown = 0
        for tok in text.split():
            m = _PERSIAN_WORD.match(tok)
            if not m:
                continue
            k = lexicon._key(m.group(1))
            if k in g or k in (protected or ()):
                continue
            if not lex.is_known(k):
                unknown += 1
        if unknown:
            out.append("unknown")
    toks = text.split()
    if _MIXED.search(text.replace("‌", "")) and any(_MIXED.search(t) for t in toks):
        out.append("mixed")
    if len(toks) >= 3 and persian.has_arabic_script(text):
        ends = (toks[0], toks[-1])
        if any(len(t) == 1 and t.isalnum() for t in ends):
            out.append("stray")
    return out


def flag_lines(lines, glossary: Optional[Iterable[str]] = None, from_ocr: bool = True) -> int:
    """Sets ``ln.check`` / ``ln.why`` on every line; returns how many lines are flagged."""
    if not from_ocr:
        return 0
    lex = lexicon.load()
    protected = lexicon.repeated_words(ln.text for ln in lines)
    n = 0
    for ln in lines:
        why = reasons_for(ln.text, ln.conf, lex, glossary, protected) if ln.text else []
        ln.check, ln.why = bool(why), why
        n += 1 if why else 0
    return n


def report_items(lines, limit: int = MAX_ITEMS) -> list:
    """The flagged lines for the report: ``[{"box": [x, y, w, h], "text": str, "conf": float, "why": [...]}]``, least confident first."""
    rows = [dict(box=[int(v) for v in ln.box], text=ln.text, conf=round(float(ln.conf), 2), why=list(ln.why)) for ln in lines if getattr(ln, "check", False)]
    rows.sort(key=lambda r: (r["conf"], r["box"][1]))
    return rows[:limit]
