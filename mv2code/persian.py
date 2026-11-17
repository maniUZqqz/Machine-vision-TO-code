"""Persian / Arabic-script text helpers: letter normalisation, digits, ZWNJ, bidi hints.

Pure functions, no dependencies.  The conversion pipeline calls them after OCR; they are
also useful on their own (and unit-tested).
"""
from __future__ import annotations

import re
import unicodedata

ZWNJ = "‌"

_AR_LETTER = re.compile("[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
_LATIN = re.compile("[A-Za-z]")
_AR_FIX = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ة": "ه", "ۀ": "ۀ", "ﻻ": "لا", "ﻷ": "لا"})
_ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "۰۱۲۳۴۵۶۷۸۹")
_TO_FA_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
_TO_EN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_FA_DIGITS = set("۰۱۲۳۴۵۶۷۸۹")
_EN_DIGITS = set("0123456789")
# digits whose shapes differ between the two scripts (4 5 6 vs ۴ ۵ ۶ ...) -> usable as evidence
_FA_ONLY = set("۴۵۶۰")
_EN_ONLY = set("4560")
_DIAC = re.compile("[ً-ٰٟـ]")


def has_arabic_script(s: str) -> bool:
    return bool(_AR_LETTER.search(s))


def script_ratio(s: str) -> float:
    """Share of Arabic-script letters among all letters (digits ignored)."""
    ar = len(_AR_LETTER.findall(s))
    la = len(_LATIN.findall(s))
    return ar / (ar + la) if (ar + la) else 0.0


def is_rtl(s: str) -> bool:
    """Base direction of ``s``: the first strong letter decides (Unicode bidi rule P2/P3)."""
    for ch in s:
        if _AR_LETTER.match(ch):
            return True
        if _LATIN.match(ch):
            return False
    return False


def normalize_letters(s: str) -> str:
    """Arabic ي/ك/ة -> Persian ی/ک/ه, drop tatweel and harakat, NFC-ish clean-up."""
    s = unicodedata.normalize("NFKC", s)  # presentation forms -> base letters
    s = _DIAC.sub("", s)
    return s.translate(_AR_FIX)


def arabic_indic_to_persian(s: str) -> str:
    return s.translate(_ARABIC_INDIC)


def to_persian_digits(s: str) -> str:
    return s.translate(_TO_FA_DIGITS)


def to_latin_digits(s: str) -> str:
    return s.translate(_TO_EN_DIGITS)


def digit_evidence(s: str):
    """(persian_only, latin_only) counts of digits whose shape identifies the script."""
    return sum(ch in _FA_ONLY for ch in s), sum(ch in _EN_ONLY for ch in s)


def fix_digits(s: str, page_digits: str, line_rtl: bool) -> str:
    """Make the digit script of ``s`` consistent.

    ``page_digits`` is 'fa' or 'en': the digit script that dominates the page (decided from
    unambiguous glyphs).  Persian lines follow the page, pure-Latin lines keep Latin digits.
    """
    s = arabic_indic_to_persian(s)
    if not any(ch.isdigit() for ch in s):
        return s
    pe, le = digit_evidence(s)
    if pe and not le:
        target = "fa"
    elif le and not pe:
        target = "en"
    elif pe and le:
        target = "fa" if pe >= le else "en"
    else:  # ambiguous glyphs only (1 2 3 7 8 9)
        target = page_digits if line_rtl else "en"
    return to_persian_digits(s) if target == "fa" else to_latin_digits(s)


_ZWNJ_PREFIX = re.compile(r"(?<![؀-ۿ])(ن?می) (?=[؀-ۿ])")
_ZWNJ_SUFFIX = re.compile(r"(?<=[؀-ۿ]) (ها|های|هایی|ام|ات|اش|مان|تان|شان|تر|ترین|ایم|اید|اند)(?![؀-ۿ])")


def restore_zwnj(s: str) -> str:
    """Re-insert the half-space after ``می``/``نمی`` and before common suffixes (OCR only sees spaces)."""
    s = _ZWNJ_PREFIX.sub(lambda m: m.group(1) + ZWNJ, s)
    s = _ZWNJ_SUFFIX.sub(lambda m: ZWNJ + m.group(1), s)
    return s


_FA_COMMA = re.compile(r"(?<=[؀-ۿ]),")
_FA_SEMI = re.compile(r"(?<=[؀-ۿ]);")
_FA_Q = re.compile(r"(?<=[؀-ۿ])\?")


def fix_punctuation(s: str) -> str:
    """Persian lines use ، ؛ ؟ ; OCR (Latin punctuation model) reads them as , ; ?"""
    if not has_arabic_script(s):
        return s
    s = _FA_COMMA.sub("،", s)
    s = _FA_SEMI.sub("؛", s)
    return _FA_Q.sub("؟", s)


def clean_line(s: str) -> str:
    s = re.sub(r"[‎‏‪-‮﻿]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_line(s: str, page_digits: str = "fa", zwnj: bool = True, conf: float = 0.7, glossary=None, protected=None, repair: bool = True) -> str:
    s = clean_line(normalize_letters(s))
    rtl = is_rtl(s) or (script_ratio(s) == 0 and page_digits == "fa" and any(ch.isdigit() for ch in s) and False)
    s = fix_digits(s, page_digits, rtl)
    if zwnj and has_arabic_script(s):
        s = restore_zwnj(s)
    if has_arabic_script(s):
        s = fix_punctuation(s)
        if repair:
            from . import lexicon
            s = lexicon.correct_text(s, conf, glossary, protected)
    return s


def decide_page_digits(lines) -> str:
    """Pick the page's digit script from the unambiguous digit glyphs of all OCR lines."""
    pe = le = 0
    for ln in lines:
        a, b = digit_evidence(arabic_indic_to_persian(ln))
        pe += a
        le += b
    if pe == le == 0:
        n_ar = sum(1 for ln in lines if has_arabic_script(ln))
        return "fa" if n_ar >= max(1, len(lines) // 2) else "en"
    return "fa" if pe >= le else "en"


def page_direction(lines) -> str:
    """'rtl' when most letters of the page are Arabic-script, else 'ltr'."""
    ar = la = 0
    for ln in lines:
        ar += len(_AR_LETTER.findall(ln))
        la += len(_LATIN.findall(ln))
    return "rtl" if ar > la else "ltr"
