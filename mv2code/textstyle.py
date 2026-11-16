"""Text styling from pixels: font family/weight/size/colour by analysis-by-synthesis.

Each line is re-rendered with the bundled OFL fonts (Vazirmatn for Persian, Inter or
Vazirmatn for Latin) and compared with the observed ink; the best (font, weight) wins and the
font size is fitted so that the rendered width equals the observed width.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import cv2
import numpy as np

from . import persian
from .imaging import Box, crop, mode_color

FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")

FACES = {  # (family, weight) -> (arabic-subset file, latin-subset file)
    ("Vazirmatn", 400): ("vazirmatn-arabic-400-normal.ttf", "vazirmatn-latin-400-normal.ttf"),
    ("Vazirmatn", 500): ("vazirmatn-arabic-500-normal.ttf", "vazirmatn-latin-500-normal.ttf"),
    ("Vazirmatn", 700): ("vazirmatn-arabic-700-normal.ttf", "vazirmatn-latin-700-normal.ttf"),
    ("Inter", 400): (None, "inter-latin-400-normal.ttf"),
    ("Inter", 500): (None, "inter-latin-500-normal.ttf"),
    ("Inter", 600): (None, "inter-latin-600-normal.ttf"),
    ("Inter", 700): (None, "inter-latin-700-normal.ttf"),
}


@dataclass
class TextStyle:
    family: str = "Vazirmatn"
    weight: int = 400
    size: float = 16.0
    color: tuple = (0, 0, 0)
    score: float = 0.0
    ascent: float = 1.0      # em units (font hhea ascent / size)
    descent: float = 0.4
    ink_top: float = 0.0     # distance from the ascent line to the ink top, in em (for this text)


def _has_raqm() -> bool:
    try:
        from PIL import features
        return bool(features.check("raqm"))
    except Exception:
        return False


HAS_RAQM = _has_raqm()


@lru_cache(maxsize=64)
def _font(path: str, size: int):
    from PIL import ImageFont
    kw = {}
    if HAS_RAQM:
        kw["layout_engine"] = ImageFont.Layout.RAQM
    return ImageFont.truetype(path, size, **kw)


def _shape(text: str) -> str:
    """Without raqm, Arabic-script text needs explicit joining; use arabic_reshaper when present."""
    if HAS_RAQM or not persian.has_arabic_script(text):
        return text
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        return get_display(arabic_reshaper.reshape(text))
    except Exception:
        return text


def _face_path(family: str, weight: int, arabic: bool) -> str:
    a, l = FACES[(family, weight)]
    name = a if (arabic and a) else (l or a)
    return os.path.join(FONT_DIR, name)


def metrics(family: str, weight: int):
    """(ascent, descent) in em (hhea)."""
    p = _face_path(family, weight, False)
    f = _font(p, 200)
    a, d = f.getmetrics()
    return a / 200.0, d / 200.0


def render_runs(text: str, family: str, weight: int, size: int, rtl: bool):
    """Render ``text`` -> (uint8 coverage mask, ink bbox, advance) using per-script faces."""
    from PIL import Image, ImageDraw
    # split into Arabic-script and other runs so each run uses a face that has its glyphs
    runs = []
    cur, cur_ar = "", None
    for ch in text:
        ar = persian.has_arabic_script(ch) or ch in "‌‍"
        if ch.isspace() or ch.isdigit() and cur_ar is not None:
            ar = cur_ar if cur_ar is not None else ar
        if cur_ar is None or ar == cur_ar:
            cur += ch
            cur_ar = ar if cur_ar is None else cur_ar
        else:
            runs.append((cur, cur_ar))
            cur, cur_ar = ch, ar
    if cur:
        runs.append((cur, bool(cur_ar)))
    if not HAS_RAQM or len(runs) == 1:
        # one pass; pick the face by dominant script
        ar = sum(1 for r, a in runs if a) >= len(runs) / 2.0 if runs else False
        font = _font(_face_path(family, weight, ar), size)
        t = _shape(text)
        l, tp, r, b = font.getbbox(t)
        w = int(max(r, font.getlength(t)) + 8)
        h = int(b + size * 0.6 + 8)
        im = Image.new("L", (w, h), 0)
        kw = {}
        if HAS_RAQM:
            kw = dict(direction="rtl" if (rtl and persian.has_arabic_script(text)) else "ltr")
            if persian.has_arabic_script(text):
                kw["language"] = "fa"
        ImageDraw.Draw(im).text((4, 4), t, font=font, fill=255, **kw)
        return np.asarray(im), font
    # raqm + mixed runs: draw each run with its own face, in visual order
    fonts = [(_font(_face_path(family, weight, a), size), r, a) for r, a in runs]
    order = list(reversed(fonts)) if rtl else fonts
    total = int(sum(f.getlength(r) for f, r, a in fonts) + 16)
    h = int(size * 2.2 + 8)
    im = Image.new("L", (total, h), 0)
    d = ImageDraw.Draw(im)
    x = 4
    for f, r, a in order:
        kw = dict(direction="rtl" if a else "ltr")
        if a:
            kw["language"] = "fa"
        d.text((x, 4), r, font=f, fill=255, **kw)
        x += f.getlength(r)
    return np.asarray(im), fonts[0][0]


def ink_extent(mask: np.ndarray):
    ys, xs = np.nonzero(mask > 90)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def observed_mask(img: np.ndarray, box: Box):
    """(binary ink mask of the ink box, fg colour, bg colour, contrast)."""
    c = crop(img, (box[0] - 2, box[1] - 2, box[2] + 4, box[3] + 4), 0)
    ring = np.concatenate([c[0], c[-1], c[:, 0], c[:, -1]])
    bg = np.array(mode_color(ring), np.int16)
    dist = np.abs(c.astype(np.int16) - bg).max(axis=2)
    peak = float(np.percentile(dist, 99.5)) if dist.size else 0.0
    if peak < 8:
        return None
    strong = dist >= 0.72 * peak
    if strong.sum() < 3:
        strong = dist >= 0.5 * peak
    fg = tuple(int(v) for v in np.median(c[strong], axis=0)) if strong.any() else (0, 0, 0)
    # coverage-like soft map in [0,1]
    soft = np.clip(dist / max(peak, 1.0), 0, 1).astype(np.float32)
    return soft, fg, tuple(int(v) for v in bg), peak


def fit_text(img: np.ndarray, box: Box, text: str, rtl: bool, conf: float = 1.0,
             families=None, weights=None) -> TextStyle:
    """Choose (family, weight, size, colour) for one text line / block line."""
    obs = observed_mask(img, box)
    arabic = persian.has_arabic_script(text)
    if families is None:
        families = ["Vazirmatn"] if arabic else ["Inter", "Vazirmatn"]
    cands = []
    for fam in families:
        for (f, w) in FACES:
            if f != fam:
                continue
            if weights and w not in weights:
                continue
            cands.append((f, w))
    best = None
    ow, oh = box[2], box[3]
    soft_o = None
    fg = (0, 0, 0)
    if obs is not None:
        soft_o, fg, bg, peak = obs
        soft_o = soft_o[2:-2, 2:-2] if soft_o.shape[0] > 6 and soft_o.shape[1] > 6 else soft_o
    for fam, w in cands:
        try:
            m100, _ = render_runs(text, fam, w, 100, rtl)
        except Exception:
            continue
        ext = ink_extent(m100)
        if ext is None:
            continue
        x0, y0, x1, y1 = ext
        w100, h100 = x1 - x0, y1 - y0
        if w100 < 2 or h100 < 2:
            continue
        # width-based size, height as a sanity cross-check
        size_w = ow / w100 * 100.0
        size_h = oh / h100 * 100.0
        size = size_w if (conf >= 0.55 and 0.6 < size_w / max(size_h, 0.1) < 1.6) else (0.5 * size_w + 0.5 * size_h if conf >= 0.4 else size_h)
        size = max(6.0, min(size, 400.0))
        score = 0.0
        if soft_o is not None and soft_o.size > 12:
            try:
                isz = max(6, int(round(size)))
                mr, _ = render_runs(text, fam, w, isz, rtl)
                e2 = ink_extent(mr)
                if e2 is not None:
                    rx0, ry0, rx1, ry1 = e2
                    r = (mr[ry0:ry1, rx0:rx1].astype(np.float32) / 255.0)
                    r = cv2.resize(r, (soft_o.shape[1], soft_o.shape[0]), interpolation=cv2.INTER_AREA)
                    a = cv2.GaussianBlur(soft_o, (3, 3), 0.8)
                    b = cv2.GaussianBlur(r, (3, 3), 0.8)
                    a -= a.mean()
                    b -= b.mean()
                    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
                    score = float((a * b).sum() / den) if den > 1e-6 else 0.0
            except Exception:
                score = 0.0
        # tiny prior towards the regular weights / Vazirmatn for Persian pages
        pri = 0.0 if w in (400, 500, 700) else -0.01
        if best is None or score + pri > best[0]:
            best = (score + pri, fam, w, size, (y0 - 4) / 100.0)
    if best is None:
        fam = "Vazirmatn" if arabic else "Inter"
        return TextStyle(family=fam, weight=400, size=max(8.0, oh / 0.9), color=fg)
    _, fam, w, size, inktop = best
    asc, desc = metrics(fam, w)
    return TextStyle(family=fam, weight=w, size=round(size * 2) / 2.0, color=fg, score=best[0], ascent=asc, descent=desc,
                     ink_top=inktop)


def css_weight(w: int) -> int:
    return w
