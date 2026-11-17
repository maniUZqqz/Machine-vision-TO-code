"""Stage 2: text-line candidates from the non-smooth pixels (engine independent).

The boxes found here are *ink* boxes measured from pixels, so they are much tighter than the
boxes OCR engines return.  An OCR engine is only asked to read the crops.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from .imaging import Box, box_inter


@dataclass
class Cand:
    box: Box                 # tight ink box
    n_comp: int = 1          # connected components inside
    ink: float = 0.0         # ink density in the box


def _group_lines(boxes: List[Box]) -> List[List[int]]:
    """Merge word-level boxes that sit on the same line with a small gap."""
    n = len(boxes)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    order = sorted(range(n), key=lambda i: boxes[i][0])
    for ii, i in enumerate(order):
        x, y, w, h = boxes[i]
        for j in order[ii + 1:]:
            x2, y2, w2, h2 = boxes[j]
            gap = x2 - (x + w)
            if gap > max(14, 0.9 * max(h, h2)):
                break
            if gap < -2 and False:
                continue
            ov = min(y + h, y2 + h2) - max(y, y2)
            if ov < 0.55 * min(h, h2):
                continue
            if max(h, h2) > 2.4 * min(h, h2) and min(h, h2) < 12:
                # tiny mark next to a tall glyph (dot, accent): attach only if very close
                if gap > 4:
                    continue
            if gap <= max(6.0, 0.62 * min(h, h2)):
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _rrect_poly(box, r, grow=0):
    x, y, w, h = box
    x0, y0, x1, y1 = x - grow, y - grow, x + w + grow, y + h + grow
    r = max(0.0, min(r + grow, (x1 - x0) / 2.0, (y1 - y0) / 2.0))
    if r < 1:
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.int32)
    pts = []
    n = max(6, int(r))
    for cx, cy, a0 in ((x1 - r, y0 + r, -90), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180)):
        for k in range(n + 1):
            a = np.deg2rad(a0 + 90.0 * k / n)
            pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    return np.round(np.array(pts)).astype(np.int32)


def clear_region_edges(ink: np.ndarray, regions, thick: int = 3) -> None:
    """Erase the outline (border / anti-aliased rounded corners) of every region from the ink mask."""
    m_out = np.zeros_like(ink)
    m_in = np.zeros_like(ink)
    for r in regions:
        if r.box[2] < 6 or r.box[3] < 6:
            continue
        bwid = max([v[0] for v in (r.border or {}).values()] or [0])
        poly_o = _rrect_poly(r.box, r.radius, 1)
        poly_i = _rrect_poly(r.inner, max(0.0, r.radius - bwid), -thick)
        m_out[:] = 0
        m_in[:] = 0
        x, y, w, h = r.box
        sl = (slice(max(0, y - 4), y + h + 5), slice(max(0, x - 4), x + w + 5))
        if r.shape == "ellipse":
            cv2.ellipse(m_out, (int(x + w / 2), int(y + h / 2)), (int(w / 2) + 2, int(h / 2) + 2), 0, 0, 360, 1, -1)
            cv2.ellipse(m_in, (int(x + w / 2), int(y + h / 2)), (max(1, int(w / 2) - thick), max(1, int(h / 2) - thick)), 0, 0, 360, 1, -1)
        else:
            cv2.fillPoly(m_out, [poly_o], 1)
            if w > 2 * thick + 2 and h > 2 * thick + 2:
                cv2.fillPoly(m_in, [poly_i], 1)
        ring = (m_out[sl] > 0) & ~(m_in[sl] > 0)
        ink[sl][ring] = 0


def find_candidates(img: np.ndarray, S: np.ndarray, photos, line_boxes: List[Box], regions=None) -> List[Cand]:
    H, W = img.shape[:2]
    ink = (~S).astype(np.uint8)
    if regions:
        clear_region_edges(ink, regions)
    for p in photos:
        x, y, w, h = p.box
        ink[max(0, y - 1):y + h + 1, max(0, x - 1):x + w + 1] = 0
    for (x, y, w, h) in line_boxes:
        ink[max(0, y - 2):y + h + 2, max(0, x - 2):x + w + 2] = 0
    for r in regions or []:
        x, y, w, h = r.box
        if 12 <= w <= 30 and 12 <= h <= 30 and abs(w - h) <= 4:
            ink[max(0, y - 1):y + h + 1, max(0, x - 1):x + w + 1] = 0
    # region edges are 1-2px non-smooth lines too: erase the strongest straight ones
    for horizontal in (True, False):
        k = (40, 1) if horizontal else (1, 40)
        o = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, k))
        # only remove lines that have empty space on both sides (not text baselines)
        n, lab, st, _ = cv2.connectedComponentsWithStats(o, connectivity=8)
        for i in range(1, n):
            x, y, w, h, a = [int(v) for v in st[i]]
            long_, thick = (w, h) if horizontal else (h, w)
            if thick > 6:
                continue
            if horizontal:
                a1 = S[max(0, y - 3):max(0, y - 1), x + 2:x + w - 2]
                a2 = S[min(H, y + h + 1):min(H, y + h + 3), x + 2:x + w - 2]
            else:
                a1 = S[y + 2:y + h - 2, max(0, x - 3):max(0, x - 1)]
                a2 = S[y + 2:y + h - 2, min(W, x + w + 1):min(W, x + w + 3)]
            if (a1.size == 0 or a1.mean() >= 0.9) and (a2.size == 0 or a2.mean() >= 0.9):
                ink[y:y + h, x:x + w] = 0
    return _fix_leaky(img, _lines_from_ink(ink), regions or [])


def _lines_from_ink(ink: np.ndarray) -> List[Cand]:
    """Ink mask (1 = ink) -> tight text-line candidates (shared by the page-wide and the inside-a-button detectors)."""
    # typical glyph component height
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    hs = st[1:, cv2.CC_STAT_HEIGHT]
    ws = st[1:, cv2.CC_STAT_WIDTH]
    sel = hs[(hs >= 4) & (hs <= 60) & (ws <= 90)]
    mh = float(np.median(sel)) if len(sel) else 10.0
    kx = int(np.clip(round(0.45 * mh), 2, 9))
    ky = int(np.clip(round(0.15 * mh), 1, 2))
    dil = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 2 * ky + 1)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(dil, connectivity=8)
    words: List[Box] = []
    for i in range(1, n):
        x, y, w, h, a = [int(v) for v in st[i]]
        if h < 4 or w < 2 or (w < 4 and h < 6):
            continue
        sub = ink[y:y + h, x:x + w] & (lab[y:y + h, x:x + w] == i)
        ys, xs = np.nonzero(sub)
        if len(xs) < 3:
            continue
        words.append((int(x + xs.min()), int(y + ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)))
    out: List[Cand] = []
    for g in _group_lines(words):
        xs0 = min(words[i][0] for i in g)
        ys0 = min(words[i][1] for i in g)
        xs1 = max(words[i][0] + words[i][2] for i in g)
        ys1 = max(words[i][1] + words[i][3] for i in g)
        b = (xs0, ys0, xs1 - xs0, ys1 - ys0)
        dens = float(ink[ys0:ys1, xs0:xs1].mean())
        out.append(Cand(box=b, n_comp=len(g), ink=dens))
    out.sort(key=lambda c: (c.box[1], c.box[0]))
    return out


def _inner_lines(img: np.ndarray, r) -> List[Cand]:
    """Text lines *inside* one button-sized region, found against the region's own local background (median filter), so gradient fills,
    soft shadows and the outline do not matter.  Boxes are in page coordinates."""
    x, y, w, h = r.box
    bwid = max([v[0] for v in (r.border or {}).values()] or [0])
    inset = int(round(max(3.0, 0.28 * r.radius, bwid + 2.0)))
    ix, iy, iw, ih = x + inset, y + inset, w - 2 * inset, h - 2 * inset
    H, W = img.shape[:2]
    ix, iy = max(0, ix), max(0, iy)
    iw, ih = min(iw, W - ix), min(ih, H - iy)
    if iw < 16 or ih < 8:
        return []
    sub = np.ascontiguousarray(img[iy:iy + ih, ix:ix + iw])
    k = int(np.clip(ih // 2 * 2 + 1, 9, 41))
    bg = cv2.medianBlur(sub, k)
    d = np.abs(sub.astype(np.int16) - bg.astype(np.int16)).max(axis=2)
    top = float(np.percentile(d, 99.5))
    if top < 30:
        return []
    ink = (d > max(26, int(0.30 * top))).astype(np.uint8)
    out = []
    for c in _lines_from_ink(ink):
        bx, by, bw, bh = c.box
        if bh < 6 or bw < 3:
            continue
        out.append(Cand(box=(bx + ix, by + iy, bw, bh), n_comp=c.n_comp, ink=c.ink))
    return out


def _fix_leaky(img: np.ndarray, cands: List[Cand], regions) -> List[Cand]:
    """A text candidate that spills out of a button/pill/chip (outline and glow merged with the label, two pills merged into one) cannot be read.
    For such regions the labels are found again inside the region and replace every candidate that touches it."""
    if not regions or not cands:
        return cands
    cands = list(cands)
    for r in regions:
        x, y, w, h = r.box
        if not (16 <= h <= 150 and 40 <= w <= 800) or r.holes_only:
            continue
        touch = [c for c in cands if box_inter(c.box, r.box) > 0]
        if not touch:
            continue
        leaky = [c for c in touch if not (c.box[0] >= x - 3 and c.box[1] >= y - 3 and c.box[0] + c.box[2] <= x + w + 3 and c.box[1] + c.box[3] <= y + h + 3)
                 and box_inter(c.box, r.box) >= 0.15 * min(c.box[2] * c.box[3], w * h)]
        if not leaky:
            continue
        inner = _inner_lines(img, r)
        if not inner:
            continue
        drop = {id(c) for c in touch}
        cands = [c for c in cands if id(c) not in drop] + inner
    cands.sort(key=lambda c: (c.box[1], c.box[0]))
    return cands
