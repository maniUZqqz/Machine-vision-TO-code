"""Stage 1: flat-colour region analysis.

UI screenshots are made of flat or smoothly shaded rectangles.  A pixel is *smooth* when its
3x3 neighbourhood has (almost) no colour change.  Connected smooth areas are the page
backgrounds, cards, buttons, inputs, badges, bars ...; the remaining *non-smooth* pixels are
glyphs, icons, thin lines and photographs.

Everything returned is in processing-pixel coordinates (``Box = (x, y, w, h)``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from .imaging import Box, box_area, box_contains, box_inter, mode_color, cdist, hex_of
from . import gradfit as gf


@dataclass
class Region:
    id: int
    box: Box                       # border-box (fill + border)
    inner: Box                     # fill box
    fill: tuple
    shape: str = "rect"            # rect | ellipse
    radius: float = 0.0
    grad: Optional[dict] = None    # {angle, c1, c2}
    border: Optional[dict] = None  # {top,right,bottom,left: (w, rgb)|None}
    shadow: Optional[dict] = None  # {oy, blur, alpha}
    area: int = 0
    parent: Optional[int] = None
    children: List[int] = field(default_factory=list)
    holes_only: bool = False       # same colour as parent and no border/shadow -> not a visible element


@dataclass
class Photo:
    box: Box
    radius: float = 0.0
    round: bool = False


@dataclass
class Line:
    box: Box
    color: tuple
    horizontal: bool


# --------------------------------------------------------------------------- masks
def estimate_threshold(img: np.ndarray):
    k = np.ones((3, 3), np.uint8)
    rng = (cv2.dilate(img, k).astype(np.int16) - cv2.erode(img, k).astype(np.int16)).max(axis=2)
    frac0 = float((rng == 0).mean())
    if frac0 >= 0.2:
        return 3, rng
    return int(np.clip(2 * np.percentile(rng, 25) + 2, 4, 16)), rng


def smooth_mask(img: np.ndarray, T: Optional[int] = None):
    if T is None:
        T, rng = estimate_threshold(img)
    else:
        k = np.ones((3, 3), np.uint8)
        rng = (cv2.dilate(img, k).astype(np.int16) - cv2.erode(img, k).astype(np.int16)).max(axis=2)
    return rng <= T, T


# --------------------------------------------------------------------------- regions
def _fit_gradient(img_crop: np.ndarray, m: np.ndarray, box_off, strict: bool = True):
    """Fit colour = c0 + g . (x, y) on masked pixels; return a 2-stop CSS-like gradient or None (``strict=False``: best effort, whatever the residual)."""
    ys, xs = np.nonzero(m)
    if len(xs) < 200:
        return None
    step = max(1, len(xs) // 20000)
    xs, ys = xs[::step], ys[::step]
    cols = img_crop[ys, xs].astype(np.float64)
    A = np.stack([xs.astype(np.float64), ys.astype(np.float64), np.ones(len(xs))], 1)
    coef, *_ = np.linalg.lstsq(A, cols, rcond=None)  # 3 x 3 (x,y,1 per channel)
    res = cols - A @ coef
    if strict and (np.median(np.abs(res)) > 2.5 or np.percentile(np.abs(res), 90) > 6 or np.percentile(np.abs(res), 98) > 12):
        return None
    G = coef[:2, :].T  # per-channel gradient (3,2)
    u, s, vt = np.linalg.svd(G, full_matrices=False)
    d = vt[0]  # direction (dx, dy) of greatest colour change
    h, w = img_crop.shape[:2]
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], np.float64)
    t = corners @ d
    p0, p1 = corners[t.argmin()], corners[t.argmax()]
    c0 = np.array([p0[0], p0[1], 1.0]) @ coef
    c1 = np.array([p1[0], p1[1], 1.0]) @ coef
    if np.abs(c1 - c0).max() < (10 if strict else 2):
        return None
    # CSS angle: 0deg = to top, clockwise.  direction vector (dx, dy) with y down.
    ang = (math.degrees(math.atan2(d[0], -d[1])) + 360) % 360
    if np.dot(d, p1 - p0) < 0:
        ang = (ang + 180) % 360
    return dict(angle=round(ang), c1=tuple(int(np.clip(round(v), 0, 255)) for v in c0),
                c2=tuple(int(np.clip(round(v), 0, 255)) for v in c1))


def _clean_pieces(crop, parts, min_area):
    """The connected pieces of a plane split as ``[(mask, piece, ragged_ok), ...]``.

    ``None`` when the gradient pieces are not clean rectangles and no piece dominates (a radial gradient cut into rings by the planes): then
    the area is better kept as ONE gradient region.  When one gradient piece dominates (>= 55% of the gradient area), the other gradient
    pieces that are not rectangles are decoration on it (translucent circles of a hero card): they are dropped, the main gradient shows there."""
    comps = []
    lead_idx = next((i for i, p in enumerate(parts) if not p["flat"]), -1)       # the biggest gradient piece may have ragged edges (cut by neighbours)
    for pi, p in enumerate(parts):
        nn, ll, ss, _ = cv2.connectedComponentsWithStats(p["mask"].astype(np.uint8), connectivity=4)
        for j in range(1, nn):
            area = int(ss[j, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            fr = area / float(max(1, int(ss[j, cv2.CC_STAT_WIDTH]) * int(ss[j, cv2.CC_STAT_HEIGHT])))
            comps.append(dict(mask=ll == j, piece=p, area=area, fr=fr, lead=(pi == lead_idx)))
    grad_area = sum(c["area"] for c in comps if not c["piece"]["flat"])
    lead_area = max([c["area"] for c in comps if c["lead"]] or [0])
    dominant = grad_area > 0 and lead_area >= 0.55 * grad_area
    out = []
    for c in comps:
        if not c["piece"]["flat"]:
            if c["fr"] < (0.45 if c["lead"] else 0.8):
                if dominant and not c["lead"]:
                    continue                     # decoration on the dominant gradient
                return None
        out.append((c["mask"], c["piece"], c["lead"]))
    return out or None


def _fit_gradient_loose(crop, m):
    return _fit_gradient(crop, m, (0, 0), strict=False)


def _region_from_mask(img, crop, m, mx0, my0, grad, regions, ragged_ok=True, verify_ellipse=False):
    """One smooth piece (bool/uint8 mask in crop coordinates) -> a Region appended to ``regions`` (or nothing when it is not a rectangle/ellipse)."""
    H, W = img.shape[:2]
    m = m.astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False
    filled = np.zeros_like(m)
    cv2.drawContours(filled, cnts, -1, 1, -1)
    area = int(m.sum())
    fill = mode_color(crop[m.astype(bool)])
    # restore the 1px lost at edges: grow into pixels of the same colour
    near = (np.abs(crop.astype(np.int16) - np.array(fill, np.int16)).max(2) <= 8) if grad is None else np.ones(m.shape, bool)
    m2 = cv2.dilate(filled, np.ones((3, 3), np.uint8)).astype(bool) & near
    m2 |= filled.astype(bool)
    ys, xs = np.nonzero(m2)
    if len(xs) == 0:
        return False
    bx0, by0, bx1, by1 = xs.min() + mx0, ys.min() + my0, xs.max() + mx0 + 1, ys.max() + my0 + 1
    bw, bh = bx1 - bx0, by1 - by0
    a2 = int(m2.sum())
    fr = a2 / float(bw * bh)
    aspect = max(bw, bh) / max(1, min(bw, bh))
    if 0.68 <= fr < 0.84 and aspect <= 1.35:
        shape, radius = "ellipse", min(bw, bh) / 2.0
        if verify_ellipse or min(bw, bh) >= 40:
            # a big area that only looks round by its fill ratio (the quarter of a translucent circle a card clips): not an ellipse
            ell = np.zeros(m2.shape, np.uint8)
            cv2.ellipse(ell, (int((bx0 - mx0) + bw / 2.0), int((by0 - my0) + bh / 2.0)), (max(1, int(bw / 2.0)), max(1, int(bh / 2.0))), 0, 0, 360, 1, -1)
            e = ell.astype(bool)
            if (m2 & e).sum() / float(max(1, (m2 | e).sum())) < 0.85:
                return False
    elif fr >= 0.80:
        shape = "rect"
        loss = bw * bh - a2
        radius = 0.0 if fr >= 0.992 else min(math.sqrt(max(loss, 0) / 0.8584), min(bw, bh) / 2.0)
    elif grad is not None and ragged_ok and fr >= 0.45 and bw * bh >= 0.02 * W * H:
        # a big gradient area with ragged edges (cut by neighbours of nearly the same colour): still one rectangle, never a picture
        shape, radius = "rect", 0.0
    else:
        return False
    if grad is not None:
        grad = gf.strip_private(grad)
        cols = [grad["c1"], grad["c2"]] + [c for _t, c in (grad.get("stops") or [])]
        if max(max(abs(int(a[k]) - int(b[k])) for k in range(3)) for a in cols for b in cols) < 5:
            grad = None             # practically one colour: a flat region, not a gradient
    regions.append(Region(id=len(regions), box=(bx0, by0, bw, bh), inner=(bx0, by0, bw, bh), fill=fill, shape=shape,
                          radius=round(radius, 1), grad=grad, area=a2))
    return True



def find_regions(img: np.ndarray, S: np.ndarray, min_area: int = 100, soft_images=None, T: int = 3) -> List[Region]:
    H, W = img.shape[:2]
    n, labels, stats, _ = cv2.connectedComponentsWithStats(S.astype(np.uint8), connectivity=4)
    regions: List[Region] = []
    order = np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1
    for i in order:
        x, y, w, h, area = [int(v) for v in stats[i]]
        if area < min_area or w < 5 or h < 5:
            break
        mx0, my0 = max(0, x - 2), max(0, y - 2)
        mx1, my1 = min(W, x + w + 2), min(H, y + h + 2)
        crop = img[my0:my1, mx0:mx1]
        lab = labels[my0:my1, mx0:mx1]
        m = (lab == i).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        filled = np.zeros_like(m)
        cv2.drawContours(filled, cnts, -1, 1, -1)
        fill_area = int(filled.sum())
        fill = mode_color(crop[m.astype(bool)])
        near_frac = float((np.abs(crop[m.astype(bool)].astype(np.int16) - np.array(fill, np.int16)).max(1) <= 2).mean())
        grad = None
        if near_frac < 0.75:
            mb = m.astype(bool)
            grad = _fit_gradient(crop, mb, (mx0, my0))
            tailed = None
            if grad is None and area >= 1500 and min(w, h) >= 24:
                # not one 2-stop plane: a multi-stop / radial gradient, or several things that touch (gradient band + page colour ...)
                grad = gf.fit_general(crop, mb)
                if grad is None:
                    tailed = gf.fit_general(crop, mb, tails_ok=True)     # one gradient that ends in a long flat colour (the area is maybe one band)
                    parts = gf.split_planes(crop, mb, tol=max(3.5, 1.2 * T))
                    pieces = _clean_pieces(crop, parts, min_area) if parts and len(parts) >= 2 else None
                    if pieces is not None:
                        for pm, p, lead in pieces:
                            g = None
                            if not p["flat"]:
                                # re-fit on this piece alone: the stops are relative to its own box
                                g = gf.fit_linear(crop, pm, 3.0, 6.0) or gf.fit_general(crop, pm) or p["grad"]
                            _region_from_mask(img, crop, pm, mx0, my0, g, regions, ragged_ok=lead, verify_ellipse=True)
                        continue
                    if parts and len(parts) == 1:
                        grad = parts[0]["grad"]
                        if grad is None and parts[0]["flat"]:
                            near_frac = 1.0     # one flat colour after all
                    if grad is None:
                        grad = tailed
            if grad is None and near_frac < 0.75 and area >= 1500 and min(w, h) >= 24:
                # a gradient that melts into the page colour (flat tail) or a mesh/soft background: the best single gradient we can fit.  It stays
                # CSS (text and buttons on top stay real HTML); a big smooth area never becomes one picture.
                grad = gf.fit_general(crop, mb, tails_ok=True)
                if grad is None and (area >= 0.18 * W * H or (w >= 0.55 * W and h >= 90)):
                    grad = gf.fit_general(crop, mb, relaxed=True, tails_ok=True) or _fit_gradient_loose(crop, mb)
            if grad is None and near_frac < 0.75:
                # smooth-but-varying blob: a blurry photo / avatar (or a shadow ring: filtered by shape)
                if soft_images is not None and area >= 300 and min(w, h) >= 18:
                    ar = area / float(w * h)
                    if ar >= 0.6 and max(w, h) / max(1, min(w, h)) <= 3.0:
                        soft_images.append(Photo(box=(x, y, w, h), round=(abs(w - h) <= 2 and ar < 0.9),
                                                 radius=min(w, h) / 2.0 if (abs(w - h) <= 2 and ar < 0.9) else 0.0))
                continue
        _region_from_mask(img, crop, m, mx0, my0, grad, regions)
    return regions


def _strip(img, side: str, box: Box, d: int):
    H, W = img.shape[:2]
    x, y, w, h = box
    if side == "top":
        yy = y - d
        if yy < 0:
            return None
        a, b = x + int(w * 0.2), x + int(w * 0.8)
        return img[yy, a:b]
    if side == "bottom":
        yy = y + h - 1 + d
        if yy >= H:
            return None
        a, b = x + int(w * 0.2), x + int(w * 0.8)
        return img[yy, a:b]
    if side == "left":
        xx = x - d
        if xx < 0:
            return None
        a, b = y + int(h * 0.2), y + int(h * 0.8)
        return img[a:b, xx]
    xx = x + w - 1 + d
    if xx >= W:
        return None
    a, b = y + int(h * 0.2), y + int(h * 0.8)
    return img[a:b, xx]


def _strip_color(strip):
    if strip is None or len(strip) == 0:
        return None
    return tuple(int(v) for v in np.median(strip, axis=0))


def analyse_outline(img: np.ndarray, r: Region):
    """Border (1-6 px) and soft shadow around the fill box of ``r``.  Updates ``r`` in place."""
    SIDES = ("top", "right", "bottom", "left")
    prof = {s: [_strip_color(_strip(img, s, r.inner, d)) for d in range(1, 22)] for s in SIDES}
    borders = {}
    for s in SIDES:
        p = prof[s]
        if p[0] is None or p[1] is None:
            continue
        c1 = p[0]
        if cdist(c1, r.fill) < 6:
            continue
        bw = 1
        while bw < 6 and p[bw] is not None and cdist(p[bw], c1) <= 5:
            bw += 1
        nxt = p[bw] if bw < len(p) else None
        if nxt is None or cdist(nxt, c1) < 5:
            continue
        # a border is a *crisp* line: the colour right after it must be (close to) constant
        far = p[min(bw + 3, len(p) - 1)]
        if far is not None and cdist(nxt, far) > 12:
            continue
        borders[s] = (bw, c1)
    # faint / sub-pixel borders (anti-aliased over two columns): look at the 85th-percentile deviation of a
    # thin strip; a real hairline deviates at d=1 but not at d=3 (a shadow would deviate at both)
    for s in SIDES:
        if s in borders:
            continue
        st1 = _strip(img, s, r.inner, 1)
        st3 = _strip(img, s, r.inner, 3)
        if st1 is None or st3 is None or len(st1) < 4:
            continue
        fa = np.array(r.fill, np.int16)
        dev1 = np.abs(st1.astype(np.int16) - fa).max(axis=1)
        dev3 = np.abs(st3.astype(np.int16) - fa).max(axis=1)
        if np.percentile(dev1, 85) >= 8 and np.percentile(dev3, 85) <= 5:
            idx = int(np.argsort(dev1)[int(0.85 * (len(dev1) - 1))])
            borders[s] = (1, tuple(int(v) for v in st1[idx]))
    # a one-sided "border" must differ crisply from the parent on the other side; keep as is
    # shadow: darker-than-surroundings, smoothly fading, not a crisp border
    shadow = None
    devs = []
    for s in SIDES:
        if s in borders:
            continue
        p = prof[s]
        if p[0] is None or p[-1] is None:
            continue
        ref = np.array(p[-1], np.float64)
        d = [float((ref - np.array(c, np.float64)).mean()) if c is not None else 0.0 for c in p]
        devs.append((s, d))
    if len(devs) >= 3 and "bottom" not in borders:
        good = [d for s, d in devs if d[0] >= 3.0 and d[0] >= d[6] - 0.5 and all(d[i] + 2.5 >= d[i + 1] for i in range(len(d) - 1))]
        if len(good) >= 2:
            ext = []
            for d in good:
                e = next((i for i, v in enumerate(d) if v <= 1.0), len(d))
                ext.append(e)
            bot = dict(devs).get("bottom")
            top = dict(devs).get("top")
            ob = next((i for i, v in enumerate(bot) if v <= 1.0), 21) if bot else 8
            ot = next((i for i, v in enumerate(top) if v <= 1.0), 21) if top else 8
            oy = int(round((ob - ot) / 2.0))
            blur = int(np.clip(round(sum(ext) / len(ext) * 1.3), 4, 40))
            lvl = max(1.0, float(np.mean(prof["top"][-1] if prof["top"][-1] is not None else prof["bottom"][-1])))
            peak = max(d[0] for d in good)
            alpha = float(np.clip(2.2 * peak / lvl, 0.04, 0.6))
            shadow = dict(oy=max(0, oy), blur=blur, alpha=round(alpha, 2))
    if r.shape == "ellipse" and not borders and max(r.inner[2], r.inner[3]) <= 34:
        # a ring around a small circle (radio button): probe the 4 axis points one pixel out
        x, y, w, h = r.inner
        pts = [(x + w // 2, y - 1), (x + w // 2, y + h), (x - 1, y + h // 2), (x + w, y + h // 2)]
        cols = [tuple(int(v) for v in img[py, px_]) for px_, py in pts if 0 <= px_ < img.shape[1] and 0 <= py < img.shape[0]]
        if len(cols) == 4 and all(cdist(c, cols[0]) <= 6 for c in cols) and cdist(cols[0], r.fill) >= 8:
            borders = {s: (1, cols[0]) for s in SIDES}
    r.border = borders or None
    r.shadow = shadow
    x, y, w, h = r.inner
    t = borders.get("top", (0, 0))[0]
    b = borders.get("bottom", (0, 0))[0]
    l = borders.get("left", (0, 0))[0]
    rr = borders.get("right", (0, 0))[0]
    r.box = (x - l, y - t, w + l + rr, h + t + b)


def corner_radius(img: np.ndarray, box: Box, fill=None, thr: int = 14) -> float:
    """Border-radius of ``box`` from how far the corner pixel is from the shape (diagonal probe)."""
    H, W = img.shape[:2]
    x, y, w, h = box
    ks = []
    for (cx, cy, dx, dy) in ((x, y, 1, 1), (x + w - 1, y, -1, 1), (x, y + h - 1, 1, -1), (x + w - 1, y + h - 1, -1, -1)):
        if not (0 < cx < W - 1 and 0 < cy < H - 1):
            continue
        ref = img[cy, cx].astype(np.int16)
        if fill is not None:
            thr_ = int(max(4, min(thr, 0.5 * np.abs(ref - np.array(fill, np.int16)).max())))
        else:
            thr_ = thr
        k = 0
        lim = int(min(w, h) / 2) + 1
        while k < lim:
            px, py = cx + dx * k, cy + dy * k
            if not (0 <= px < W and 0 <= py < H):
                break
            if np.abs(img[py, px].astype(np.int16) - ref).max() > thr_:
                break
            k += 1
        ks.append(k)
    if not ks:
        return 0.0
    k = float(np.median(ks))
    if k <= 0.5:
        return 0.0
    r = min(3.41 * k + 1.0, min(w, h) / 2.0)
    if min(w, h) <= 48 and r >= 0.7 * min(w, h) / 2.0:
        r = min(w, h) / 2.0  # pill
    return float(round(r / 2.0) * 2)


def _mk_region(rid, box, fill):
    return Region(id=rid, box=box, inner=box, fill=fill, area=box[2] * box[3])


def merge_stacks(img: np.ndarray, regions: List[Region]) -> List[Region]:
    """Cards split by row rules / column rules (tables, lists) -> one bordered region."""
    out = list(regions)
    for axis in (0, 1):
        used = set()
        cand = sorted(out, key=lambda r: (r.inner[1], r.inner[0]) if axis == 0 else (r.inner[0], r.inner[1]))
        chains = []
        for a in cand:
            if a.id in used:
                continue
            chain = [a]
            used.add(a.id)
            grown = True
            while grown:
                grown = False
                last = chain[-1]
                lx, ly, lw, lh = last.inner
                for b in cand:
                    if b.id in used:
                        continue
                    bx, by, bw, bh = b.inner
                    if axis == 0:
                        same = abs(bx - lx) <= 2 and abs(bw - lw) <= 3
                        gap = by - (ly + lh)
                    else:
                        same = abs(by - ly) <= 2 and abs(bh - lh) <= 3
                        gap = bx - (lx + lw)
                    if same and 0 <= gap <= 5:
                        chain.append(b)
                        used.add(b.id)
                        grown = True
                        break
            if len(chain) >= 2:
                chains.append(chain)
        for chain in chains:
            xs0 = min(c.inner[0] for c in chain)
            ys0 = min(c.inner[1] for c in chain)
            xs1 = max(c.inner[0] + c.inner[2] for c in chain)
            ys1 = max(c.inner[1] + c.inner[3] for c in chain)
            U = (xs0, ys0, xs1 - xs0, ys1 - ys0)
            tmp = _mk_region(-1, U, chain[0].fill)
            analyse_outline(img, tmp)
            nb = len(tmp.border or {})
            if nb < 3 and not tmp.shadow:
                continue
            # majority fill
            by_col = {}
            for c in chain:
                by_col.setdefault(c.fill, 0)
                by_col[c.fill] += c.area
            fill = max(by_col, key=by_col.get)
            rid = max(r.id for r in out) + 1
            u = Region(id=rid, box=tmp.box, inner=U, fill=fill, area=U[2] * U[3], border=tmp.border, shadow=tmp.shadow)
            ids = {c.id for c in chain if cdist(c.fill, fill) <= 6}
            out = [r for r in out if r.id not in ids] + [u]
    for i, r in enumerate(out):
        r.id = i
    return out


def prune_regions(img: np.ndarray, regions: List[Region], W: int, H: int, bg) -> List[Region]:
    """Drop 'holes': regions that are the same colour as their parent and show no border/shadow."""
    build_tree(regions, W, H)
    byid = {r.id: r for r in regions}
    keep = []
    for r in regions:
        par = byid.get(r.parent)
        pfill = par.fill if par is not None else bg
        if cdist(r.fill, pfill) <= 6 and not r.grad:
            borders = dict(r.border or {})
            if par is not None:
                # a "border" that is the parent's own border is not ours
                for s, (i, side) in (("top", (1, 0)), ("bottom", (1, 0)), ("left", (0, 0)), ("right", (0, 0))):
                    pass
            nb = len(borders)
            band = (r.box[2] >= 0.6 * W or r.box[3] >= 0.6 * H) and nb >= 1
            small_ctl = max(r.box[2], r.box[3]) <= 34 and abs(r.box[2] - r.box[3]) <= 4 and (nb >= 2 or (r.shape == "ellipse" and nb >= 1))
            if small_ctl:
                pass
            elif nb >= 3 and not _flush_parent(r, par):
                pass
            elif r.shadow or band:
                pass
            else:
                continue
        keep.append(r)
    for i, r in enumerate(keep):
        r.id = i
    build_tree(keep, W, H)
    kb = {r.id: r for r in keep}
    for r in keep:
        pp = kb.get(r.parent)
        if pp is not None:
            px, py, pw, ph = pp.inner
            x, y, w, h = r.box
            flush = [abs(x - px) <= 1, abs(x + w - (px + pw)) <= 1, abs(y - py) <= 1, abs(y + h - (py + ph)) <= 1]
            if sum(flush) >= 2:
                r.radius = 0.0
        if not r.border:
            continue
        par = kb.get(r.parent)
        for side in list(r.border):
            bw = r.border[side][0]
            x, y, w, h = r.inner
            if par is not None:
                px, py, pw, ph = par.inner
                flush = {"top": abs(y - py) <= 1, "bottom": abs(y + h - (py + ph)) <= 1,
                         "left": abs(x - px) <= 1, "right": abs(x + w - (px + pw)) <= 1}[side]
                if flush:
                    del r.border[side]
                    continue
            # a border shared with a smaller neighbour belongs to the neighbour
            qx, qy = {"top": (x + w // 2, y - bw - 1), "bottom": (x + w // 2, y + h + bw),
                      "left": (x - bw - 1, y + h // 2), "right": (x + w + bw, y + h // 2)}[side]
            for o in keep:
                if o is r or o.area >= r.area:
                    continue
                ox, oy, ow, oh = o.inner
                if ox <= qx < ox + ow and oy <= qy < oy + oh and o.id != r.parent:
                    del r.border[side]
                    break
        if not r.border:
            r.border = None
        x, y, w, h = r.inner
        t = (r.border or {}).get("top", (0, 0))[0]
        b = (r.border or {}).get("bottom", (0, 0))[0]
        l = (r.border or {}).get("left", (0, 0))[0]
        rr = (r.border or {}).get("right", (0, 0))[0]
        r.box = (x - l, y - t, w + l + rr, h + t + b)
    return keep


def _flush_parent(r: Region, par) -> bool:
    """True when most of r's border sides are really the parent's border (r touches parent edges)."""
    if par is None:
        return False
    px, py, pw, ph = par.inner
    x, y, w, h = r.box
    touch = [abs(x - px) <= 1, abs(x + w - (px + pw)) <= 1, abs(y - py) <= 1, abs(y + h - (py + ph)) <= 1]
    return sum(touch) >= 2


def build_tree(regions: List[Region], W: int, H: int):
    """Nest regions by bbox containment.  Returns list of root ids (children of the page)."""
    regs = sorted(regions, key=lambda r: -box_area(r.box))
    for r in regions:
        r.parent = None
        r.children = []
    for i, r in enumerate(regs):
        best = None
        for q in regs[:i]:
            if q is r:
                continue
            if box_contains(q.box, r.box, 2) and box_area(q.box) > box_area(r.box) * 1.02:
                if best is None or box_area(q.box) < box_area(best.box):
                    best = q
        if best is not None:
            r.parent = best.id
            best.children.append(r.id)
    return [r.id for r in regs if r.parent is None]


# --------------------------------------------------------------------------- photos / lines
def find_photos(img: np.ndarray, S: np.ndarray, min_side: int = 36) -> List[Photo]:
    N = (~S).astype(np.float32)
    dens = cv2.boxFilter(N, -1, (21, 21))
    m = (dens >= 0.55).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((min_side // 2, min_side // 2), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    photos: List[Photo] = []
    for i in range(1, n):
        x, y, w, h, a = [int(v) for v in st[i]]
        if w < min_side or h < min_side:
            continue
        sub = (lab[y:y + h, x:x + w] == i) & (N[y:y + h, x:x + w] > 0)
        ys, xs = np.nonzero(sub)
        if len(xs) < 0.3 * w * h:
            continue
        bx0, by0, bx1, by1 = xs.min() + x, ys.min() + y, xs.max() + x + 1, ys.max() + y + 1
        bw, bh = bx1 - bx0, by1 - by0
        if bw < min_side or bh < min_side:
            continue
        # corner radius from the loss of non-smooth area in the bbox
        cov = float(sub[ys.min():ys.max() + 1, xs.min():xs.max() + 1].mean())
        photos.append(Photo(box=(int(bx0), int(by0), int(bw), int(bh))))
        ph = photos[-1]
        # corner radius: first non-smooth pixel on the top row, measured from the left edge
        top = np.nonzero(sub[ys.min()])[0]
        if len(top):
            ph.radius = float(min(max(0, int(top.min()) - int(xs.min())) * 1.6, min(bw, bh) / 2))
        if abs(bw - bh) <= 2 and cov < 0.9 and cov > 0.7:
            ph.round = True
            ph.radius = min(bw, bh) / 2.0
    return photos


def find_lines(img: np.ndarray, S: np.ndarray, regions: List[Region], photos: List[Photo], min_len: int = 30) -> List[Line]:
    """Thin horizontal/vertical rules that are not region borders (dividers, table rules)."""
    H, W = img.shape[:2]
    N = (~S).astype(np.uint8)
    out: List[Line] = []
    for horizontal in (True, False):
        k = (min_len, 1) if horizontal else (1, min_len)
        o = cv2.morphologyEx(N, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, k))
        n, lab, st, _ = cv2.connectedComponentsWithStats(o, connectivity=8)
        for i in range(1, n):
            x, y, w, h, a = [int(v) for v in st[i]]
            long_, thick = (w, h) if horizontal else (h, w)
            if long_ < min_len or thick > 6 or (not horizontal and long_ < 40) or long_ < 12 * max(1, thick - 2):
                continue
            if any(box_inter((x, y, w, h), p.box) > 0 for p in photos):
                continue
            # centre line colour (farthest from the outer halo)
            if horizontal:
                row = img[y + h // 2, x + 2:x + w - 2]
            else:
                row = img[y + 2:y + h - 2, x + w // 2]
            if len(row) == 0:
                continue
            col = tuple(int(v) for v in np.median(row, axis=0))
            t = max(1, thick - 2)
            # a real rule has empty (smooth) space on both sides; a text baseline does not
            if horizontal:
                a = S[max(0, y - 3):max(0, y - 1), x + 2:x + w - 2]
                b = S[min(H, y + h + 1):min(H, y + h + 3), x + 2:x + w - 2]
            else:
                a = S[y + 2:y + h - 2, max(0, x - 3):max(0, x - 1)]
                b = S[y + 2:y + h - 2, min(W, x + w + 1):min(W, x + w + 3)]
            if (a.size and a.mean() < 0.9) or (b.size and b.mean() < 0.9):
                continue
            box = (x, y + (h - t) // 2, w, t) if horizontal else (x + (w - t) // 2, y, t, h)
            out.append(Line(box=box, color=col, horizontal=horizontal))
    # drop the ones that are region borders (within 3px of a region edge, same span)
    keep: List[Line] = []
    for ln in out:
        x, y, w, h = ln.box
        is_border = False
        for r in regions:
            bx, by, bw, bh = r.box
            if ln.horizontal:
                for ey in (by, by + bh):
                    if abs(y - ey) <= 3 and x >= bx - 3 and x + w <= bx + bw + 3:
                        is_border = True
            else:
                for ex in (bx, bx + bw):
                    if abs(x - ex) <= 3 and y >= by - 3 and y + h <= by + bh + 3:
                        is_border = True
            if is_border:
                break
        if not is_border:
            keep.append(ln)
    return keep


def page_background(img: np.ndarray):
    H, W = img.shape[:2]
    per = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
    return mode_color(per)


def finalize_radius(regions: List[Region]) -> None:
    """Bands that sit flush inside their parent (table header, list rows) are not rounded."""
    byid = {r.id: r for r in regions}
    for r in regions:
        pp = byid.get(r.parent)
        if pp is None or r.shape != "rect":
            continue
        px_, py_, pw_, ph_ = pp.inner
        x, y, w, h = r.box
        flush = [abs(x - px_) <= 1, abs(x + w - (px_ + pw_)) <= 1, abs(y - py_) <= 1, abs(y + h - (py_ + ph_)) <= 1]
        if sum(flush) >= 2:
            r.radius = 0.0
