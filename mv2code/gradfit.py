"""Gradient models for smooth colour areas (hero bands, CTA bands, gradient cards).

``segment.find_regions`` first tries one linear two-stop plane (``segment._fit_gradient``).  What that cannot explain lands here:

* ``fit_general``  - a *linear* gradient with up to 5 stops (any angle) or a *radial* gradient (any centre), fitted from the pixels;
* ``split_planes`` - a smooth area that is several things glued together by colour neighbourhood (a gradient hero whose end colour equals
  the page background, a flat card that touches a gradient band ...) is split into flat / planar pieces, each becomes its own region.

A fitted gradient is a dict the emitter understands (``emit.css_gradient``):

    {"kind": "linear", "angle": 135, "c1": (r,g,b), "c2": (r,g,b), "stops": [(0.0, rgb), (0.55, rgb), (1.0, rgb)]}
    {"kind": "radial", "angle": 0, "cx": 0.7, "cy": 0.3, "c1": ..., "c2": ..., "stops": [(0.0, rgb), (1.0, rgb)]}   # circle farthest-corner

``stops`` is optional for the plain two-stop case (``c1``/``c2`` then are the whole truth, as before 2.3).
Everything is plain numpy; no randomness except a fixed-seed generator (the same image always gives the same result).
"""
from __future__ import annotations

import math
from typing import List, Optional

import cv2
import numpy as np

MAX_STOPS = 5
_NB = 40            # quantile bins along the gradient axis


def _samples(m: np.ndarray, limit: int = 24000):
    ys, xs = np.nonzero(m)
    step = max(1, len(xs) // limit)
    return xs[::step], ys[::step]


# --------------------------------------------------------------------------- 1-D profile
def _profile(t: np.ndarray, cols: np.ndarray):
    """Quantile-binned median colour along ``t``; returns (res_median, res_p90, tc, cc)."""
    n = len(t)
    if n < 300:
        return 99.0, 99.0, None, None
    order = np.argsort(t, kind="stable")
    ts, cs = t[order], cols[order]
    nb = _NB if n >= _NB * 10 else max(6, n // 10)
    edges = np.linspace(0, n, nb + 1).astype(int)
    tc = np.empty(nb)
    cc = np.empty((nb, 3))
    for b in range(nb):
        a, z = edges[b], edges[b + 1]
        tc[b] = float(np.median(ts[a:z]))
        cc[b] = np.median(cs[a:z], axis=0)
    # bins can share one t value (flat areas): np.interp needs increasing x
    keep = np.concatenate([[True], np.diff(tc) > 1e-9])
    tc, cc = tc[keep], cc[keep]
    if len(tc) < 3:
        return 99.0, 99.0, None, None
    pred = np.stack([np.interp(ts, tc, cc[:, k]) for k in range(3)], 1)
    res = np.abs(cs - pred).max(axis=1)
    return float(np.median(res)), float(np.percentile(res, 90)), tc, cc


def _simplify(tc: np.ndarray, cc: np.ndarray, tol: float = 3.2):
    """Douglas-Peucker over the binned profile (colour error in max-channel levels)."""
    n = len(tc)
    keep = {0, n - 1}

    def rec(i, j):
        if j <= i + 1:
            return
        k, dmax = -1, 0.0
        for q in range(i + 1, j):
            f = (tc[q] - tc[i]) / max(tc[j] - tc[i], 1e-9)
            d = float(np.abs(cc[q] - (cc[i] + f * (cc[j] - cc[i]))).max())
            if d > dmax:
                k, dmax = q, d
        if dmax > tol:
            keep.add(k)
            rec(i, k)
            rec(k, j)
    rec(0, n - 1)
    idx = sorted(keep)
    if len(idx) > MAX_STOPS:                      # keep the biggest corners only
        idx = idx[:1] + sorted(sorted(idx[1:-1], key=lambda q: -_corner(tc, cc, idx, q))[:MAX_STOPS - 2]) + idx[-1:]
    return tc[idx], cc[idx]


def _corner(tc, cc, idx, q) -> float:
    p = idx.index(q)
    a, b = idx[p - 1], idx[p + 1]
    f = (tc[q] - tc[a]) / max(tc[b] - tc[a], 1e-9)
    return float(np.abs(cc[q] - (cc[a] + f * (cc[b] - cc[a]))).max())


def flat_share(stops) -> float:
    """Share of the gradient line (0..1) that is one flat colour at its head or tail (a gradient that ran into the page colour)."""
    def same(a, b):
        return max(abs(a[k] - b[k]) for k in range(3)) <= 3
    share = 0.0
    if len(stops) >= 3:
        if same(stops[-1][1], stops[-2][1]):
            share += 1.0 - stops[-2][0]
        if same(stops[0][1], stops[1][1]):
            share += stops[1][0]
    return share


def _stops(tc: np.ndarray, cc: np.ndarray):
    """Simplified stops stretched to t = 0 and t = 1 (the profile may cover only the middle: the end of a cut piece)."""
    st, sc = _simplify(tc, cc)

    def edge(i0, i1, t_end):
        a, b = (st[i0], sc[i0]), (st[i1], sc[i1])
        if abs(b[0] - a[0]) < 1e-6:
            return a[1]
        return a[1] + (t_end - a[0]) * (b[1] - a[1]) / (b[0] - a[0])
    pts = [(float(t), tuple(float(v) for v in c)) for t, c in zip(st, sc)]
    if pts[0][0] > 0.02:
        pts.insert(0, (0.0, tuple(float(v) for v in edge(0, 1, 0.0))))
    else:
        pts[0] = (0.0, pts[0][1])
    if pts[-1][0] < 0.98:
        pts.append((1.0, tuple(float(v) for v in edge(len(st) - 2, len(st) - 1, 1.0))))
    else:
        pts[-1] = (1.0, pts[-1][1])
    out = []
    for t, c in pts:
        rgb = tuple(int(np.clip(round(v), 0, 255)) for v in c)
        out.append((round(min(1.0, max(0.0, t)), 3), rgb))
    return out


# --------------------------------------------------------------------------- linear
def _line_t(xs, ys, w, h, ang_deg):
    a = math.radians(ang_deg)
    dx, dy = math.sin(a), -math.cos(a)
    length = abs(w * dx) + abs(h * dy)
    if length < 1e-6:
        return None
    return ((xs - w / 2.0) * dx + (ys - h / 2.0) * dy) / length + 0.5


def _plane_angle(xs, ys, cols) -> Optional[float]:
    A = np.stack([xs.astype(np.float64), ys.astype(np.float64), np.ones(len(xs))], 1)
    coef, *_ = np.linalg.lstsq(A, cols.astype(np.float64), rcond=None)
    G = coef[:2, :].T
    _u, s, vt = np.linalg.svd(G, full_matrices=False)
    if s[0] < 1e-6:
        return None
    d = vt[0]
    ang = (math.degrees(math.atan2(d[0], -d[1])) + 360) % 360
    # orient it so that the first channel pair increases along the line (the sign does not matter for the fit, only for c1/c2 order)
    return ang


def fit_linear(crop: np.ndarray, m: np.ndarray, med_tol: float = 2.0, p90_tol: float = 5.0):
    xs, ys = _samples(m)
    if len(xs) < 300:
        return None
    x0, y0 = int(xs.min()), int(ys.min())
    w, h = int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1
    if w < 8 or h < 8:
        return None
    cols = crop[ys, xs].astype(np.float64)
    xs0, ys0 = (xs - x0).astype(np.float64), (ys - y0).astype(np.float64)
    angs = [0, 45, 90, 135, 180, 225, 270, 315]
    pa = _plane_angle(xs0, ys0, cols)
    cand = []
    if pa is not None:
        cand += [pa, (pa + 12) % 360, (pa - 12) % 360]
    cand += angs
    best = None
    for ang in cand:
        t = _line_t(xs0, ys0, w, h, ang)
        if t is None:
            continue
        med, p90, tc, cc = _profile(t, cols)
        if tc is None:
            continue
        score = med + 0.3 * p90
        if best is None or score < best[0]:
            best = (score, med, p90, ang, tc, cc)
    if best is None:
        return None
    _score, med, p90, ang, tc, cc = best
    if med > med_tol or p90 > p90_tol:
        return None
    stops = _stops(tc, cc)
    if abs(stops[0][1][0] - stops[-1][1][0]) + abs(stops[0][1][1] - stops[-1][1][1]) + abs(stops[0][1][2] - stops[-1][1][2]) < 6 and len(stops) <= 2:
        return None
    return dict(kind="linear", angle=int(round(ang)) % 360, c1=stops[0][1], c2=stops[-1][1], stops=stops,
                _res=(round(med, 2), round(p90, 2)), _bbox=(x0, y0, w, h), _flat=flat_share(stops))


# --------------------------------------------------------------------------- radial
def _radial_t(xs, ys, w, h, cx, cy):
    r = np.hypot(xs - cx * w, ys - cy * h)
    corners = [(0, 0), (w, 0), (0, h), (w, h)]
    rfc = max(math.hypot(px - cx * w, py - cy * h) for px, py in corners)
    return r / max(rfc, 1e-6)


def fit_radial(crop: np.ndarray, m: np.ndarray, med_tol: float = 2.2, p90_tol: float = 6.0):
    xs, ys = _samples(m, 12000)
    if len(xs) < 300:
        return None
    x0, y0 = int(xs.min()), int(ys.min())
    w, h = int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1
    if w < 16 or h < 16:
        return None
    cols = crop[ys, xs].astype(np.float64)
    xs0, ys0 = (xs - x0).astype(np.float64), (ys - y0).astype(np.float64)

    def score(cx, cy):
        med, p90, tc, cc = _profile(_radial_t(xs0, ys0, w, h, cx, cy), cols)
        return (med + 0.3 * p90 if tc is not None else 1e9), med, p90, tc, cc
    best = None
    grid = [-0.2, 0.0, 0.25, 0.5, 0.75, 1.0, 1.2]
    for cx in grid:
        for cy in grid:
            s = score(cx, cy)
            if best is None or s[0] < best[0][0]:
                best = (s, cx, cy)
    (s, cx, cy) = best
    step = 0.125
    for _ in range(3):
        improved = True
        while improved:
            improved = False
            for dx, dy in ((step, 0), (-step, 0), (0, step), (0, -step), (step, step), (-step, -step), (step, -step), (-step, step)):
                s2 = score(cx + dx, cy + dy)
                if s2[0] < s[0] - 0.05:
                    s, cx, cy, improved = s2, cx + dx, cy + dy, True
        step /= 2.0
    _sc, med, p90, tc, cc = s
    if tc is None or med > med_tol or p90 > p90_tol:
        return None
    stops = _stops(tc, cc)
    if len(stops) < 2 or sum(abs(a - b) for a, b in zip(stops[0][1], stops[-1][1])) < 8:
        return None
    return dict(kind="radial", angle=0, cx=round(cx, 3), cy=round(cy, 3), c1=stops[0][1], c2=stops[-1][1], stops=stops,
                _res=(round(med, 2), round(p90, 2)), _bbox=(x0, y0, w, h), _flat=flat_share(stops))


def fit_general(crop: np.ndarray, m: np.ndarray, relaxed: bool = False, tails_ok: bool = False):
    """Best of linear (multi-stop) and radial.  ``relaxed`` accepts a worse fit (a big smooth area must never become a picture).  A fit whose
    gradient runs into a long flat colour (a hero that melts into the page background) is refused unless ``tails_ok``: such an area is better
    split into the gradient band and the flat page (``split_planes``)."""
    lin = fit_linear(crop, m, *((6.0, 14.0) if relaxed else (2.0, 5.0)))
    if lin is not None and not tails_ok and lin["_flat"] >= 0.12:
        lin = None
    if lin is not None and lin["_res"][0] <= 1.2 and lin["_res"][1] <= 3.0 and not relaxed:
        return lin
    rad = fit_radial(crop, m, *((6.0, 14.0) if relaxed else (2.2, 6.0)))
    if rad is not None and not tails_ok and rad["_flat"] >= 0.12:
        rad = None
    cands = [g for g in (lin, rad) if g is not None]
    if not cands:
        return None
    return min(cands, key=lambda g: g["_res"][0] + 0.3 * g["_res"][1])


# --------------------------------------------------------------------------- planes
def _fit_plane3(P, C):
    """Plane (per channel) through 3 points; None when they are (nearly) collinear."""
    A = np.concatenate([P, np.ones((3, 1))], 1)
    if abs(np.linalg.det(A)) < 2000.0:
        return None
    return np.linalg.solve(A, C)


def _plane_inliers(xs, ys, cols, coef, tol):
    pred = xs[:, None] * coef[0] + ys[:, None] * coef[1] + coef[2]
    return np.abs(cols - pred).max(axis=1) <= tol


def split_planes(crop: np.ndarray, m: np.ndarray, tol: float = 3.5, min_share: float = 0.02, max_planes: int = 4, seed: int = 7):
    """Split a smooth blob into pieces that are each (almost) one colour plane (flat or a linear gradient).

    Returns ``[{"mask": bool array (crop coords), "grad": dict|None, "flat": bool}, ...]`` or ``None`` when the planes do not explain
    at least 85% of the blob (a blurry photo, a mesh gradient ...).
    """
    ys_all, xs_all = np.nonzero(m)
    total = len(xs_all)
    if total < 400:
        return None
    rng = np.random.RandomState(seed)
    sub = _samples(m, 5000)
    sxs, sys_ = sub[0].astype(np.float64), sub[1].astype(np.float64)
    scols = crop[sub[1], sub[0]].astype(np.float64)
    alive = np.ones(len(sxs), bool)
    pieces = []
    covered = np.zeros(total, bool)
    fx, fy = xs_all.astype(np.float64), ys_all.astype(np.float64)
    fcols = crop[ys_all, xs_all].astype(np.float64)
    for _ in range(max_planes):
        idx = np.nonzero(alive)[0]
        if len(idx) < max(60, 0.03 * len(sxs)):
            break
        best, best_n = None, 0
        for _try in range(160):
            tri = idx[rng.randint(0, len(idx), 3)]
            coef = _fit_plane3(np.stack([sxs[tri], sys_[tri]], 1), scols[tri])
            if coef is None:
                continue
            n = int(_plane_inliers(sxs[idx], sys_[idx], scols[idx], coef, tol).sum())
            if n > best_n:
                best, best_n = coef, n
        if best is None or best_n < max(40, min_share * len(sxs)):
            break
        # refine on the inliers (least squares), then re-count on every pixel of the blob that is not taken yet
        inl = idx[_plane_inliers(sxs[idx], sys_[idx], scols[idx], best, tol)]
        A = np.stack([sxs[inl], sys_[inl], np.ones(len(inl))], 1)
        coef, *_ = np.linalg.lstsq(A, scols[inl], rcond=None)
        inl_all = _plane_inliers(fx, fy, fcols, coef, tol) & ~covered
        if int(inl_all.sum()) < max(120, min_share * total):
            alive[inl] = False
            continue
        covered |= inl_all
        alive &= ~_plane_inliers(sxs, sys_, scols, coef, tol)
        pm = np.zeros(m.shape, bool)
        pm[ys_all[inl_all], xs_all[inl_all]] = True
        flat = bool(np.abs(coef[0]).max() * m.shape[1] < 3.0 and np.abs(coef[1]).max() * m.shape[0] < 3.0)
        pieces.append(dict(mask=pm, coef=coef, flat=flat, n=int(inl_all.sum())))
    if not pieces or covered.sum() < 0.85 * total:
        return None
    # exclusive assignment: every pixel goes to the plane that explains it best (the bigger plane wins ties), so a flat header never leaks into
    # a gradient that happens to pass through its colour at the far end; a majority vote over a small window removes speckle at the seams
    K = len(pieces)
    pieces.sort(key=lambda q: -q["n"])
    res = np.empty((total, K), np.float32)
    for k, p in enumerate(pieces):
        c = p["coef"]
        for a0 in range(0, total, 400000):
            sl = slice(a0, a0 + 400000)
            pred = fx[sl, None] * c[0] + fy[sl, None] * c[1] + c[2]
            res[sl, k] = np.abs(fcols[sl] - pred).max(axis=1)
    bias = np.arange(K, dtype=np.float32) * 0.25 - np.array([0.6 if p["flat"] else 0.0 for p in pieces], np.float32)     # a flat colour is the better explanation of a tie
    pick = np.argmin(res + bias[None, :], axis=1)
    best = res[np.arange(total), pick]
    lab = np.where(best <= 1.5 * tol + 1.0, pick, 255).astype(np.uint8)
    L = np.full(m.shape, 255, np.uint8)
    L[ys_all, xs_all] = lab
    if min(m.shape) >= 9:
        Lm = cv2.medianBlur(L, 5)
        L = np.where(m & (L != 255), Lm, 255).astype(np.uint8)
    out = []
    for k, p in enumerate(pieces):
        pm = (L == k)
        if int(pm.sum()) < 60:
            continue
        grad = None
        flat = p["flat"]
        if not flat:
            g = fit_linear(crop, pm, 3.0, 6.0)
            if g is None:
                g = fit_general(crop, pm)
            grad = g
            if grad is None:
                flat = True
        out.append(dict(mask=pm, grad=grad, flat=flat))
    return out


def strip_private(g: Optional[dict]) -> Optional[dict]:
    if g is None:
        return None
    return {k: v for k, v in g.items() if not k.startswith("_")}
