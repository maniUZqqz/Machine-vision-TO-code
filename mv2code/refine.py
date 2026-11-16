"""Stage 6: the feedback loop.

Render the generated page in a headless browser, compare every element with where the pixels
said it should be, correct font sizes and spacings, repeat, and keep the best version (by SSIM
against the screenshot).  Needs the optional ``[browser]`` extra; without it the converter simply
returns the first version.
"""
from __future__ import annotations

import copy
import os
import shutil
import tempfile
from typing import Callable, Dict

import cv2
import numpy as np

from .layout import lay_box

MEASURE_JS = r"""
() => { const o = {};
  document.querySelectorAll('[data-mv]').forEach(e => {
    const r = e.getBoundingClientRect(); let tw = null, th = null;
    if (e.children.length === 0 && e.textContent.trim()) {
      const g = document.createRange(); g.selectNodeContents(e); const q = g.getBoundingClientRect(); tw = q.width; th = q.height; }
    o[e.getAttribute('data-mv')] = [r.left + scrollX, r.top + scrollY, r.width, r.height, tw, th]; });
  return {dom: o, W: document.documentElement.scrollWidth}; }
"""


def ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    H = max(a.shape[0], b.shape[0])
    W = max(a.shape[1], b.shape[1])

    def pad(x):
        o = np.empty((H, W, 3), np.uint8)
        o[:] = x[-1:, -1:, :]
        o[:x.shape[0], :x.shape[1]] = x
        return o
    ga = cv2.cvtColor(pad(a), cv2.COLOR_RGB2GRAY).astype(np.float32)
    gb = cv2.cvtColor(pad(b), cv2.COLOR_RGB2GRAY).astype(np.float32)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = (11, 11)
    ma, mb = cv2.GaussianBlur(ga, k, 1.5), cv2.GaussianBlur(gb, k, 1.5)
    saa = cv2.GaussianBlur(ga * ga, k, 1.5) - ma * ma
    sbb = cv2.GaussianBlur(gb * gb, k, 1.5) - mb * mb
    sab = cv2.GaussianBlur(ga * gb, k, 1.5) - ma * mb
    m = ((2 * ma * mb + c1) * (2 * sab + c2)) / ((ma ** 2 + mb ** 2 + c1) * (saa + sbb + c2))
    return float(m.mean())


class PageRenderer:
    """Writes the page + fonts to a temp dir and renders it with a browser."""

    def __init__(self, browser, assets, width: int):
        self.br, self.width = browser, width
        self.dir = tempfile.mkdtemp(prefix="mv2code-")
        for a in assets:
            dst = os.path.join(self.dir, a["path"])
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(a["source"], dst)

    def close(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def render(self, html: str, js: str = None):
        path = os.path.join(self.dir, "page.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        img, data = self.br.render("file://" + path.replace("\\", "/") if os.name != "nt" else "file:///" + path.replace("\\", "/"),
                                   self.width, probe=False, js=js or MEASURE_JS)
        return img, data


def _plans(root):
    for n in root.walk():
        p = (n.lay or {}).get("plan")
        if p is not None:
            yield p


def _walk_plan(p):
    yield p
    for k in p.kids:
        yield from _walk_plan(k)


def _key(k):
    return k.node.id if (k.kind == "item" and k.node is not None) else f"g{k.gid}"


def _ink_bbox(img: np.ndarray, box, color, pad: int = 6):
    """Ink box (x0, y0, x1, y1) of text of ``color`` inside ``box`` (+pad) of ``img``; None if nothing found."""
    H, W = img.shape[:2]
    x0, y0 = max(0, int(box[0]) - pad), max(0, int(box[1]) - pad)
    x1, y1 = min(W, int(box[0] + box[2]) + pad), min(H, int(box[1] + box[3]) + pad)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    c = img[y0:y1, x0:x1].astype(np.int16)
    flat = c.reshape(-1, 3)
    vals, cnt = np.unique(flat[:: max(1, len(flat) // 8000)], axis=0, return_counts=True)
    bg = vals[cnt.argmax()]
    fg = np.array(color if color is not None else (0, 0, 0), np.float64)
    span = float(np.abs(fg - bg).max())
    if span < 40:
        return None
    m = np.abs(c - fg).max(axis=2) <= 0.45 * span
    ys, xs = np.nonzero(m)
    if len(xs) < 6:
        return None
    return (x0 + xs.min(), y0 + ys.min(), x0 + xs.max() + 1, y0 + ys.max() + 1)


def update_adjust(root, dom: Dict[str, list], adj: Dict, rtl: bool, damp: float = 1.0, rendered=None) -> Dict:
    """One correction step: font scale per text node, margin deltas per flow item, pixel offsets per text."""
    new = copy.deepcopy(adj)
    if rendered is not None:
        for n in root.walk():
            if n.role != "text":
                continue
            tb = n.box
            rb = _ink_bbox(rendered, tb, n.color)
            if rb is None:
                continue
            tcx, tcy = tb[0] + tb[2] / 2.0, tb[1] + tb[3] / 2.0
            rcx, rcy = (rb[0] + rb[2]) / 2.0, (rb[1] + rb[3]) / 2.0
            dx, dy = tcx - rcx, tcy - rcy
            e = new.setdefault(n.id, {})
            if abs(dx) <= 4.5 and abs(dy) <= 4.5:      # only small, safe pixel nudges
                e["ox"] = float(np.clip(e.get("ox", 0.0) + dx * damp, -4.5, 4.5))
                e["oy"] = float(np.clip(e.get("oy", 0.0) + dy * damp, -4.5, 4.5))
    for n in root.walk():
        if n.role != "text" or len(n.lines) != 1:
            continue
        m = dom.get(str(n.id))
        if not m or m[4] is None:
            continue
        cur = m[4] - 1.5
        want = n.box[2]
        if cur > 4 and want > 4:
            r = want / cur
            if abs(r - 1) > 0.012:
                e = new.setdefault(n.id, {})
                e["scale"] = float(np.clip(e.get("scale", 1.0) * (1 + (r - 1) * damp), 0.8, 1.25))
    for plan in _plans(root) if os.environ.get('MV2CODE_REFINE_GAPS', '0') == '1' else ():
        for p in _walk_plan(plan):
            if p.kind not in ("col", "row") or len(p.kids) < 2:
                continue
            for a, b in zip(p.kids, p.kids[1:]):
                da, db = dom.get(str(_key(a))), dom.get(str(_key(b)))
                if not da or not db:
                    continue
                if p.kind == "col":
                    want = b.box[1] - (a.box[1] + a.box[3])
                    have = db[1] - (da[1] + da[3])
                    d = (want - have)
                    if abs(d) >= 1.5:
                        e = new.setdefault(_key(b), {})
                        e["dy"] = e.get("dy", 0.0) + d * damp
                else:
                    if rtl:
                        want = a.box[0] - (b.box[0] + b.box[2])
                        have = da[0] - (db[0] + db[2])
                    else:
                        want = b.box[0] - (a.box[0] + a.box[2])
                        have = db[0] - (da[0] + da[2])
                    d = want - have
                    if abs(d) >= 1.5:
                        e = new.setdefault(_key(b), {})
                        e["dx"] = e.get("dx", 0.0) + d * damp
    return new


def refine(rgb: np.ndarray, root, emit_fn: Callable[[Dict], dict], browser, assets, rtl: bool,
           iterations: int = 3, log=None):
    """Returns (best emit result, report dict)."""
    H, W = rgb.shape[:2]
    pr = PageRenderer(browser, assets, W)
    try:
        adj: Dict = {}
        best = None
        scores = []
        for it in range(iterations + 1):
            out = emit_fn(adj)
            img, data = pr.render(out["html"])
            sc = ssim_gray(rgb, img)
            scores.append(round(sc, 4))
            if best is None or sc > best[0] + 1e-4:
                best = (sc, out, copy.deepcopy(adj))
            elif it > 0 and sc < best[0] - 0.01:
                break
            if it == iterations:
                break
            adj = update_adjust(root, data["dom"], adj, rtl, damp=0.9, rendered=img)
        return best[1], dict(ssim_steps=scores, best_ssim=round(best[0], 4), iterations=len(scores) - 1)
    finally:
        pr.close()
