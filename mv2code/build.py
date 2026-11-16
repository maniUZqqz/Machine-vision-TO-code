"""Stage 3: OCR the text lines, then build the component tree (roles + hierarchy).

Region + text + media -> ``Node`` tree with roles: section, card, nav, button, input, badge,
control (checkbox/radio/toggle), image, icon, text, divider.  Layout (rows/columns/table) is
inferred afterwards by ``layout.py``.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from . import persian
from .imaging import (Box, box_area, box_center, box_contains, box_inter, box_union, cdist, center_in, crop,
                      hex_of, luminance, mode_color)
from .model import Node
from .ocr import Backend, OcrResult, prepare_crop
from .pipeline import Analysis
from .textstyle import TextStyle, fit_text, ink_extent


@dataclass
class TL:
    """A recognised text line."""
    box: Box
    text: str
    conf: float
    rtl: bool = False
    style: Optional[TextStyle] = None


# --------------------------------------------------------------------------- OCR step
def tight_box(img: np.ndarray, box: Box, thr: int = 28) -> Box:
    c = crop(img, box, 0)
    if c.size == 0:
        return box
    ring = np.concatenate([c[0], c[-1], c[:, 0], c[:, -1]])
    bg = np.array(mode_color(ring), np.int16)
    m = np.abs(c.astype(np.int16) - bg).max(axis=2) > thr
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return box
    x0, y0 = max(0, box[0]), max(0, box[1])
    return (int(x0 + xs.min()), int(y0 + ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))


def _alnum_ratio(s: str) -> float:
    n = sum(1 for ch in s if ch.isalnum())
    return n / max(1, len(s.replace(" ", "")))


def read_text(an: Analysis, backend: Backend, lang: str = "auto", ocr_lines=None, min_conf: float = 0.30):
    """Returns (text lines, non-text candidate boxes)."""
    img = an.img
    lines: List[TL] = []
    nontext: List[Box] = []
    if ocr_lines is not None:
        for d in ocr_lines:
            b = tuple(int(round(v)) for v in d["box"])
            b = tight_box(img, b) if d.get("tighten", True) else b
            lines.append(TL(box=b, text=str(d["text"]), conf=float(d.get("conf", 1.0))))
        for c in an.cands:
            if any(box_inter(c.box, ln.box) >= 0.3 * min(box_area(c.box), box_area(ln.box)) for ln in lines):
                continue
            w, h = c.box[2], c.box[3]
            if 6 <= w <= 72 and 6 <= h <= 72 and w / max(1, h) <= 3.2 and h / max(1, w) <= 3.2 and not _corner_artifact(c.box, an.regions):
                nontext.append(c.box)
        return lines, nontext
    cands = [c for c in an.cands if c.box[3] >= 5 and c.box[2] >= 3 and c.box[3] <= 160]
    if not cands:
        return lines, nontext
    crops = [prepare_crop(img, c.box) for c in cands]
    res: List[OcrResult] = backend.read(crops, lang)
    for c, r in zip(cands, res):
        t = (r.text or "").strip()
        w, h = c.box[2], c.box[3]
        compact = max(w, h) <= 72 and w / max(1, h) <= 3.2 and h / max(1, w) <= 3.2
        letters = sum(1 for ch in t if ch.isalnum())
        if w / max(1, h) > 22 or h / max(1, w) > 6:
            continue  # rules and bars, not text
        ok = letters >= 1 and _alnum_ratio(t) >= 0.5
        if letters == 1:
            # a lone character is almost always an icon glyph; keep clear digits only
            ok = ok and t.strip().isdigit() and r.conf >= 0.9 and h >= 10 and not (c.n_comp == 1 and w <= 12)
        else:
            ok = ok and (r.conf >= 0.5 or (letters >= 5 and r.conf >= 0.35))
            if ok and compact and letters <= 2 and r.conf < 0.8 and not any(ch.isdigit() for ch in t):
                ok = False
        if ok:
            lines.append(TL(box=c.box, text=t, conf=r.conf))
        elif compact and w >= 6 and h >= 6 and not _corner_artifact(c.box, an.regions):
            nontext.append(c.box)
    return lines, nontext


def _corner_artifact(b: Box, regions) -> bool:
    """Anti-aliased rounded-corner leftovers of a region are not icons."""
    bx, by, bw, bh = b
    for r in regions:
        rx, ry, rw, rh = r.box
        if r.radius < 2 and r.shape != "ellipse":
            continue
        rad = max(r.radius, 4) + 6
        for cx, cy in ((rx, ry), (rx + rw, ry), (rx, ry + rh), (rx + rw, ry + rh)):
            if abs(bx + bw / 2 - cx) <= rad and abs(by + bh / 2 - cy) <= rad and bw <= rad + 2 and bh <= rad + 2:
                return True
    return False


def postprocess_lines(lines: List[TL], zwnj: bool = True):
    raw = [ln.text for ln in lines]
    pdig = persian.decide_page_digits(raw)
    for ln in lines:
        ln.text = persian.normalize_line(ln.text, pdig, zwnj=zwnj)
        ln.rtl = persian.is_rtl(ln.text)
    texts = [ln.text for ln in lines if ln.text]
    return pdig, persian.page_direction(texts)


# --------------------------------------------------------------------------- helpers
def _color_close(a, b, t=40):
    return a is not None and b is not None and cdist(a, b) <= t


def icon_svg(img: np.ndarray, box: Box, fg, bg):
    """Trace a single-colour glyph into an inline SVG path (None when it is not simple)."""
    c = crop(img, box, 1)
    if c.size == 0 or min(c.shape[:2]) < 6:
        return None
    dist = np.abs(c.astype(np.int16) - np.array(bg, np.int16)).max(axis=2).astype(np.float32)
    peak = float(np.percentile(dist, 99))
    if peak < 20:
        return None
    # single colour check: pixels with strong ink all near the fg colour
    strong = dist >= 0.6 * peak
    if strong.sum() < 8:
        return None
    cols = c[strong].astype(np.int16)
    if np.median(np.abs(cols - np.median(cols, axis=0)).max(axis=1)) > 40:
        return None
    k = 4
    up = cv2.resize(dist / peak, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
    m = (up >= 0.5).astype(np.uint8)
    cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not cnts or len(cnts) > 40:
        return None
    parts = []
    for cn in cnts:
        if cv2.contourArea(cn) < 2.0 * k * k:
            continue
        ap = cv2.approxPolyDP(cn, 0.9 * k, True)
        if len(ap) < 3:
            continue
        pts = ap.reshape(-1, 2) / float(k)
        parts.append("M" + " ".join(f"{x:.1f} {y:.1f}" for x, y in pts) + "Z")
    if not parts:
        return None
    h, w = c.shape[:2]
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">'
            f'<path fill="{hex_of(fg)}" fill-rule="evenodd" d="{" ".join(parts)}"/></svg>'), (w, h)


def png_data_uri(img: np.ndarray, box: Box, quality: int = 88, fmt: str = "auto") -> str:
    c = crop(img, box, 0)
    bgr = cv2.cvtColor(c, cv2.COLOR_RGB2BGR)
    if fmt in ("auto", "jpg") and box[2] * box[3] > 120 * 120 and fmt != "png":
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        mime = "image/jpeg"
    else:
        ok, buf = cv2.imencode(".png", bgr)
        mime = "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.tobytes()).decode("ascii")


# --------------------------------------------------------------------------- tree
class Builder:
    def __init__(self, an: Analysis, lines: List[TL], nontext: List[Box], page_dir: str, opts: dict):
        self.an, self.img = an, an.img
        self.W, self.H = an.size
        self.lines, self.nontext = lines, nontext
        self.page_dir = page_dir
        self.opts = opts
        self._id = 0

    def nid(self):
        self._id += 1
        return self._id

    # ---- text blocks
    def style_lines(self):
        for ln in self.lines:
            ln.rtl = persian.is_rtl(ln.text) if persian.has_arabic_script(ln.text) or any(c.isalpha() for c in ln.text) else (self.page_dir == "rtl")
            ln.style = fit_text(self.img, ln.box, ln.text, ln.rtl, ln.conf)
        self._share_styles()

    def _share_styles(self):
        """Lines of the same ink height and script share one font size / weight: a doubtful reading
        (low OCR confidence, or a size far from its peers) takes the style of its confident peers."""
        groups = {}
        for ln in self.lines:
            if ln.style is None:
                continue
            key = (persian.has_arabic_script(ln.text), int(round(ln.box[3] / 2.0)))
            groups.setdefault(key, []).append(ln)
        for key, members in groups.items():
            good = [m for m in members if m.conf >= 0.8 and len(m.text) >= 3]
            if not good:
                # no trustworthy peer: at least keep the size within a sane range for this ink height
                for m in members:
                    h = m.box[3]
                    lo, hi = 0.62 * h, 1.7 * h
                    m.style.size = float(min(max(m.style.size, lo), hi)) if (m.conf < 0.7) else m.style.size
                continue
            sizes = sorted(m.style.size for m in good)
            med = sizes[len(sizes) // 2]
            wts = [m.style.weight for m in good]
            wmode = max(set(wts), key=wts.count)
            for m in members:
                dev = abs(m.style.size - med) / med
                if m.conf < 0.7 or dev > 0.3 or len(m.text) <= 2:
                    m.style.size = med
                    m.style.weight = wmode

    def blocks(self, lines: List[TL]) -> List[Node]:
        """Group consecutive same-style lines (paragraphs) into text blocks."""
        lines = sorted(lines, key=lambda l: (l.box[1], l.box[0]))
        used = [False] * len(lines)
        nodes: List[Node] = []
        for i, l in enumerate(lines):
            if used[i]:
                continue
            blk = [l]
            used[i] = True
            while True:
                last = blk[-1]
                best = None
                for j in range(len(lines)):
                    if used[j]:
                        continue
                    m = lines[j]
                    if m.box[1] <= last.box[1]:
                        continue
                    if self._joinable(blk, m):
                        if best is None or m.box[1] < lines[best].box[1]:
                            best = j
                if best is None:
                    break
                blk.append(lines[best])
                used[best] = True
            nodes.append(self.text_node(blk))
        return nodes

    def _joinable(self, blk: List[TL], m: TL) -> bool:
        last = blk[-1]
        lb, mb = last.box, m.box
        gap = mb[1] - (lb[1] + lb[3])
        h = max(lb[3], mb[3])
        if gap < -2 or gap > 1.05 * h + 2:
            return False
        s1, s2 = last.style, m.style
        if s1 and s2:
            if abs(s1.size - s2.size) / max(s1.size, s2.size) > 0.14 or abs(s1.weight - s2.weight) > 100 or s1.family != s2.family:
                return False
            if cdist(s1.color, s2.color) > 60:
                return False
        if last.rtl != m.rtl:
            return False
        tol = max(4, 0.03 * max(lb[2], mb[2]))
        left = abs(lb[0] - mb[0]) <= tol
        right = abs((lb[0] + lb[2]) - (mb[0] + mb[2])) <= tol
        cen = abs((lb[0] + lb[2] / 2) - (mb[0] + mb[2] / 2)) <= tol
        if len(blk) >= 2:
            # keep a consistent pitch
            pitch = blk[-1].box[1] - blk[-2].box[1]
            if abs((mb[1] - lb[1]) - pitch) > max(4, 0.25 * pitch):
                return False
        # a short last line of a left/right aligned paragraph keeps its alignment edge
        ok = left or right or cen
        if not ok and m.box[2] < lb[2] * 0.7:
            ok = (left or right)
        return ok

    def text_node(self, blk: List[TL]) -> Node:
        ub = blk[0].box
        for l in blk[1:]:
            ub = box_union(ub, l.box)
        sty = max(blk, key=lambda l: len(l.text)).style or TextStyle()
        size = float(np.median([l.style.size for l in blk if l.style])) if blk[0].style else sty.size
        rtl = blk[0].rtl
        n = Node(role="text", box=ub, text=" ".join(l.text for l in blk), lines=[(l.box, l.text) for l in blk],
                 font=sty.family, weight=sty.weight, size=size, color=sty.color, rtl=rtl,
                 conf=float(np.mean([l.conf for l in blk])), id=self.nid())
        # alignment of a multi-line block / single line relative to its own lines
        if len(blk) > 1:
            lefts = [l.box[0] for l in blk]
            rights = [l.box[0] + l.box[2] for l in blk]
            cens = [l.box[0] + l.box[2] / 2 for l in blk]
            sp = lambda v: max(v) - min(v)
            if sp(lefts) <= sp(rights) and sp(lefts) <= sp(cens):
                n.align = "left"
            elif sp(rights) <= sp(cens):
                n.align = "right"
            else:
                n.align = "center"
            pitch = float(np.median([blk[i + 1].box[1] - blk[i].box[1] for i in range(len(blk) - 1)]))
            n.line_h = round(pitch, 1)
        else:
            n.align = "left"
        sty_f = sty
        # css content box: line-height L, ink top offset inside the line box
        L = n.line_h or round(size * 1.5, 1)
        n.line_h = L
        half = (L - (sty_f.ascent + sty_f.descent) * size) / 2.0
        top = ub[1] - (half + sty_f.ink_top * size)
        # first line of the block anchors the box
        first = blk[0].box
        top = first[1] - (half + sty_f.ink_top * size)
        n.lbox = (ub[0], int(round(top)), ub[2], int(round(L * len(blk))))
        return n

    # ---- regions
    def build(self) -> Node:
        an = self.an
        W, H = self.W, self.H
        root = Node(role="page", box=(0, 0, W, H), fill=an.bg, id=self.nid())
        regions = sorted(an.regions, key=lambda r: -box_area(r.box))
        rnodes = {}
        for r in regions:
            n = Node(role="box", box=r.box, fill=r.fill, grad=r.grad, border=r.border, radius=r.radius, shadow=r.shadow,
                     shape=r.shape, id=self.nid())
            n.lay = dict(inner=r.inner)
            rnodes[r.id] = n
        # attach regions to parents
        for r in regions:
            par = rnodes.get(r.parent) if r.parent is not None else root
            par.children.append(rnodes[r.id])
        # leaf items -> smallest containing region
        def container_for(b: Box, prefer_not: Optional[Node] = None) -> Node:
            best, ba = root, box_area(root.box)
            for r in regions:
                n = rnodes[r.id]
                if n is prefer_not:
                    continue
                if center_in(b, r.inner) and box_area(r.inner) < ba and (box_inter(b, r.inner) >= 0.5 * box_area(b)):
                    best, ba = n, box_area(r.inner)
            return best

        lines_by_cont = {}
        for ln in self.lines:
            c = container_for(ln.box)
            lines_by_cont.setdefault(id(c), (c, []))[1].append(ln)
        for _, (c, ls) in lines_by_cont.items():
            for tn in self.blocks(ls):
                c.children.append(tn)
        for p in self.an.photos:
            c = container_for(p.box)
            n = Node(role="image", box=p.box, radius=p.radius, shape="ellipse" if p.round else "rect", id=self.nid())
            n.src = png_data_uri(self.img, p.box, 80)
            c.children.append(n)
        for b in self.nontext:
            if any(box_inter(b, p.box) > 0.5 * box_area(b) for p in self.an.photos):
                continue
            c = container_for(b)
            c.children.append(self.icon_node(b, c))
        for ln in self.an.lines:
            c = container_for(ln.box)
            n = Node(role="divider", box=ln.box, fill=ln.color, id=self.nid(), kind="h" if ln.horizontal else "v")
            c.children.append(n)
        self.classify(root)
        self.prune(root)
        return root

    def icon_node(self, b: Box, cont: Node) -> Node:
        bg = cont.fill if cont.fill is not None else self.an.bg
        n = Node(role="icon", box=b, id=self.nid())
        obs = None
        try:
            from .textstyle import observed_mask
            obs = observed_mask(self.img, b)
        except Exception:
            pass
        fg = obs[1] if obs else (0, 0, 0)
        n.color = fg
        res = icon_svg(self.img, b, fg, obs[2] if obs else bg)
        if res is not None:
            n.svg = res[0]
        else:
            n.src = png_data_uri(self.img, b, 88, "png")
        return n

    # ---- classification
    def classify(self, node: Node, depth: int = 0):
        for ch in list(node.children):
            if ch.role == "box":
                self.classify(ch, depth + 1)
        self.merge_toggles(node)
        if node.role != "box":
            return
        self.decide(node)

    def merge_toggles(self, node: Node):
        """A track pill whose knob (a small circle) touches it -> one toggle control."""
        kids = node.children
        for k in [c for c in kids if c.role in ("box", "control") and c.shape == "ellipse" and 10 <= c.box[2] <= 36 and abs(c.box[2] - c.box[3]) <= 3]:
            for p in kids:
                if p is k or p.role != "box" or p.shape != "rect" or p.children:
                    continue
                px_, py_, pw_, ph_ = p.box
                kx, ky, kw, kh = k.box
                if not (12 <= ph_ <= 40 and kh <= ph_ + 8 and abs((ky + kh / 2) - (py_ + ph_ / 2)) <= 3):
                    continue
                kcx = kx + kw / 2.0
                if px_ - 14 <= kcx <= px_ + pw_ + 14:
                    margin = max(0, (ph_ - kh) // 2)
                    x0 = min(px_, kx - margin)
                    x1 = max(px_ + pw_, kx + kw + margin)
                    ub = (x0, py_, x1 - x0, ph_)
                    if 1.3 <= ub[2] / max(1, ub[3]) <= 2.8 and ub[2] <= 80:
                        p.box, p.role, p.kind = ub, "control", "toggle"
                        p.radius = ub[3] / 2.0
                        p.lay = dict(p.lay or {}, knob=k.box, knob_fill=k.fill or (255, 255, 255))
                        p.children = []
                        node.children = [c for c in node.children if c is not k]
                        return

    def decide(self, n: Node):
        W, H = self.W, self.H
        x, y, w, h = n.box
        kids = [c for c in n.children if c.role != "divider"]
        texts = [c for c in kids if c.role == "text"]
        icons = [c for c in kids if c.role == "icon"]
        subs = [c for c in kids if c.role in ("box", "card", "button", "input", "badge", "control", "nav", "section", "image")]
        full_w = w >= 0.92 * W
        # ---- tiny shapes: controls
        if not texts and w <= 64 and h <= 64:
            subs_shapes = [c for c in kids if c.role in ("box",)]
            if n.shape == "ellipse" and 12 <= w <= 30 and abs(w - h) <= 3 and len(kids) <= 1:
                n.role, n.kind = "control", "radio"
                n.children = []
                return
            if n.shape == "rect" and 12 <= w <= 30 and abs(w - h) <= 3 and n.radius <= 0.35 * w and len(subs_shapes) + len(icons) <= 1:
                n.role, n.kind = "control", "checkbox"
                n.children = []
                return
            pill = n.radius >= 0.4 * min(w, h) - 1
            if pill and 28 <= w <= 70 and 14 <= h <= 36 and 1.4 <= w / max(1, h) <= 2.6:
                n.role, n.kind = "control", "toggle"
                n.children = []
                return
        # ---- bars
        if full_w and y <= 4 and h <= 130 and (len(texts) + len(icons) + len(subs)) >= 2:
            n.role, n.kind = "nav", "header"
            return
        if full_w and y + h >= H - 4 and y > 0.45 * H and (len(texts) >= 2):
            n.role, n.kind = "nav", "footer"
            return
        if h >= 0.55 * H and w <= 0.34 * W and (x <= 4 or x + w >= W - 4) and len(texts) + len(icons) >= 3:
            n.role, n.kind = "nav", "sidebar"
            return
        if full_w and h <= 140 and y + h >= H - 4 and len(texts) + len(icons) >= 3 and w <= 600:
            n.role, n.kind = "nav", "bottom"
            return
        if full_w:
            n.role = "section"
            return
        # ---- single-text controls
        one_text = len(texts) == 1 and len(texts[0].lines) == 1 and len([c for c in subs if c.role != "image"]) == 0
        if one_text and len(icons) <= 1 and h <= 200 and w >= 20:
            t = texts[0]
            tb = t.box
            has_border = bool(n.border) and len(n.border) >= 3
            cx_off = abs((tb[0] + tb[2] / 2) - (x + w / 2)) / max(1.0, w)
            # chevron icon at the far end -> select
            start_off = (x + w - (tb[0] + tb[2])) if t.rtl else (tb[0] - x)
            end_off = (tb[0] - x) if t.rtl else (x + w - (tb[0] + tb[2]))
            if icons:
                ic = icons[0]
                content = box_union(tb, ic.box)
            else:
                content = tb
            ccx_off = abs((content[0] + content[2] / 2) - (x + w / 2)) / max(1.0, w)
            pill = n.radius >= 0.4 * min(w, h) - 1
            if h <= 34 and pill and w <= 260 and (t.size <= 17) and not has_border:
                n.role = "badge"
                self._as_leaf_text(n, texts[0])
                return
            centered = ccx_off <= 0.12
            txt_cx_off = abs((tb[0] + tb[2] / 2) - (x + w / 2)) / max(1.0, w)
            far_icon = False
            if icons:
                ib = icons[0].box
                at_end = ((ib[0] - x) <= 28) if t.rtl else ((x + w) - (ib[0] + ib[2]) <= 28)
                far_icon = at_end and start_off <= 28
            if 28 <= h <= 72 and has_border and w >= 90 and ((not centered and start_off <= 0.3 * w + 6) or far_icon):
                n.role, n.kind = "input", ("select" if icons else "text")
                self._as_leaf_text(n, texts[0], placeholder=True)
                return
            if h > 72 and has_border and w >= 200 and not icons and (tb[1] - y) <= 28 and start_off <= 24:
                n.role, n.kind = "input", "textarea"
                self._as_leaf_text(n, texts[0], placeholder=True)
                return
            if centered and (n.fill is not None or has_border) and 22 <= h <= 90 and w <= 0.7 * W:
                n.role = "button"
                self._as_leaf_text(n, texts[0])
                return
        # empty bordered box: input
        if not kids and n.border and len(n.border) >= 3 and 28 <= h <= 72 and w >= 90:
            n.role, n.kind = "input", "text"
            return
        # ---- containers
        if len(kids) == 0:
            if h <= 6 or w <= 6:
                n.role = "divider"
            else:
                n.role = "box"
            return
        if (w >= 80 and h >= 50) or n.border or n.shadow:
            n.role = "card"
        else:
            n.role = "box"

    def _as_leaf_text(self, n: Node, t: Node, placeholder: bool = False):
        """Control that owns exactly one text child (the label/placeholder): keep it as a child."""
        n.text = t.text

    def prune(self, node: Node):
        keep = []
        for c in node.children:
            self.prune(c)
            if c.role == "box" and not c.children and c.fill is not None and node.fill is not None and cdist(c.fill, node.fill) <= 6 and not c.border:
                continue
            keep.append(c)
        node.children = keep


def build_tree(an: Analysis, lines: List[TL], nontext: List[Box], page_dir: str, opts=None) -> Node:
    b = Builder(an, lines, nontext, page_dir, opts or {})
    b.style_lines()
    root = b.build()
    root.kind = page_dir
    # number nodes in document order
    for i, n in enumerate(root.walk()):
        n.id = i
    return root
