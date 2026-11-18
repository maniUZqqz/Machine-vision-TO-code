"""Stage 5: HTML / CSS generation.

* ``responsive`` - semantic tags, nested flexbox / table, no absolute positioning (except for
  genuinely overlapping stacks), colours as CSS variables, sections full-bleed.
* ``exact``      - every node absolutely positioned (pixel-faithful at the source size).
"""
from __future__ import annotations

import html as _html
import os
import re
from collections import Counter
from typing import Dict, List, Optional

from .imaging import Box, box_area, hex_of
from .layout import Plan, lay_box
from .model import Node

FONT_FACES = [
    ("Vazirmatn", 400, "vazirmatn-arabic-400-normal.woff2", "U+0600-06FF,U+0750-077F,U+FB50-FDFF,U+FE70-FEFC,U+200C-200E,U+066A-066C"),
    ("Vazirmatn", 500, "vazirmatn-arabic-500-normal.woff2", "U+0600-06FF,U+0750-077F,U+FB50-FDFF,U+FE70-FEFC,U+200C-200E,U+066A-066C"),
    ("Vazirmatn", 700, "vazirmatn-arabic-700-normal.woff2", "U+0600-06FF,U+0750-077F,U+FB50-FDFF,U+FE70-FEFC,U+200C-200E,U+066A-066C"),
    ("Vazirmatn", 400, "vazirmatn-latin-400-normal.woff2", "U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD"),
    ("Vazirmatn", 500, "vazirmatn-latin-500-normal.woff2", "U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD"),
    ("Vazirmatn", 700, "vazirmatn-latin-700-normal.woff2", "U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD"),
    ("Inter", 400, "inter-latin-400-normal.woff2", None),
    ("Inter", 500, "inter-latin-500-normal.woff2", None),
    ("Inter", 600, "inter-latin-600-normal.woff2", None),
    ("Inter", 700, "inter-latin-700-normal.woff2", None),
]
FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")


def font_assets():
    """Files the generated CSS refers to (``fonts/<name>``) so callers can write a bundle."""
    out = []
    for fam, w, fn, ur in FONT_FACES:
        p = os.path.join(FONT_DIR, fn)
        if os.path.exists(p):
            out.append(dict(path="fonts/" + fn, source=p))
    return out


def esc(s: str) -> str:
    return _html.escape(s, quote=True)


def px(v) -> str:
    v = round(float(v), 1)
    return f"{int(v)}px" if v == int(v) else f"{v}px"


def css_color(c) -> str:
    return hex_of(c)


def css_gradient(g: dict) -> str:
    """CSS value of a fitted gradient (``segment``/``gradfit`` dict): linear (2..5 stops) or radial (circle, farthest-corner)."""
    stops = g.get("stops")
    if g.get("kind") == "radial":
        head = "radial-gradient(circle farthest-corner at %d%% %d%%" % (round(100 * g.get("cx", 0.5)), round(100 * g.get("cy", 0.5)))
    else:
        head = "linear-gradient(%ddeg" % int(g["angle"])
    if stops and len(stops) > 2:
        body = ",".join("%s %s%%" % (css_color(c), ("%.1f" % (100 * t)).rstrip("0").rstrip(".")) for t, c in stops)
    else:
        body = "%s,%s" % (css_color(g["c1"]), css_color(g["c2"]))
    return head + "," + body + ")"


class Sheet:
    """Collects declaration sets and hands out short class names."""

    def __init__(self):
        self.rules: Dict[tuple, str] = {}
        self.order: List[tuple] = []

    def cls(self, decls: Dict[str, str]) -> str:
        key = tuple((k, v) for k, v in decls.items() if v not in (None, ""))
        if not key:
            return ""
        if key not in self.rules:
            self.rules[key] = f"s{len(self.rules) + 1}"
            self.order.append(key)
        return self.rules[key]

    def css(self) -> str:
        out = []
        for key in self.order:
            name = self.rules[key]
            body = ";".join(f"{k}:{v}" for k, v in key)
            out.append(f".{name}{{{body}}}")
        return "\n".join(out)


class Emitter:
    def __init__(self, root: Node, mode: str, rtl_page: bool, size, opts=None, adjust=None):
        self.adj = adjust or {}
        self.root, self.mode = root, mode
        self.rtl_page = rtl_page
        self.W, self.H = size
        self.opts = opts or {}
        self.sheet = Sheet()
        self.extra_css: List[str] = []
        self.body_size = self._body_size()
        self.heads = self._heading_sizes()
        self.n_el = 0

    # ---- typography helpers
    def _text_nodes(self):
        return [n for n in self.root.walk() if n.role == "text"]

    def _body_size(self):
        c = Counter()
        for n in self._text_nodes():
            c[round(n.size)] += len(n.text)
        return c.most_common(1)[0][0] if c else 16

    def _heading_sizes(self):
        sizes = sorted({round(n.size) for n in self._text_nodes() if n.size >= 1.3 * self.body_size and len(n.lines) <= 3}, reverse=True)
        return {s: i + 1 for i, s in enumerate(sizes[:3])}

    def heading_tag(self, n: Node) -> Optional[str]:
        if n.role != "text":
            return None
        lvl = self.heads.get(round(n.size))
        return f"h{lvl}" if lvl else None

    # ---- style props
    def box_props(self, n: Node, abs_pos: bool = False) -> Dict[str, str]:
        d: Dict[str, str] = {}
        if n.grad:
            d["background"] = css_gradient(n.grad)
        elif n.fill is not None:
            d["background"] = css_color(n.fill)
        if n.border:
            sides = n.border
            vals = {s: sides[s] for s in sides}
            if len(vals) == 4 and len({(v[0], v[1]) for v in vals.values()}) == 1:
                w, c = next(iter(vals.values()))
                d["border"] = f"{px(w)} solid {css_color(c)}"
            else:
                for s, (w, c) in vals.items():
                    d[f"border-{s}"] = f"{px(w)} solid {css_color(c)}"
        if n.shape == "ellipse":
            d["border-radius"] = "50%"
        elif n.radius and n.radius >= 1:
            d["border-radius"] = px(n.radius)
        if n.shadow:
            s = n.shadow
            d["box-shadow"] = f"0 {px(s['oy'])} {px(s['blur'])} rgba(0,0,0,{s['alpha']})"
        return d

    def text_props(self, n: Node) -> Dict[str, str]:
        fam = "Vazirmatn" if n.font == "Vazirmatn" else "Inter"
        d = {
            "font-family": f'{fam},sans-serif' if fam == "Vazirmatn" else "Inter,Vazirmatn,sans-serif",
            "font-size": px(n.size * self.adj.get(n.id, {}).get("scale", 1.0)),
            "line-height": px(n.line_h),
        }
        if n.weight != 400:
            d["font-weight"] = str(n.weight)
        d["color"] = css_color(n.color or (0, 0, 0))
        return d

    # ------------------------------------------------------------------ document
    def document(self, title: str = "Converted page") -> dict:
        lang = "fa" if self.rtl_page else "en"
        direction = "rtl" if self.rtl_page else "ltr"
        body = self.emit_root()
        css = self.finish_css()
        fonts = "".join(
            "@font-face{font-family:%s;font-weight:%d;font-style:normal;font-display:swap;src:url(fonts/%s) format('woff2')%s}\n"
            % (fam, w, fn, (";unicode-range:" + ur) if ur else "")
            for fam, w, fn, ur in FONT_FACES
        )
        html = (f'<!doctype html>\n<html lang="{lang}" dir="{direction}">\n<head>\n<meta charset="utf-8">\n'
                f'<meta name="viewport" content="width=device-width, initial-scale=1">\n<title>{esc(title)}</title>\n'
                f'<style>\n{fonts}{css}\n</style>\n</head>\n<body>\n{body}\n</body>\n</html>\n')
        return dict(html=html, css=fonts + css, body=body)

    def finish_css(self) -> str:
        base = ("*,*::before,*::after{box-sizing:border-box}\n"
                "html,body{margin:0;padding:0}\n"
                "img,svg{display:block;max-width:100%}\n"
                "div,section,header,footer,aside,nav,table,button,input,p,h1,h2,h3{max-width:100%}\n"
                "p,h1,h2,h3,h4,h5,h6{margin:0;font-weight:inherit}\n"
                "button,input{font:inherit;margin:0}\n"
                "hr{border:0;margin:0}\n"
                "body{overflow-wrap:anywhere}\n")
        bp = int(max(640, min(self.W - 1, 0.93 * self.W))) if self.mode == "responsive" else 0
        mq = ("\n@media (max-width:%dpx){.r{flex-wrap:wrap}.button,.badge,.nw{white-space:normal}}" % bp) if bp else ""
        css = self.sheet.css() + ("\n" + "\n".join(self.extra_css) if self.extra_css else "") + mq
        css = self.apply_vars(base, css)
        return css

    def apply_vars(self, base: str, css: str) -> str:
        """Colours used more than once become CSS variables (--bg, --text, --surface, ...)."""
        cnt = Counter(re.findall(r"#[0-9a-f]{6}", css))
        names: Dict[str, str] = {}
        bgc = self.root.fill
        roles = {}
        if bgc is not None:
            roles[css_color(bgc)] = "--bg"
        # most common text colour / most common fill of containers / most common button fill
        tcol = Counter(css_color(n.color) for n in self._text_nodes() if n.color)
        if tcol:
            roles.setdefault(tcol.most_common(1)[0][0], "--text")
        fills = Counter(css_color(n.fill) for n in self.root.walk() if n.role in ("card",) and n.fill is not None)
        if fills:
            roles.setdefault(fills.most_common(1)[0][0], "--surface")
        acc = Counter(css_color(n.fill) for n in self.root.walk() if n.role == "button" and n.fill is not None)
        if acc:
            roles.setdefault(acc.most_common(1)[0][0], "--accent")
        k = 1
        for col, c in cnt.most_common():
            if c < 2:
                break
            names[col] = roles.get(col) or f"--c{k}"
            if col not in roles:
                k += 1
        for col, nm in roles.items():
            if col in cnt and col not in names:
                names[col] = nm
        used = {}
        for col, nm in names.items():
            if nm in used.values():
                nm = f"{nm}-{len(used)}"
            used[col] = nm
        for col, nm in used.items():
            css = css.replace(col, f"var({nm})")
        root_vars = ":root{" + ";".join(f"{nm}:{col}" for col, nm in used.items()) + "}\n" if used else ""
        return root_vars + base + css

    # ------------------------------------------------------------------ generic element pieces
    def tag(self, name: str, cls: str, attrs: str = "", inner: str = "", void: bool = False, extra_cls: str = "", nid=None) -> str:
        self.n_el += 1
        if nid is not None and self.opts.get("ids", True):
            attrs += f' data-mv="{nid}"'
        c = (extra_cls + " " + cls).strip()
        a = f' class="{c}"' if c else ""
        if void:
            return f"<{name}{a}{attrs}>"
        return f"<{name}{a}{attrs}>{inner}</{name}>"

    # ------------------------------------------------------------------ leaves
    def text_html(self, n: Node, slot: Dict[str, str], exact: bool = False) -> str:
        d = dict(self.text_props(n))
        if n.rtl != self.rtl_page:
            pass
        attrs = f' dir="{"rtl" if n.rtl else "ltr"}"' if n.rtl != self.rtl_page else ""
        multi = len(n.lines) > 1
        if exact:
            d.update(slot)
            d["white-space"] = "nowrap"
            d["text-align"] = ("center" if n.align == "center" else ("right" if (n.align == "right") else "left")) if multi else ("right" if n.rtl else "left")
            inner = "<br>".join(esc(t) for _, t in n.lines)
        else:
            d.update(slot)
            if multi:
                d["text-align"] = {"left": "left", "right": "right", "center": "center"}[n.align] if n.align in ("left", "right", "center") else "start"
                d.setdefault("max-width", "100%")
                inner = esc(n.text)
            else:
                inner = esc(n.text)
                d.setdefault("max-width", "100%")
            if n.rtl != self.rtl_page and not multi:
                d["text-align"] = "right" if n.rtl else "left"
        a = self.adj.get(n.id, {})
        if (a.get("ox") or a.get("oy")) and not exact:
            d["position"] = "relative"           # sub-4px visual correction found by the feedback loop
            d["left"] = px(a.get("ox", 0))
            d["top"] = px(a.get("oy", 0))
        tg = self.heading_tag(n) or ("p" if multi else "div")
        if tg == "div" and self.opts.get("inline_text"):
            tg = "span"
        if not multi and not exact:
            d["white-space"] = "nowrap"   # one line in the screenshot: keep it one line (wraps below the breakpoint)
        cls = self.sheet.cls(d)
        return self.tag(tg, cls, attrs, inner, nid=n.id, extra_cls="" if (multi or exact) else "nw")

    def icon_html(self, n: Node, slot: Dict[str, str]) -> str:
        w, h = n.box[2], n.box[3]
        d = dict(slot)
        d.update({"width": px(w), "height": px(h), "flex": "none"})
        cls = self.sheet.cls(d)
        if n.svg:
            return self.tag("span", cls, ' aria-hidden="true"', n.svg, extra_cls="ico", nid=n.id)
        return self.tag("img", cls, f' src="{n.src}" alt="" aria-hidden="true"', void=True, extra_cls="ico", nid=n.id)

    def image_html(self, n: Node, slot: Dict[str, str]) -> str:
        d = dict(slot)
        d.update({"width": px(n.box[2]), "height": px(n.box[3]), "object-fit": "cover"})
        if n.shape == "ellipse":
            d["border-radius"] = "50%"
        elif n.radius:
            d["border-radius"] = px(n.radius)
        d.setdefault("max-width", "100%")
        cls = self.sheet.cls(d)
        return self.tag("img", cls, f' src="{n.src}" alt="" width="{int(n.box[2])}" height="{int(n.box[3])}"', void=True, extra_cls="img", nid=n.id)

    def divider_html(self, n: Node, slot: Dict[str, str]) -> str:
        d = dict(slot)
        horiz = n.kind != "v"
        d["background"] = css_color(n.fill)
        free = any(k in slot for k in ("align-self", "margin-inline-start", "flex"))
        d.update({"height": px(n.box[3]), "width": px(n.box[2]) if free else "100%", "max-width": "100%"} if horiz
                 else {"width": px(n.box[2]), "align-self": "stretch"})
        d["flex"] = "none"
        return self.tag("hr", self.sheet.cls(d), void=True, extra_cls="rule", nid=n.id)

    # ------------------------------------------------------------------ exact mode
    def emit_root(self) -> str:
        if self.mode == "exact":
            return self.emit_root_exact()
        return self.emit_root_responsive()

    def emit_root_exact(self) -> str:
        root = self.root
        body_d = {"background": css_color(root.fill or (255, 255, 255)), "margin": "0"}
        self.extra_css.append(f"body{{background:{css_color(root.fill or (255,255,255))};margin:0}}")
        page_d = {"position": "relative", "width": px(self.W), "height": px(self.H), "overflow": "hidden"}
        inner = "".join(self.abs_node(c, (0, 0)) for c in root.children)
        return self.tag("div", self.sheet.cls(page_d), "", "\n" + inner + "\n", extra_cls="page")

    def abs_node(self, n: Node, origin) -> str:
        ox, oy = origin
        x, y, w, h = n.box
        slot = {"position": "absolute"}
        if n.role == "text":
            lb = n.lbox or n.box
            a = self.adj.get(n.id, {})
            slot.update({"left": px(lb[0] - ox + a.get("ox", 0)), "top": px(lb[1] - oy + a.get("oy", 0)), "width": px(lb[2] + 2), "height": px(lb[3])})
            if n.rtl and len(n.lines) == 1:
                slot.update({"left": px(lb[0] - ox - 1 + a.get("ox", 0)), "width": px(lb[2] + 2)})
            return self.text_html(n, slot, exact=True)
        slot.update({"left": px(x - ox), "top": px(y - oy), "width": px(w), "height": px(h)})
        if n.role == "icon":
            slot.pop("width"), slot.pop("height")
            return self.icon_html(n, slot)
        if n.role == "image":
            slot.pop("width"), slot.pop("height")
            return self.image_html(n, slot)
        if n.role == "divider":
            d = dict(slot)
            d["background"] = css_color(n.fill)
            return self.tag("hr", self.sheet.cls(d), void=True, extra_cls="rule", nid=n.id)
        d = dict(slot)
        d.update(self.box_props(n))
        bl = (n.border or {}).get("left", (0, 0))[0]
        bt = (n.border or {}).get("top", (0, 0))[0]
        kids = "".join(self.abs_node(c, (x + bl, y + bt)) for c in n.children if not (n.role in ("control",)))
        tg, extra = self.semantic(n)
        if n.role == "input":
            d["font"] = "inherit"
        return self.tag(tg, self.sheet.cls(d), extra, kids, extra_cls=n.role if n.role != "box" else "", nid=n.id)

    def semantic(self, n: Node):
        if n.role == "button":
            return "button", ' type="button"'
        if n.role == "nav":
            return {"header": "header", "footer": "footer", "sidebar": "aside", "bottom": "nav"}.get(n.kind, "nav"), ""
        if n.role == "section":
            return "section", ""
        if n.role == "badge":
            return "span", ""
        return "div", ""

    # ------------------------------------------------------------------ responsive mode
    def emit_root_responsive(self) -> str:
        root = self.root
        self.extra_css.append(f"body{{background:{css_color(root.fill or (255, 255, 255))};margin:0}}")
        plan = root.lay["plan"] if root.lay else None
        if plan is None:
            return ""
        # top-level full-width bands are full-bleed; everything else lives in a centred page column
        kids = plan.kids if plan.kind == "col" else [plan]
        G = (0, 0, self.W, self.H)
        out = []
        prev_bottom = 0
        for k in kids:
            n = k.node if k.kind == "item" else None
            gap = max(-4000, int(round(k.box[1] - prev_bottom)))
            if n is not None and n.role in ("section", "nav") and n.box[2] >= 0.98 * self.W:
                out.append(self.bleed(k, n, gap))
            else:
                slot: Dict[str, str] = {}
                self.cross_align(slot, k.box, G, self.rtl_page, k)
                inner = self.plan_html(k, slot, self.W, "col")
                wrap = {"max-width": px(self.W), "margin": f"{px(gap)} auto 0", "width": "100%", "display": "flex",
                        "flex-direction": "column"}
                out.append(self.tag("div", self.sheet.cls(wrap), "", "\n" + inner + "\n", extra_cls="wrap"))
            prev_bottom = k.box[1] + k.box[3]
        return "\n".join(out)

    def bleed(self, k: Plan, n: Node, gap: int = 0) -> str:
        """Full-width band: outer element carries the background, inner centres the content."""
        props = self.box_props(n)
        props.pop("border-radius", None)
        if gap:
            props["margin-top"] = px(gap)
        outer = self.sheet.cls(props)
        tg, extra = self.semantic(n)
        inner_n = Node(**{**n.__dict__, "fill": None, "grad": None, "border": None, "shadow": None, "radius": 0.0})
        inner_n.lay = n.lay
        inner = self.container_html(inner_n, {"max-width": px(self.W), "margin": "0 auto", "width": "100%"},
                                    inner_box=(n.lay or {}).get("inner", n.box), avail=self.W, bleed=True)
        return self.tag(tg, outer, extra, "\n" + inner + "\n", extra_cls=n.role, nid=n.id)

    def plan_html(self, p: Plan, slot: Dict[str, str], avail: float, parent_kind: str) -> str:
        if p.kind == "item":
            return self.node_html(p.node, slot, avail, parent_kind)
        if p.kind == "table":
            return self.table_html(p, slot)
        if p.kind == "stack":
            return self.stack_html(p, slot)
        return self.group_html(p, slot, avail)

    # --- group (col / row) --------------------------------------------------------
    def group_decls(self, p: Plan, avail: float) -> Dict[str, str]:
        d = {"display": "flex"}
        if p.kind == "col":
            d["flex-direction"] = "column"
        else:
            d["flex-wrap"] = "nowrap"
            gb = p.box
            # vertical alignment of the cells
            kids = [k.box for k in p.kids]
            cen = [b[1] + b[3] / 2 for b in kids]
            top = [b[1] for b in kids]
            bot = [b[1] + b[3] for b in kids]
            tall = max(b[3] for b in kids) > 90
            if max(cen) - min(cen) <= 2.5 and not tall:
                d["align-items"] = "center"
            elif max(top) - min(top) <= 2:
                d["align-items"] = "flex-start"
            elif max(bot) - min(bot) <= 2:
                d["align-items"] = "flex-end"
            else:
                d["align-items"] = "flex-start"
        return d

    def group_html(self, p: Plan, slot: Dict[str, str], avail: float) -> str:
        d = self.group_decls(p, avail)
        d.update(slot)
        kids_html = self.kids_html(p, avail, d)
        return self.tag("div", self.sheet.cls(d), "", "\n" + kids_html + "\n", extra_cls=("g r" if p.kind == "row" else "g"), nid=f"g{p.gid}")

    @staticmethod
    def key(k: Plan):
        return k.node.id if (k.kind == "item" and k.node is not None) else f"g{k.gid}"

    def kids_html(self, p: Plan, avail: float, gd: Dict[str, str]) -> str:
        rtl = self.rtl_page
        out = []
        G = p.box
        fills = abs(G[2] - avail) <= 3
        prev = None
        n = len(p.kids)
        if p.kind == "col":
            for i, k in enumerate(p.kids):
                b = k.box
                slot: Dict[str, str] = {}
                gap = (b[1] - (prev[1] + prev[3])) if prev is not None else 0
                gap += self.adj.get(self.key(k), {}).get("dy", 0)
                if gap:
                    slot["margin-top"] = px(gap)
                self.cross_align(slot, b, G, rtl, k)
                out.append(self.plan_html(k, slot, G[2], "col"))
                prev = b
        else:
            # row: DOM order = reading order (already reversed for rtl by the cutter)
            gaps = []
            for i, k in enumerate(p.kids):
                if i:
                    pb = p.kids[i - 1].box
                    b = k.box
                    g = (pb[0] - (b[0] + b[2])) if rtl else (b[0] - (pb[0] + pb[2]))
                    gaps.append(g)
            big = None
            if len(gaps) >= 1 and fills:
                mx = max(range(len(gaps)), key=lambda i: gaps[i])
                others = [g for j, g in enumerate(gaps) if j != mx]
                med = sorted(others)[len(others) // 2] if others else 0
                if gaps[mx] >= max(40, 3 * med + 20):
                    big = mx
            for i, k in enumerate(p.kids):
                b = k.box
                slot = {}
                if i:
                    if big is not None and i - 1 == big:
                        slot["margin-inline-start"] = "auto"
                    elif gaps[i - 1] or self.adj.get(self.key(k), {}).get("dx"):
                        slot["margin-inline-start"] = fluid(max(0, gaps[i - 1] + self.adj.get(self.key(k), {}).get("dx", 0)), G[2])
                ai = gd.get("align-items")
                if ai == "flex-start" and abs(b[1] - G[1]) > 2:
                    slot["margin-top"] = px(b[1] - G[1])
                elif ai == "flex-end":
                    pass
                self.row_item(slot, k, fills, G, p)
                out.append(self.plan_html(k, slot, b[2], "row"))
        return "\n".join(out)

    def cross_align(self, slot, b, G, rtl, k: Plan):
        """Horizontal placement of a child inside a column group."""
        if abs(b[2] - G[2]) <= 2:
            if k.kind == "item" and k.node is not None and k.node.role == "text":
                return
            return
        start_off = (G[0] + G[2]) - (b[0] + b[2]) if rtl else b[0] - G[0]
        end_off = (b[0] - G[0]) if rtl else (G[0] + G[2]) - (b[0] + b[2])
        small = k.kind == "item" and k.node is not None and k.node.role in ("text", "icon", "image", "control", "badge", "button", "divider")
        if b[2] >= 0.9 * G[2] and not small:
            if start_off > 2:
                slot["margin-inline-start"] = fluid(start_off, G[2])
            return
        cen = abs((b[0] + b[2] / 2) - (G[0] + G[2] / 2))
        if start_off <= 2:
            slot["align-self"] = "flex-start"
        elif end_off <= 2:
            slot["align-self"] = "flex-end"
        elif cen <= 2:
            slot["align-self"] = "center"
        else:
            slot["align-self"] = "flex-start"
            slot["margin-inline-start"] = fluid(start_off, G[2])

    def _has_wide(self, k: Plan) -> bool:
        """A group is allowed to grow when it contains a container-like item."""
        for q in k.kids:
            if q.kind == "item" and q.node is not None and q.node.role in ("card", "box", "section", "input", "nav"):
                return True
            if q.kind != "item" and self._has_wide(q):
                return True
        return False

    def row_item(self, slot, k: Plan, fills: bool, G, p: Plan):
        w = k.box[2]
        n = k.node if k.kind == "item" else None
        if n is not None and n.role in ("icon", "image", "control", "badge", "button", "divider"):
            slot["flex"] = "none"
        elif n is not None and n.role == "text":
            if len(n.lines) > 1:
                slot["flex"] = f"0 1 {px(w * 1.03 + 3)}"
            else:
                slot["flex"] = "0 0 auto"
                slot["max-width"] = "100%"
            slot["min-width"] = "0"
        elif fills and w >= 120 and ((n is not None and n.role in ("card", "box", "section", "input", "nav")) or (n is None and self._has_wide(k))):
            slot["flex"] = f"1 1 {px(w)}"
            slot["min-width"] = "0"
        elif n is None and not self._has_wide(k):
            slot["flex"] = "0 0 auto"      # content-sized group (text + small things)
            slot["max-width"] = "100%"
            slot["min-width"] = "0"
        else:
            slot["flex"] = f"0 1 {px(w)}"
            slot["min-width"] = "0"

    # --- stacks (overlaps): absolute inside a relative box -----------------------------
    def stack_html(self, p: Plan, slot: Dict[str, str]) -> str:
        G = p.box
        d = {"position": "relative", "width": px(G[2]), "height": px(G[3]), "max-width": "100%", "flex": "none"}
        d.update(slot)
        kids = []
        for k in p.kids:
            n = k.node
            kslot = {"position": "absolute", "left": px(k.box[0] - G[0]), "top": px(k.box[1] - G[1])}
            if n.role == "text":
                kslot["white-space"] = "nowrap"
            elif n.role not in ("icon", "image"):
                kslot["width"] = px(k.box[2])
                kslot["height"] = px(k.box[3])
            kids.append(self.node_html(n, kslot, G[2], "stack", force_abs=True))
        return self.tag("div", self.sheet.cls(d), "", "\n" + "\n".join(kids) + "\n", extra_cls="stack")

    # --- tables ------------------------------------------------------------------------
    def table_html(self, p: Plan, slot: Dict[str, str]) -> str:
        cells = p.meta["cells"]
        cy = p.meta["centers"]
        rtl = self.rtl_page
        G = p.box
        ncol = len(cells[0])
        # column x-extents
        col_edges = []
        for j in range(ncol):
            xs0 = min(c[j].box[0] for c in cells)
            xs1 = max(c[j].box[0] + c[j].box[2] for c in cells)
            col_edges.append((xs0, xs1))
        pitch = float(sum(cy[i + 1] - cy[i] for i in range(len(cy) - 1)) / max(1, len(cy) - 1))
        # horizontal rules between rows (divider nodes inside the parent) are recovered by the caller via meta
        rules = p.meta.get("rules")
        # cell padding so that columns sit where they were seen
        d = dict(slot)
        d.update({"border-collapse": "collapse", "width": "100%", "table-layout": "fixed"})
        trs = []
        thead = p.meta.get("thead")
        total_w = max(1, G[2])
        cols = "".join(self.tag("col", self.sheet.cls({"width": f"{round(100 * (e1 - e0) / total_w, 2)}%"}), void=True) for e0, e1 in col_edges)
        for i, row in enumerate(cells):
            tds = []
            for j, c in enumerate(row):
                nodes = [c.node] if c.kind == "item" else [q.node for q in c.kids if q.kind == "item"]
                inner = "".join(self.node_html(nn, {}, c.box[2], "cell") for nn in nodes)
                align = "right" if rtl else "left"
                e0, e1 = col_edges[j]
                lo, hi = c.box[0], c.box[0] + c.box[2]
                if abs((lo + hi) / 2 - (e0 + e1) / 2) <= 3 and (e1 - e0) - c.box[2] > 6:
                    align = "center"
                elif (abs(hi - e1) <= abs(lo - e0)):
                    align = "right"
                else:
                    align = "left"
                pad_start = (lo - e0) if align == "left" else (e1 - hi)
                dd = {"text-align": align, "vertical-align": "middle", "height": px(pitch), "padding": f"0 {px(max(0, pad_start))}"}
                if rules and i < len(rules) and rules[i] is not None:
                    w, col = rules[i]
                    dd["border-bottom"] = f"{px(w)} solid {css_color(col)}"
                tds.append(self.tag("td", self.sheet.cls(dd), "", inner))
            trs.append(self.tag("tr", "", "", "".join(tds)))
        head_html = ""
        if thead is not None:
            ths = []
            order = sorted(thead.children, key=lambda c: c.box[0], reverse=rtl)
            for j, c in enumerate(order[:ncol]):
                e0, e1 = col_edges[j] if j < len(col_edges) else (c.box[0], c.box[0] + c.box[2])
                lo, hi = c.box[0], c.box[0] + c.box[2]
                align = "right" if abs(hi - e1) <= abs(lo - e0) else "left"
                pad_start = (e1 - hi) if align == "right" else (lo - e0)
                dd = {"text-align": align, "vertical-align": "middle", "height": px(thead.box[3]), "padding": f"0 {px(max(0, pad_start))}",
                      "font-weight": "inherit"}
                ths.append(self.tag("th", self.sheet.cls(dd), "", self.node_html(c, {}, c.box[2], "cell")))
            trd = self.box_props(thead)
            trd.pop("border-radius", None)
            trd.pop("box-shadow", None)
            head_html = "<thead>" + self.tag("tr", self.sheet.cls(trd), "", "".join(ths)) + "</thead>"
        body = "<colgroup>" + cols + "</colgroup>" + head_html + "<tbody>" + "".join(trs) + "</tbody>"
        return self.tag("table", self.sheet.cls(d), "", body, extra_cls="tbl")

    # --- nodes -------------------------------------------------------------------------
    def node_html(self, n: Node, slot: Dict[str, str], avail: float, parent_kind: str, force_abs: bool = False) -> str:
        if n.role == "text":
            s = dict(slot)
            if len(n.lines) > 1 and "flex" not in s:
                s.setdefault("max-width", px(n.box[2] * 1.03 + 3))
            return self.text_html(n, s)
        if n.role == "icon":
            return self.icon_html(n, slot)
        if n.role == "image":
            return self.image_html(n, slot)
        if n.role == "divider":
            return self.divider_html(n, slot)
        if n.role == "control":
            return self.control_html(n, slot)
        if n.role == "table":
            plan = next((q for q in _find_tables(self.root) if q.node is n), None)
            if plan is not None:
                return self.table_html(plan, slot)
        return self.container_html(n, slot, inner_box=(n.lay or {}).get("inner", n.box), avail=avail)

    def control_html(self, n: Node, slot: Dict[str, str]) -> str:
        d = dict(slot)
        d.update(self.box_props(n))
        d.update({"width": px(n.box[2]), "height": px(n.box[3]), "flex": "none"})
        if n.kind == "toggle":
            d["position"] = "relative"
            k = (n.lay or {}).get("knob")
            cls = self.sheet.cls(d)
            if k:
                kd = {"position": "absolute", "top": px(k[1] - n.box[1]), "inset-inline-start": px(k[0] - n.box[0] if not self.rtl_page else (n.box[0] + n.box[2]) - (k[0] + k[2])),
                      "width": px(k[2]), "height": px(k[3]), "border-radius": "50%", "background": css_color((n.lay or {}).get("knob_fill", (255, 255, 255)))}
                inner = self.tag("i", self.sheet.cls(kd))
            else:
                inner = ""
            return self.tag("span", cls, ' role="switch"', inner, extra_cls="toggle", nid=n.id)
        name = "radio" if n.kind == "radio" else "checkbox"
        if n.kind == "radio":
            d["border-radius"] = "50%"
        d["appearance"] = "none"
        d["-webkit-appearance"] = "none"
        d["margin"] = d.get("margin", "0")
        return self.tag("input", self.sheet.cls(d), f' type="{name}" disabled', void=True, extra_cls=name, nid=n.id)

    def container_html(self, n: Node, slot: Dict[str, str], inner_box: Box, avail: float, bleed: bool = False) -> str:
        d = dict(slot)
        if not bleed:
            d.update(self.box_props(n))
        plan: Optional[Plan] = (n.lay or {}).get("plan")
        x, y, w, h = n.box
        ix, iy, iw, ih = inner_box
        # content box = fill box (inside the border)
        bl = (n.border or {}).get("left", (0, 0))[0]
        br = (n.border or {}).get("right", (0, 0))[0]
        bt = (n.border or {}).get("top", (0, 0))[0]
        bb = (n.border or {}).get("bottom", (0, 0))[0]
        cx0, cy0, cx1, cy1 = x + bl, y + bt, x + w - br, y + h - bb
        if bleed:
            cx0, cy0, cx1, cy1 = x, y, x + w, y + h
        rtl = self.rtl_page
        # sizing
        if "flex" not in d and "width" not in d and "align-self" not in d:
            pass
        if n.role in ("button", "badge", "input", "box", "card", "nav", "section") and "flex" in d and d["flex"].startswith("none"):
            pass
        sized = False
        if "width" not in d and ("flex" not in d or d.get("flex") == "none"):
            if not bleed and (parent_is_flex(slot) or n.role in ("button", "badge", "input")):
                if "align-self" in slot or "flex" in slot or n.role in ("button", "badge"):
                    d["width"] = px(w)
                    d.setdefault("max-width", "100%")
                    sized = True
        if "width" not in d and "align-self" in slot and not sized:
            d["width"] = px(w)
            d["max-width"] = "100%"
        d.setdefault("min-height", px(h)) if (h > 0 and n.role not in ("badge",)) else None
        if n.role == "input" and not plan:
            d["height"] = px(h)
            d.pop("min-height", None)
        if not plan:
            tg, extra = self.semantic(n)
            if n.role == "input":
                return self.tag("input", self.sheet.cls({**d, "color": "inherit"}), ' type="text" placeholder="' + esc(n.text) + '"' if n.text else ' type="text"', void=True, extra_cls="input", nid=n.id)
            return self.tag(tg, self.sheet.cls(d), extra, "", extra_cls=n.role if n.role != "box" else "", nid=n.id)
        # padding from the plan's box
        pb = plan.box
        pad_top = pb[1] - cy0
        pad_bot = cy1 - (pb[1] + pb[3])
        if rtl:
            pad_start, pad_end = cx1 - (pb[0] + pb[2]), pb[0] - cx0
        else:
            pad_start, pad_end = pb[0] - cx0, cx1 - (pb[0] + pb[2])
        centered_wrap = False
        if (plan.kind in ("col", "item", "row", "stack", "table")):
            sym = abs(pad_start - pad_end) <= 3 and pad_start >= 24 and (cx1 - cx0) >= 600
            if sym:
                centered_wrap = True
        pads = {}
        if plan.kind == "stack":
            pass
        d["display"] = d.get("display", "flex")
        if plan.kind == "row":
            d["flex-direction"] = "row"
            gd = self.group_decls(plan, cx1 - cx0)
            d.update({k: v for k, v in gd.items() if k not in ("display",)})
        else:
            d["flex-direction"] = "column"
        if n.role in ("button",):
            d.update({"align-items": "center", "justify-content": "center", "white-space": "nowrap", "cursor": "pointer", "font": "inherit"})
            d["flex-direction"] = "row"
            d["flex-wrap"] = "nowrap"
        if n.role == "badge":
            d.update({"align-items": "center", "justify-content": "center", "white-space": "nowrap", "flex-direction": "row", "flex-wrap": "nowrap"})
        pt, pbm = max(0, round(pad_top)), max(0, round(pad_bot))
        ps, pe = max(0, round(pad_start)), max(0, round(pad_end))
        if centered_wrap and not bleed and False:
            pass
        if centered_wrap and n.role in ("section", "nav") or (bleed and centered_wrap):
            cw = cx1 - cx0 - ps - pe
            d["padding-block"] = f"{px(pt)} {px(pbm)}"
            d["padding-inline-start"] = f"max({px(min(ps, 24))},calc((100% - {px(cw)})/2))"
            d["padding-inline-end"] = f"max({px(min(pe, 24))},calc((100% - {px(cw)})/2 - 6px))"
        else:
            if plan.kind == "row":
                pe = max(0, pe - 6)   # a little slack at the end so exact-fit rows never wrap
            d["padding"] = f"{px(pt)} {px(pe)} {px(pbm)} {px(ps)}" if not rtl else f"{px(pt)} {px(ps)} {px(pbm)} {px(pe)}"
            # logical mapping: ps = start, pe = end; in rtl physical right = start
            iw_ = max(1.0, cx1 - cx0)
            if rtl:
                d["padding"] = f"{px(pt)} {fluid(ps, iw_)} {px(pbm)} {fluid(pe, iw_)}"
            else:
                d["padding"] = f"{px(pt)} {fluid(pe, iw_)} {px(pbm)} {fluid(ps, iw_)}"
        tg, extra = self.semantic(n)
        if plan.kind == "stack":
            inner_html = self.stack_inner(plan)
            d["display"] = "block"
            d["position"] = d.get("position", "relative")
            d["height"] = px(h)
            d.pop("min-height", None)
            d["padding"] = "0"
            return self.tag(tg, self.sheet.cls(d), extra, "\n" + inner_html + "\n", extra_cls=n.role if n.role != "box" else "", nid=n.id)
        if plan.kind in ("col", "row"):
            inner_html = self.kids_html(plan, cx1 - cx0 - ps - pe, d)
        elif plan.kind == "table":
            inner_html = self.table_html(plan, {})
        else:
            only = plan.node
            s: Dict[str, str] = {}
            b = plan.box
            self.cross_align(s, b, (cx0 + (pad_end if rtl else pad_start), cy0, cx1 - cx0 - ps - pe, 0), rtl, plan)
            inner_html = self.plan_html(plan, s, cx1 - cx0 - ps - pe, "col")
        return self.tag(tg, self.sheet.cls(d), extra, "\n" + inner_html + "\n", extra_cls=((n.role if n.role != "box" else "") + (" r" if (plan.kind == "row" and n.role not in ("button", "badge")) else "")).strip(), nid=n.id)

    def stack_inner(self, plan: Plan) -> str:
        G = plan.box
        out = []
        for k in plan.kids:
            n = k.node
            kslot = {"position": "absolute", "left": px(k.box[0] - G[0]), "top": px(k.box[1] - G[1])}
            if n.role == "text":
                kslot["white-space"] = "nowrap"
            elif n.role not in ("icon", "image"):
                kslot["width"] = px(k.box[2])
                kslot["height"] = px(k.box[3])
            out.append(self.node_html(n, kslot, G[2], "stack", force_abs=True))
        return "\n".join(out)


def _find_tables(root: Node):
    from .layout import _tables_in
    for n in root.walk():
        pl = (n.lay or {}).get("plan")
        if pl is not None:
            yield from _tables_in(pl)


def fluid(v: float, avail: float, min_px: float = 48.0) -> str:
    """Large horizontal offsets scale with the container (min(Npx, P%)); small ones stay fixed."""
    if v < min_px or avail <= 0:
        return px(v)
    return f"min({px(v)},{round(118.0 * v / avail, 1)}%)"


def parent_is_flex(slot) -> bool:
    return any(k in slot for k in ("flex", "align-self", "margin-inline-start", "margin-top"))


def emit(root: Node, mode: str, rtl_page: bool, size, opts=None, adjust=None) -> dict:
    em = Emitter(root, mode, rtl_page, size, opts, adjust)
    return em.document()
