"""Stage 4: layout inference (recursive XY-cut) -> rows / columns / tables / stacks.

For every container the children are cut recursively by empty horizontal / vertical bands.
Rows become ``display:flex`` rows, columns ``display:flex`` columns, groups of aligned rows
become ``<table>`` and anything that cannot be cut (overlaps) becomes an absolute *stack*.
All geometry is kept in image pixels; ``emit.py`` turns plans into CSS.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .imaging import Box, box_union
from .model import Node


@dataclass
class Plan:
    kind: str                       # item | col | row | stack | table
    node: Optional[Node] = None
    kids: List["Plan"] = field(default_factory=list)
    box: Box = (0, 0, 0, 0)         # layout bounding box
    rtl: bool = False
    meta: dict = field(default_factory=dict)
    gid: int = 0


def lay_box(n: Node) -> Box:
    """Box used for positioning: text uses its CSS content box, everything else its border-box."""
    if n.role == "text" and n.lbox is not None:
        return n.lbox
    return n.box


def _bbox(boxes) -> Box:
    b = boxes[0]
    for q in boxes[1:]:
        b = box_union(b, q)
    return b


def _intervals(items, axis: int, min_gap: int = 0):
    """Group items whose [lo, hi) intervals overlap along ``axis`` (0 = x, 1 = y)."""
    order = sorted(items, key=lambda n: n.box[axis])
    groups, cur, hi = [], [], None
    for n in order:
        lo_n, hi_n = n.box[axis], n.box[axis] + n.box[2 + axis]
        if cur and lo_n >= hi + min_gap:
            groups.append(cur)
            cur, hi = [], None
        cur.append(n)
        hi = hi_n if hi is None else max(hi, hi_n)
    if cur:
        groups.append(cur)
    return groups


def _cut(items: List[Node], rtl: bool, depth: int = 0) -> Plan:
    boxes = [n.box for n in items]
    gbox = _bbox([lay_box(n) for n in items])
    if len(items) == 1:
        return Plan("item", items[0], [], gbox, rtl)
    if depth > 12:
        return Plan("stack", None, [Plan("item", n, [], lay_box(n), rtl) for n in items], gbox, rtl)
    # vertical stacking first (rows of content), then side by side
    for axis in (1, 0):
        groups = _intervals(items, axis)
        if len(groups) >= 2:
            kids = [_cut(g, rtl, depth + 1) for g in groups]
            if axis == 0 and rtl:
                kids.reverse()
            kind = "col" if axis == 1 else "row"
            return Plan(kind, None, kids, gbox, rtl)
    return Plan("stack", None, [Plan("item", n, [], lay_box(n), rtl) for n in items], gbox, rtl)


def _simplify(p: Plan) -> Plan:
    p.kids = [_simplify(k) for k in p.kids]
    # flatten col-in-col / row-in-row
    if p.kind in ("col", "row"):
        flat = []
        for k in p.kids:
            if k.kind == p.kind:
                flat.extend(k.kids)
            else:
                flat.append(k)
        p.kids = flat
    return p


def _row_cells(row: Plan):
    """Cells (boxes) of a row plan, or None when the row is not a simple one-level row."""
    if row.kind == "row":
        return [k for k in row.kids]
    if row.kind == "item":
        return [row]
    return None


def _is_rule(k: Plan) -> bool:
    return k.kind == "item" and k.node is not None and k.node.role == "divider" and k.node.kind != "v"


def detect_table(col: Plan, rtl: bool, prev: Optional[Plan] = None) -> Optional[Plan]:
    """A column of >=3 rows with the same number (>=2) of columns whose edges line up -> table."""
    seq = col.kids
    rows = [k for k in seq if not _is_rule(k)]
    if len(rows) < 3:
        return None
    cells = []
    for r in rows:
        c = _row_cells(r)
        if c is None or len(c) < 2:
            return None
        for k in c:
            nodes = [k.node] if k.kind == "item" else [q.node for q in k.kids if q.kind == "item"]
            if not nodes or any(n is None or n.role in ("card", "section", "nav") or (n.role == "image" and n.box[3] > 80) for n in nodes):
                return None
        cells.append(c)
    ncol = len(cells[0])
    if any(len(c) != ncol for c in cells):
        return None
    # a table has text in at least two columns (an icon + label menu is not a table)
    def has_text(k):
        nodes = [k.node] if k.kind == "item" else [q.node for q in k.kids if q.kind == "item"]
        return any(n is not None and n.role in ("text", "badge", "button") for n in nodes)
    text_cols = sum(1 for j in range(ncol) if sum(has_text(c[j]) for c in cells) >= 0.6 * len(cells))
    if text_cols < 2:
        return None
    ok_cols = 0
    for j in range(ncol):
        L = [c[j].box[0] for c in cells]
        R = [c[j].box[0] + c[j].box[2] for c in cells]
        C = [c[j].box[0] + c[j].box[2] / 2 for c in cells]
        if min(np.ptp(L), np.ptp(R), np.ptp(C)) <= 5:
            ok_cols += 1
    if ok_cols < ncol:
        return None
    cy = [float(np.mean([k.box[1] + k.box[3] / 2 for k in c])) for c in cells]
    pitch = np.diff(cy)
    if len(pitch) >= 2 and (pitch.max() - pitch.min()) > max(6, 0.2 * pitch.mean()):
        return None
    rl, rule_nodes = [], []
    for k in seq:
        if _is_rule(k):
            rl.append((max(1, k.node.box[3]), k.node.fill))
            rule_nodes.append(k.node)
    # evidence of a real table: row rules, or a header band right above
    has_head = False
    if prev is not None and prev.kind == "item" and prev.node is not None and prev.node.role in ("card", "box") and prev.node.children:
        kids_ = prev.node.children
        has_head = len(kids_) == ncol and all(q.role in ("text", "badge") for q in kids_)
    if len(rule_nodes) < 2 and not has_head:
        return None
    rl = (rl + [None] * len(rows))[:len(rows)]
    t = Plan("table", None, [], col.box, rtl, meta=dict(cells=cells, centers=cy, rules=rl, rule_nodes=rule_nodes))
    return t


def _upgrade_tables(p: Plan, rtl: bool) -> Plan:
    p.kids = [_upgrade_tables(k, rtl) for k in p.kids]
    if p.kind == "col":
        t = detect_table(p, rtl, None)
        if t is not None:
            return t
        # table directly followed/preceded by other rows: look for the longest run
        n = len(p.kids)
        best = None
        for i in range(n):
            if _is_rule(p.kids[i]):
                continue
            for j in range(i + 3, n + 1):
                sub = Plan("col", None, p.kids[i:j], _bbox([k.box for k in p.kids[i:j]]), rtl)
                t = detect_table(sub, rtl, p.kids[i - 1] if i > 0 else None)
                if t is not None and (best is None or (j - i) > best[1] - best[0]):
                    best = (i, j, t)
        if best is not None and (best[1] - best[0]) >= 3 and (best[1] - best[0]) <= n:
            i, j, t = best
            p.kids = p.kids[:i] + [t] + p.kids[j:]
    return p


def plan_children(node: Node, rtl: bool) -> Optional[Plan]:
    items = [c for c in node.children]
    if not items:
        return None
    p = _cut(items, rtl)
    p = _simplify(p)
    p = _upgrade_tables(p, rtl)
    return p


def _tables_in(p: Optional[Plan]):
    if p is None:
        return
    if p.kind == "table":
        yield p
    for k in p.kids:
        yield from _tables_in(k)


def _cell_nodes(c: Plan):
    return [c.node] if c.kind == "item" else [q.node for q in c.kids if q.kind == "item"]


def materialize_tables(container: Node, plan: Plan, rtl: bool):
    """Turn table plans into real ``table`` nodes (so they show up in the tree)."""
    for t in list(_tables_in(plan)):
        nodes, boxes = [], []
        for row in t.meta["cells"]:
            for c in row:
                nodes += _cell_nodes(c)
                boxes.append(c.box)
        for r in t.meta.get("rule_nodes", []):
            boxes.append(r.box)
        # optional header band directly above the rows
        top_box = _bbox(boxes)
        for ch in list(container.children):
            if ch.role in ("card", "box") and ch.children and abs((ch.box[1] + ch.box[3]) - top_box[1]) <= 12                     and abs(ch.box[0] - top_box[0]) <= 6 and abs(ch.box[2] - top_box[2]) <= 8:
                texts = [c for c in ch.children if c.role in ("text", "badge")]
                if len(texts) == len(t.meta["cells"][0]) and len(texts) == len(ch.children):
                    t.meta["thead"] = ch
                    boxes.append(ch.box)
                    nodes += ch.children
                    container.children = [c for c in container.children if c is not ch]
                    break
        ub = _bbox(boxes)
        # extend to the bottom rule / half a row pitch
        if len(t.meta["centers"]) >= 2:
            pitch = (t.meta["centers"][-1] - t.meta["centers"][0]) / (len(t.meta["centers"]) - 1)
            ub = (ub[0], ub[1], ub[2], max(ub[3], int(t.meta["centers"][-1] + pitch / 2 - ub[1])))
        tb = Node(role="table", box=tuple(int(v) for v in ub), children=nodes)
        rm = {id(n) for n in nodes} | {id(r) for r in t.meta.get("rule_nodes", [])}
        container.children = [c for c in container.children if id(c) not in rm] + [tb]
        t.node = tb
        t.box = tb.box


def layout_tree(root: Node, rtl: bool):
    """Attach ``lay['plan']`` to every container node (and materialise tables)."""
    for n in list(root.walk()):
        if n.children and n.role not in ("text", "icon", "image", "table"):
            n.lay = dict(n.lay or {})
            plan = plan_children(n, rtl)
            n.lay["plan"] = plan
            if plan is not None and any(True for _ in _tables_in(plan)):
                materialize_tables(n, plan, rtl)
    for i, n in enumerate(root.walk()):
        n.id = i
    counter = [0]

    def number(p):
        if p is None:
            return
        counter[0] += 1
        p.gid = counter[0]
        for k in p.kids:
            number(k)
    for n in root.walk():
        number((n.lay or {}).get("plan"))


def dump_plan(p: Optional[Plan], depth: int = 0) -> str:
    """Human-readable plan (debugging / report)."""
    if p is None:
        return ""
    pad = "  " * depth
    if p.kind == "item":
        n = p.node
        return f"{pad}item {n.role}:{n.kind} {tuple(int(v) for v in p.box)} {n.text[:18]!r}\n"
    out = f"{pad}{p.kind} {tuple(int(v) for v in p.box)}\n"
    for k in p.kids:
        out += dump_plan(k, depth + 1)
    return out
