"""Layout inference (XY-cut -> rows/columns/tables) and HTML/CSS emission on hand-built trees."""
import re

from mv2code.emit import emit
from mv2code.layout import dump_plan, layout_tree
from mv2code.model import Node


def text(x, y, w, h, s, rtl=False, size=16.0):
    n = Node(role="text", box=(x, y, w, h), text=s, lines=[((x, y, w, h), s)], font="Vazirmatn" if rtl else "Inter",
             weight=400, size=size, color=(30, 30, 30), rtl=rtl, line_h=size * 1.5)
    n.lbox = (x, y - 3, w, int(size * 1.5))
    return n


def page(children, w=800, h=600, rtl=False):
    root = Node(role="page", box=(0, 0, w, h), fill=(255, 255, 255), children=children)
    layout_tree(root, rtl)
    return root


def test_column_of_rows():
    kids = [text(20, 20, 200, 18, "Title"),
            text(20, 60, 100, 18, "A"), text(200, 60, 100, 18, "B"), text(380, 60, 100, 18, "C")]
    root = page(kids)
    dump = dump_plan(root.lay["plan"])
    assert dump.splitlines()[0].startswith("col")
    assert "row" in dump


def test_responsive_has_no_absolute_positioning():
    kids = [text(20, 20, 200, 18, "Title"), text(20, 60, 100, 18, "A"), text(200, 60, 100, 18, "B")]
    root = page(kids)
    out = emit(root, "responsive", False, (800, 600))
    assert "position:absolute" not in out["css"]
    assert "display:flex" in out["css"]
    assert 'lang="en"' in out["html"] and 'dir="ltr"' in out["html"]


def test_exact_is_absolute_and_faithful():
    kids = [text(20, 20, 200, 18, "Title")]
    root = page(kids)
    out = emit(root, "exact", False, (800, 600))
    assert "position:absolute" in out["css"] and "left:20px" in out["css"]


def test_rtl_page_and_escaping():
    kids = [text(300, 20, 200, 18, "<b>سلام & دنیا</b>", rtl=True)]
    root = page(kids, rtl=True)
    out = emit(root, "responsive", True, (800, 600))
    assert 'dir="rtl"' in out["html"] and 'lang="fa"' in out["html"]
    assert "&lt;b&gt;" in out["html"] and "<b>" not in out["html"].split("<body>")[1]


def test_colours_become_css_variables():
    a = Node(role="card", box=(10, 10, 300, 100), fill=(37, 99, 235), radius=8.0,
             children=[text(30, 40, 100, 18, "Hello")])
    b = Node(role="card", box=(10, 130, 300, 100), fill=(37, 99, 235), radius=8.0,
             children=[text(30, 160, 100, 18, "World")])
    root = page([a, b])
    out = emit(root, "responsive", False, (800, 600))
    assert ":root{" in out["css"] and "var(--" in out["css"]


def test_table_detection_from_aligned_rows():
    from mv2code.model import Node as N
    rows = []
    for i in range(4):
        y = 100 + i * 50
        rows.append(text(20, y, 80, 18, f"r{i}"))
        rows.append(text(300, y, 80, 18, f"v{i}"))
        rows.append(text(560, y, 80, 18, f"w{i}"))
        if i < 3:
            rows.append(N(role="divider", box=(10, y + 34, 700, 1), fill=(220, 220, 220), kind="h"))
    root = page(rows, h=400)
    tables = [n for n in root.walk() if n.role == "table"]
    assert len(tables) == 1
    out = emit(root, "responsive", False, (800, 400))
    assert "<table" in out["html"] and out["html"].count("<tr") == 4
