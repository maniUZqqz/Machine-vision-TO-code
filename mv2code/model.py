"""The component tree (what ``tree_json`` serialises)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .imaging import Box, hex_of


@dataclass
class Node:
    role: str                      # page section card nav button input badge control image icon text divider table
    box: Box                       # border-box (text: ink box), image px
    children: List["Node"] = field(default_factory=list)
    text: str = ""
    kind: str = ""                 # sub-type: h1 h2 h3 p label | checkbox radio toggle | select | row col ...
    # --- visual style
    fill: Optional[tuple] = None
    grad: Optional[dict] = None
    border: Optional[dict] = None  # side -> (width, rgb)
    radius: float = 0.0
    shadow: Optional[dict] = None
    shape: str = "rect"
    # --- text
    lines: List[tuple] = field(default_factory=list)      # [(ink box, text)]
    font: str = ""
    weight: int = 400
    size: float = 0.0
    color: Optional[tuple] = None
    rtl: bool = False
    align: str = "start"           # start | end | center
    line_h: float = 0.0            # css line-height px
    lbox: Optional[Box] = None     # css content box of a text block (for layout)
    conf: float = 1.0
    # --- media
    src: str = ""                  # data URI (raster crop)
    svg: str = ""                  # inline SVG (traced icon)
    # --- layout result (filled by layout.py)
    lay: Optional[dict] = None
    id: int = 0

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()

    def to_json(self, with_src: bool = False) -> dict:
        d = dict(id=self.id, role=self.role, box=[int(v) for v in self.box])
        if self.kind:
            d["kind"] = self.kind
        if self.text:
            d["text"] = self.text
        if self.fill is not None:
            d["background"] = hex_of(self.fill)
        if self.grad:
            d["gradient"] = dict(angle=self.grad["angle"], from_=hex_of(self.grad["c1"]), to=hex_of(self.grad["c2"]))
            d["gradient"]["from"] = d["gradient"].pop("from_")
        if self.border:
            d["border"] = {k: dict(width=v[0], color=hex_of(v[1])) for k, v in self.border.items()}
        if self.radius:
            d["radius"] = round(self.radius, 1)
        if self.shadow:
            d["shadow"] = self.shadow
        if self.role == "text":
            d["font"] = dict(family=self.font, weight=self.weight, size=self.size, color=hex_of(self.color or (0, 0, 0)))
            d["direction"] = "rtl" if self.rtl else "ltr"
            d["align"] = self.align
            if len(self.lines) > 1:
                d["lines"] = [t for _, t in self.lines]
        if self.role == "image":
            d["embedded_raster"] = True
        if self.role == "icon":
            d["vector"] = bool(self.svg)
        if self.children:
            d["children"] = [c.to_json(with_src) for c in self.children]
        return d
