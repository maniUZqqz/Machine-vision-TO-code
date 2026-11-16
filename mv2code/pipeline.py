"""Pixel analysis shared by the converter, the benchmark tools and the tests."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from . import segment as sg
from . import textdet as td
from .imaging import box_inter, box_area


@dataclass
class Analysis:
    img: np.ndarray
    bg: tuple
    S: np.ndarray
    regions: List[sg.Region]
    photos: List[sg.Photo]
    lines: List[sg.Line]
    cands: List[td.Cand]
    threshold: int = 3

    @property
    def size(self):
        return self.img.shape[1], self.img.shape[0]


def _dedupe_photos(photos):
    photos = sorted(photos, key=lambda p: -box_area(p.box))
    out = []
    for p in photos:
        if any(box_inter(p.box, q.box) > 0.5 * box_area(p.box) for q in out):
            continue
        out.append(p)
    return out


def analyze(img: np.ndarray) -> Analysis:
    H, W = img.shape[:2]
    S, T = sg.smooth_mask(img)
    soft: list = []
    regs = sg.find_regions(img, S, soft_images=soft)
    for r in regs:
        sg.analyse_outline(img, r)
    regs = sg.merge_stacks(img, regs)
    bg = sg.page_background(img)
    regs = sg.prune_regions(img, regs, W, H, bg)
    regs = sg.prune_regions(img, regs, W, H, bg)   # second pass: borders may have changed owner
    for r in regs:
        r.radius = sg.corner_radius(img, r.box, r.fill) if r.shape == "rect" else r.radius
    photos = _dedupe_photos(sg.find_photos(img, S) + soft)
    for ph in photos:
        if not ph.round and min(ph.box[2], ph.box[3]) >= 24:
            ph.radius = sg.corner_radius(img, ph.box, None, thr=40)
    # regions that live inside a photo are just its smooth parts
    regs = [r for r in regs if not any(sg_inside(r.box, p.box) for p in photos)]
    for i, r in enumerate(regs):
        r.id = i
    sg.build_tree(regs, W, H)
    sg.finalize_radius(regs)
    lines = sg.find_lines(img, S, regs, photos)
    cands = td.find_candidates(img, S, photos, [l.box for l in lines], regs)
    return Analysis(img=img, bg=bg, S=S, regions=regs, photos=photos, lines=lines, cands=cands, threshold=T)


def sg_inside(inner, outer) -> bool:
    return box_inter(inner, outer) >= 0.8 * box_area(inner)
