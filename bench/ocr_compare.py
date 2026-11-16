"""Compare OCR backends on the benchmark pages:  python -m bench.ocr_compare --backends tesseract,easyocr

Candidates (text-line boxes) are found by mv2code's own pixel analysis, so every engine reads
exactly the same crops.  Each crop is matched to the ground-truth text line it overlaps most
(IoU >= 0.4) and the character error rate is computed per script.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import cv2
import numpy as np

from . import metrics as M
from .run import DEFAULT_DATA, prep_gt
from mv2code import persian
from mv2code.ocr import get_backend, prepare_crop
from mv2code.pipeline import analyze


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="tesseract")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--langs", default="auto")
    ap.add_argument("--postprocess", action="store_true", help="apply Persian normalisation to the OCR text")
    a = ap.parse_args(argv)
    ids = sorted(f[:-5] for f in os.listdir(os.path.join(a.data, "gt")) if f.endswith(".json"))
    if a.limit:
        ids = ids[:a.limit]
    pre = {}
    for pid in ids:
        img = cv2.cvtColor(cv2.imread(os.path.join(a.data, "shots", pid + ".png")), cv2.COLOR_BGR2RGB)
        gt = prep_gt(json.load(open(os.path.join(a.data, "gt", pid + ".json"), encoding="utf-8")))
        an = analyze(img)
        glines = []
        for t in gt["texts"]:
            if t["kind"] != "text":
                continue
            for ln in t.get("lines", []):
                glines.append((ln.get("ink") or ln["box"], ln["text"]))
        pre[pid] = (img, an, glines)
    for bname in a.backends.split(","):
        be = get_backend(bname)
        agg = {"fa": [0, 0], "en": [0, 0]}
        t0 = time.time()
        ncrop = 0
        for pid, (img, an, glines) in pre.items():
            crops = [prepare_crop(img, c.box) for c in an.cands]
            ncrop += len(crops)
            res = be.read(crops, a.langs)
            lines_txt = [r.text for r in res]
            pdig = persian.decide_page_digits(lines_txt) if a.postprocess else "fa"
            for c, r in zip(an.cands, res):
                best, bi = 0.4, None
                for j, (gb, gtxt) in enumerate(glines):
                    v = M.iou(c.box, gb)
                    if v > best:
                        best, bi = v, j
                if bi is None:
                    continue
                gtxt = glines[bi][1]
                txt = persian.normalize_line(r.text, pdig) if a.postprocess else r.text
                g = M.norm(gtxt)
                if not g:
                    continue
                cer = min(1.0, M.lev(g, M.norm(txt)) / len(g))
                sc = M.script_of(gtxt)
                agg[sc][0] += cer * len(g)
                agg[sc][1] += len(g)
        dt = time.time() - t0
        print(f"{bname:10s} pages={len(pre)} crops={ncrop} time={dt:.1f}s ({dt / max(1, len(pre)):.1f}s/page)  "
              f"CER fa={agg['fa'][0] / max(1, agg['fa'][1]):.3f} ({agg['fa'][1]} chars)  "
              f"en={agg['en'][0] / max(1, agg['en'][1]):.3f} ({agg['en'][1]} chars)")


if __name__ == "__main__":
    main()
