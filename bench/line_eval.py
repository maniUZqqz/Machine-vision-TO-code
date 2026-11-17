"""Line-level OCR evaluation on real pages whose text is known from the DOM.

    python -m bench.line_eval --shots DIR [--backends tesseract,...] [--show 12] [--json out.json]

DIR holds ``NAME.png`` + ``NAME.lines.json`` (a list of ``{"box": [x, y, w, h], "text": "..."}`` read from a browser's DOM; the NM runner
``shot --lines-out`` writes exactly this).  Every page goes through the converter's own text stage (pixel line finder -> OCR ->
junk filter -> Persian clean-up) and each DOM line is matched to the recognised line that covers it most.  Reported per backend:

* ``exact``  share of DOM lines read exactly (letters/digits normalised, spaces and ZWNJ ignored);
* ``cer``    character error rate over all DOM characters (a missing line counts as 100% wrong);
* ``found``  share of DOM lines that got any recognised text; ``junk`` recognised lines that match no DOM line (icons read as letters ...).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import cv2

from . import metrics as M
from mv2code import persian
from mv2code.build import postprocess_lines, read_text
from mv2code.ocr import get_backend
from mv2code.pipeline import analyze


def _cover(a, b) -> float:
    """share of the smaller box that lies inside the other."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return inter / max(1.0, min(a[2] * a[3], b[2] * b[3]))


def evaluate(img, gt, backend, lang="auto", glossary=None):
    """Every DOM line is matched to the recognised lines that cover it.  A DOM text node and a pixel line are not the same thing (a button
    label and its neighbour may be one pixel line; one DOM line may be cut by an icon), so lines that share a covering are compared as one group,
    in reading order."""
    an = analyze(img)
    lines, _non = read_text(an, backend, lang)
    postprocess_lines(lines, glossary=glossary) if glossary else postprocess_lines(lines)
    # union-find over DOM lines (0..n-1) and recognised lines (n..)
    n = len(gt)
    parent = list(range(n + len(lines)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    used = set()
    for i, g in enumerate(gt):
        for j, ln in enumerate(lines):
            if _cover(g["box"], ln.box) >= 0.5:
                parent[find(i)] = find(n + j)
                used.add(j)
    groups = {}
    for k in range(n + len(lines)):
        groups.setdefault(find(k), []).append(k)
    rows = []
    for members in groups.values():
        gts = [gt[k] for k in members if k < n]
        ocr = [lines[k - n] for k in members if k >= n]
        if not gts:
            continue

        def order(items, boxf, textf):
            rtl = sum(1 for it in items if persian.has_arabic_script(textf(it))) * 2 >= len(items)
            return sorted(items, key=lambda it: (round(boxf(it)[1] / 12), -boxf(it)[0] if rtl else boxf(it)[0]))
        want = " ".join(g["text"] for g in order(gts, lambda g: g["box"], lambda g: g["text"]))
        got = " ".join(o.text for o in order(ocr, lambda o: o.box, lambda o: o.text))
        w, h = M.norm(want), M.norm(got)
        cer = 1.0 if not h else min(1.0, M.lev(w, h) / max(1, len(w)))
        rows.append({"want": want, "got": got, "exact": bool(w) and w == h, "cer": cer, "n": len(w), "found": bool(ocr), "k": len(gts)})
    junk = [lines[j].text for j in range(len(lines)) if j not in used]
    return rows, junk, an


def summarise(rows, junk):
    """``lines`` counts DOM lines (a group of k DOM lines counts k times); exact/found are shares of DOM lines, cer of DOM characters."""
    total = max(1, sum(r["k"] for r in rows))
    chars = max(1, sum(r["n"] for r in rows))
    return {"lines": sum(r["k"] for r in rows), "exact": round(sum(r["exact"] * r["k"] for r in rows) / total, 3),
            "found": round(sum(r["found"] * r["k"] for r in rows) / total, 3), "cer": round(sum(r["cer"] * r["n"] for r in rows) / chars, 3), "junk": len(junk)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", required=True)
    ap.add_argument("--backends", default="tesseract")
    ap.add_argument("--lang", default="auto")
    ap.add_argument("--glossary", default="", help="comma separated brand words (never changed by the word-list repair)")
    ap.add_argument("--show", type=int, default=0, help="print this many wrong lines per page")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    names = sorted(f[:-4] for f in os.listdir(a.shots) if f.endswith(".png") and os.path.exists(os.path.join(a.shots, f[:-4] + ".lines.json")))
    out = {}
    for bname in a.backends.split(","):
        be = get_backend(bname)
        tot_rows, tot_junk = [], []
        t0 = time.time()
        per = {}
        for nm in names:
            img = cv2.cvtColor(cv2.imread(os.path.join(a.shots, nm + ".png")), cv2.COLOR_BGR2RGB)
            gt = json.load(open(os.path.join(a.shots, nm + ".lines.json"), encoding="utf-8"))
            rows, junk, _an = evaluate(img, gt, be, a.lang, [g for g in a.glossary.split(',') if g])
            per[nm] = summarise(rows, junk)
            tot_rows += rows
            tot_junk += junk
            print(f"  {bname:10s} {nm:22s} {per[nm]}")
            for r in [r for r in rows if not r["exact"]][:a.show]:
                print(f"      want: {r['want']}\n      got : {r['got']}")
            for j in junk[:a.show]:
                print(f"      junk: {j}")
        s = summarise(tot_rows, tot_junk)
        s["seconds"] = round(time.time() - t0, 1)
        print(f"{bname:10s} ALL {s}")
        out[bname] = {"all": s, "pages": per}
    if a.json:
        json.dump(out, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
