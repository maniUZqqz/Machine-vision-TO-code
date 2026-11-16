"""Benchmark metrics.  Everything here is deliberately strict: a page made of image
crops of the original must score ~0 on text/editability even if its pixels match.
"""
from __future__ import annotations

import re
import unicodedata

import cv2
import numpy as np

try:  # fast edit distance
    from rapidfuzz.distance import Levenshtein as _Lev

    def lev(a: str, b: str) -> int:
        return _Lev.distance(a, b)
except Exception:  # pragma: no cover - pure python fallback
    def lev(a: str, b: str) -> int:
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]

EVAL_ROLES = ["button", "input", "image", "text", "card", "table", "nav"]
EXTRA_ROLES = ["icon", "badge", "control"]
SPEC_ROLES = EVAL_ROLES

_AR2FA = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ة": "ه", "ۀ": "ه", "أ": "ا", "إ": "ا", "ؤ": "و", "ئ": "ی"})
_DIG = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩٬٫", "01234567890123456789,.")
_STRIP = re.compile("[‌‍‎‏‪-‮ـً-ٰٟ﻿]")
_AR_RE = re.compile("[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
_FA_DIG = set("۰۱۲۳۴۵۶۷۸۹")


def norm(s: str, keep_space: bool = False) -> str:
    """Normalise text for comparison: Arabic/Persian letter forms, digits, ZWNJ, spaces."""
    s = unicodedata.normalize("NFKC", s)
    s = _STRIP.sub("", s).translate(_AR2FA).translate(_DIG)
    s = re.sub(r"\s+", " ", s).strip()
    return s if keep_space else s.replace(" ", "")


def script_of(s: str) -> str:
    ar = len(_AR_RE.findall(s))
    la = len(re.findall("[A-Za-z]", s))
    if ar == 0 and any(ch in _FA_DIG for ch in s):
        return "fa"
    return "fa" if ar >= la and ar > 0 else "en"


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, x1 - x0), max(0.0, y1 - y0)
    inter = iw * ih
    u = aw * ah + bw * bh - inter
    return inter / u if u > 0 else 0.0


def _center_in(box, outer, m=3.0) -> bool:
    cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
    return outer[0] - m <= cx <= outer[0] + outer[2] + m and outer[1] - m <= cy <= outer[1] + outer[3] + m


def _reading_order(nodes, rtl: bool):
    if not nodes:
        return nodes
    hs = sorted(n["box"][3] for n in nodes)
    tol = max(3.0, 0.6 * hs[len(hs) // 2])
    nodes = sorted(nodes, key=lambda n: n["box"][1] + n["box"][3] / 2)
    lines, cur = [], [nodes[0]]
    for n in nodes[1:]:
        cy = n["box"][1] + n["box"][3] / 2
        ly = sum(m["box"][1] + m["box"][3] / 2 for m in cur) / len(cur)
        if abs(cy - ly) <= tol:
            cur.append(n)
        else:
            lines.append(cur)
            cur = [n]
    lines.append(cur)
    out = []
    for ln in lines:
        out += sorted(ln, key=lambda n: n["box"][0] + n["box"][2] / 2, reverse=rtl)
    return out


def _fuzzy_eq(a: str, b: str) -> bool:
    if a == b:
        return True
    m = max(len(a), len(b))
    return m > 0 and 1 - lev(a, b) / m >= 0.6


def _lcs(a, b, eq) -> int:
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            tmp = dp[j]
            dp[j] = prev + 1 if eq(x, y) else max(dp[j], dp[j - 1])
            prev = tmp
    return dp[-1]


def text_eval(gt_texts, out_texts) -> dict:
    """Per-page text metrics (raw sums so that callers can pool across pages).

    *Content* matching (the headline metric): every ground-truth text is matched with the output text that
    reads it best - either the text nodes sitting inside its box, or (when those read worse) the best still
    unused text node anywhere on the page.  Wrong *placement* is judged by the visual and structure metrics;
    ``placed`` below counts only the first, position-bound kind of match.
    """
    out = [o for o in out_texts if not o.get("occluded") and norm(o["text"])]
    for o in out:
        o["_n"] = norm(o["text"])
    gt_boxes = [g["box"] for g in gt_texts]
    acc = {s: dict(chars=0.0, err=0.0, present=0.0, placed=0.0, n=0, order_num=0.0, order_den=0.0,
                   dig_ok=0.0, dig_n=0.0) for s in ("fa", "en")}
    fs_errs, col_errs, bold_ok = [], [], []
    used = set()
    order = sorted(range(len(gt_texts)), key=lambda i: -len(norm(gt_texts[i]["text"])))
    for gi in order:
        g = gt_texts[gi]
        gtxt = norm(g["text"])
        if not gtxt:
            continue
        sc = script_of(g["text"])
        gb = g["box"]
        rtl = g.get("dir", "ltr") == "rtl"
        cand = _reading_order([o for o in out if _center_in(o["box"], gb)], rtl)
        otxt = norm(" ".join(o["text"] for o in cand))
        cer_s = min(1.0, lev(gtxt, otxt) / len(gtxt)) if otxt else 1.0
        best, cer_g, node_g = cand, cer_s, None
        if cer_s > 0.15:
            for k, o in enumerate(out):
                if k in used or len(o["_n"]) > 2 * len(gtxt) + 6:
                    continue
                c = lev(gtxt, o["_n"]) / len(gtxt)
                if c < cer_g - 1e-9:
                    cer_g, node_g = min(1.0, c), k
            if node_g is not None:
                best = [out[node_g]]
                used.add(node_g)
        if node_g is None:
            for o in cand:
                used.add(next((k for k, q in enumerate(out) if q is o), -1))
        cer = cer_g
        a = acc[sc]
        w = len(gtxt)
        a["chars"] += w
        a["err"] += cer * w
        a["n"] += 1
        if cer < 0.5:
            a["present"] += w
        if cer_s < 0.5:
            a["placed"] += w
        chosen_text = " ".join(o["text"] for o in best)
        gd = [ch for ch in g["text"] if ch.isdigit()]
        if gd and cer < 0.5:
            same = sum(1 for ch in chosen_text if ch.isdigit() and ((ch in _FA_DIGITS_SET) == (gd[0] in _FA_DIGITS_SET)))
            a["dig_ok"] += min(len(gd), same)
            a["dig_n"] += len(gd)
        gw = norm(g["text"], True).split()
        ow = norm(chosen_text, True).split()
        if len(gw) >= 2 and ow:
            matched = sum(1 for x in gw if any(_fuzzy_eq(x, y) for y in ow))
            if matched >= 2:
                a["order_num"] += _lcs(gw, ow, _fuzzy_eq)
                a["order_den"] += matched
        if cer < 0.5 and len(best) >= 1:
            o = max(best, key=lambda n: len(n["text"]))
            if g.get("fs"):
                fs_errs.append(abs(o["fs"] - g["fs"]) / g["fs"])
            col_errs.append(_color_dist(o["color"], g["color"]))
            bold_ok.append(1.0 if (o["fw"] >= 600) == (g["fw"] >= 600) else 0.0)
    tot_out = sum(len(o["_n"]) for o in out)
    extra = sum(len(o["_n"]) for o in out if not any(_center_in(o["box"], b, 4.0) for b in gt_boxes))
    return dict(by_script=acc, fs_err=fs_errs, col_err=col_errs, bold_ok=bold_ok,
                out_chars=tot_out, extra_chars=extra)


_FA_DIGITS_SET = set("۰۱۲۳۴۵۶۷۸۹")


def _parse_rgb(c: str):
    m = re.findall(r"[\d.]+", c or "")
    if len(m) >= 3:
        return [float(m[0]), float(m[1]), float(m[2])]
    return [0.0, 0.0, 0.0]


def _color_dist(a: str, b: str) -> float:
    pa, pb = np.array(_parse_rgb(a)), np.array(_parse_rgb(b))
    return float(np.linalg.norm(pa - pb))


def struct_eval(gt_elems, pred_elems, roles=None, thr: float = 0.5) -> dict:
    """Pooled TP/FP/FN per role at IoU >= thr (greedy, highest IoU first)."""
    roles = roles or (EVAL_ROLES + EXTRA_ROLES)
    res = {}
    for role in roles:
        g = [e for e in gt_elems if e["role"] == role and not (role == "text" and e.get("inner"))]
        p = [e for e in pred_elems if e["role"] == role and not (role == "text" and e.get("inner"))]
        pairs = []
        for i, ge in enumerate(g):
            for j, pe in enumerate(p):
                v = iou(ge["box"], pe["box"])
                if v >= thr:
                    pairs.append((v, i, j))
        pairs.sort(reverse=True)
        ug, up, tp = set(), set(), 0
        for v, i, j in pairs:
            if i in ug or j in up:
                continue
            ug.add(i)
            up.add(j)
            tp += 1
        res[role] = dict(tp=tp, fp=len(p) - tp, fn=len(g) - tp)
    return res


def prf(tp: float, fp: float, fn: float):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


# --------------------------------------------------------------------------- visual
def _pad_to(img: np.ndarray, H: int, W: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == H and w == W:
        return img
    out = np.empty((H, W, 3), np.uint8)
    out[:] = img[-1:, -1:, :] if h else 255
    out[:min(h, H), :min(w, W)] = img[:min(h, H), :min(w, W)]
    if h < H:
        out[h:, :min(w, W)] = img[-1:, :min(w, W)]
    return out


def ssim_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = (11, 11)
    mu_a = cv2.GaussianBlur(a, k, 1.5)
    mu_b = cv2.GaussianBlur(b, k, 1.5)
    saa = cv2.GaussianBlur(a * a, k, 1.5) - mu_a * mu_a
    sbb = cv2.GaussianBlur(b * b, k, 1.5) - mu_b * mu_b
    sab = cv2.GaussianBlur(a * b, k, 1.5) - mu_a * mu_b
    return ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2))


def visual_eval(gt_img: np.ndarray, out_img: np.ndarray, text_boxes=None) -> dict:
    H = max(gt_img.shape[0], out_img.shape[0])
    W = max(gt_img.shape[1], out_img.shape[1])
    a, b = _pad_to(gt_img, H, W), _pad_to(out_img, H, W)
    ga, gb = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    sm = ssim_map(ga, gb)
    pix = 1.0 - float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16)))) / 255.0
    # the metric the original project reported (share of pixels within 30 grey levels)
    d = np.maximum(cv2.absdiff(ga, gb), np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32)), axis=2).astype(np.uint8))
    legacy = float(np.mean(d < 30))
    # control: a blank page in the dominant border colour
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    vals, cnt = np.unique(border, axis=0, return_counts=True)
    blank = np.empty_like(a)
    blank[:] = vals[cnt.argmax()]
    db = np.maximum(cv2.absdiff(ga, cv2.cvtColor(blank, cv2.COLOR_RGB2GRAY)),
                    np.mean(np.abs(a.astype(np.float32) - blank.astype(np.float32)), axis=2).astype(np.uint8))
    res = dict(ssim=float(sm.mean()), pix_sim=pix, legacy_match=legacy, blank_match=float(np.mean(db < 30)),
               size_gt=[int(gt_img.shape[1]), int(gt_img.shape[0])], size_out=[int(out_img.shape[1]), int(out_img.shape[0])])
    if text_boxes:
        mask = np.zeros((H, W), bool)
        for x, y, w, h in text_boxes:
            mask[max(0, int(y) - 2):min(H, int(y + h) + 2), max(0, int(x) - 2):min(W, int(x + w) + 2)] = True
        if mask.any():
            res["ssim_text"] = float(sm[mask].mean())
    return res


def editability_eval(W: int, H: int, rasters, gt_elems) -> dict:
    """Share of the page area that is real HTML/CSS (rasters inside GT image boxes are allowed)."""
    raster = np.zeros((H, W), bool)
    for r in rasters:
        x, y, w, h = r["box"]
        raster[max(0, int(y)):min(H, int(y + h + 0.5)), max(0, int(x)):min(W, int(x + w + 0.5))] = True
    allowed = np.zeros((H, W), bool)
    for g in gt_elems:
        if g["role"] == "image":
            x, y, w, h = g["box"]
            allowed[max(0, int(y) - 3):min(H, int(y + h) + 4), max(0, int(x) - 3):min(W, int(x + w) + 4)] = True
    bad = int((raster & ~allowed).sum())
    denom = max(1, H * W - int(allowed.sum()))
    return dict(editable=max(0.0, 1.0 - bad / denom), raster_px=int(raster.sum()), bad_px=bad)


def responsive_check(probe: dict, width: int) -> dict:
    """Horizontal overflow and overlapping text of one render at ``width`` px."""
    overflow = probe["W"] > width + 1
    texts = [t for t in probe["texts"] if not t.get("occluded")]
    boxes = [t["box"] for t in texts]
    overlaps = 0
    for i in range(len(boxes)):
        ax, ay, aw, ah = boxes[i]
        for j in range(i + 1, len(boxes)):
            bx, by, bw, bh = boxes[j]
            iw = min(ax + aw, bx + bw) - max(ax, bx)
            ih = min(ay + ah, by + bh) - max(ay, by)
            if iw > 1 and ih > 1 and iw * ih > 0.25 * min(aw * ah, bw * bh):
                overlaps += 1
    return dict(width=width, overflow=bool(overflow), overlaps=overlaps, ok=bool((not overflow) and overlaps == 0))
