"""Benchmark driver:  python -m bench run --quick | --full  [--engine mv2code|legacy] [--mode responsive|exact]"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
try:
    import resource
except ImportError:  # Windows: peak RAM not reported
    resource = None
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

from . import metrics as M
from .render import Browser, ensure_fonts
from .templates import make_page, page_specs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = os.environ.get("MV2CODE_BENCH_DATA", os.path.join(ROOT, "bench_cache"))
REAL_DIR = os.path.join(ROOT, "test_samples", "input")
REAL_GT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "real_gt")
RESP_WIDTHS = (360, 768, 1280)


# --------------------------------------------------------------------------- dataset
def ink_box(img: np.ndarray, box, color=None, thr: int = 28):
    """Tight box of the visible ink inside ``box``: pixels close to the text colour (``color``,
    a CSS ``rgb(...)`` string) when given, otherwise pixels that differ from the local background."""
    import re
    H, W = img.shape[:2]
    x0, y0 = max(0, int(box[0]) - 1), max(0, int(box[1]) - 1)
    x1, y1 = min(W, int(box[0] + box[2]) + 2), min(H, int(box[1] + box[3]) + 2)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return list(box)
    c = img[y0:y1, x0:x1].astype(np.int16)
    flat = c.reshape(-1, 3)
    vals, cnt = np.unique(flat[:: max(1, len(flat) // 20000)], axis=0, return_counts=True)
    bg = vals[cnt.argmax()]
    m = None
    if color:
        v = re.findall(r"[\d.]+", color)
        if len(v) >= 3:
            fg = np.array([float(v[0]), float(v[1]), float(v[2])])
            span = float(np.abs(fg - bg).max())
            if span >= 40:
                m = (np.abs(c - fg).max(axis=2) <= 0.45 * span)
    if m is None:
        m = np.abs(c - bg).max(axis=2) > thr
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return list(box)
    return [float(x0 + xs.min()), float(y0 + ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)]


def add_ink_boxes(img: np.ndarray, probe: dict) -> None:
    for t in probe["texts"]:
        t["ink"] = ink_box(img, t["box"], t.get("color"))
        for ln in t.get("lines", []):
            ln["ink"] = ink_box(img, ln["box"], t.get("color"))
    for g in probe["gt"]:
        if g["role"] == "text":
            g["ink"] = ink_box(img, g["box"], g.get("color"))


def build_dataset(data_dir: str, specs, force: bool = False, log=print) -> list[str]:
    """Render the synthetic pages once (screenshot + ground truth).  Returns the page ids."""
    pages_dir = os.path.join(data_dir, "pages")
    os.makedirs(os.path.join(data_dir, "shots"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "gt"), exist_ok=True)
    os.makedirs(pages_dir, exist_ok=True)
    ensure_fonts(pages_dir)
    ids, todo = [], []
    for spec in specs:
        pid = "%s-%s-%s-%d" % spec
        ids.append(pid)
        if force or not os.path.exists(os.path.join(data_dir, "gt", pid + ".json")):
            todo.append(spec)
    if todo:
        with Browser() as br:
            for spec in todo:
                pg = make_page(*spec)
                path = os.path.join(pages_dir, pg["id"] + ".html")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(pg["html"])
                img, probe = br.render("file://" + path, pg["width"])
                cv2.imwrite(os.path.join(data_dir, "shots", pg["id"] + ".png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                add_ink_boxes(img, probe)
                probe["meta"] = dict(id=pg["id"], kind=pg["kind"], lang=pg["lang"], theme=pg["theme"], width=pg["width"])
                with open(os.path.join(data_dir, "gt", pg["id"] + ".json"), "w", encoding="utf-8") as f:
                    json.dump(probe, f, ensure_ascii=False)
                log(f"  generated {pg['id']}  {img.shape[1]}x{img.shape[0]}  gt={len(probe['gt'])} texts={len(probe['texts'])}")
    return ids


# --------------------------------------------------------------------------- one page
def _peak_mb() -> float:
    if resource is None:
        return 0.0
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def prep_gt(gt: dict) -> dict:
    """Text boxes are compared as *ink* boxes (what a detector can see), not CSS font boxes."""
    for t in gt["texts"]:
        if t.get("ink"):
            t["box"] = t["ink"]
    for g in gt["gt"]:
        if g["role"] == "text" and g.get("ink"):
            g["box"] = g["ink"]
    return gt


def eval_page(br, adapter, pid: str, data_dir: str, out_dir: str, responsive: bool = True) -> dict:
    shot = cv2.cvtColor(cv2.imread(os.path.join(data_dir, "shots", pid + ".png")), cv2.COLOR_BGR2RGB)
    with open(os.path.join(data_dir, "gt", pid + ".json"), encoding="utf-8") as f:
        gt = prep_gt(json.load(f))
    res = _convert_and_render(br, adapter, pid, shot, out_dir, responsive)
    return _score(res, shot, gt)


def _convert_and_render(br, adapter, pid, shot, out_dir, responsive=True) -> dict:
    t0 = time.perf_counter()
    err = None
    try:
        conv = adapter.convert(shot, browser=br)
    except Exception as e:  # a crash is a result, not an exception
        import traceback
        err = traceback.format_exc()
        conv = dict(html="<!doctype html><html><body></body></html>", elements=[], meta={}, assets=[])
    dt = time.perf_counter() - t0
    os.makedirs(out_dir, exist_ok=True)
    ensure_fonts(out_dir)
    path = os.path.join(out_dir, pid + ".html")
    for a in conv.get("assets", []) or []:
        ap = os.path.join(out_dir, a["path"])
        os.makedirs(os.path.dirname(ap), exist_ok=True)
        if a.get("bytes") is not None:
            with open(ap, "wb") as f:
                f.write(a["bytes"])
    with open(path, "w", encoding="utf-8") as f:
        f.write(conv["html"])
    W = shot.shape[1]
    out_img, probe = br.render("file://" + path, W)
    resp = []
    if responsive:
        for w in RESP_WIDTHS:
            _, pr = br.render("file://" + path, w, shot=False)
            resp.append(M.responsive_check(pr, w))
    return dict(conv=conv, seconds=dt, peak_mb=_peak_mb(), out_img=out_img, probe=probe, resp=resp, error=err,
                html_bytes=len(conv["html"].encode("utf-8")))


def _score(res, shot, gt) -> dict:
    conv, probe = res["conv"], res["probe"]
    H, W = shot.shape[:2]
    tm = M.text_eval(gt["texts"], probe["texts"])
    vis = M.visual_eval(shot, res["out_img"], [t["box"] for t in gt["texts"]])
    st = M.struct_eval(gt["gt"], conv["elements"])
    ed = M.editability_eval(W, H, probe["rasters"], gt["gt"])
    return dict(text=tm, visual=vis, struct=st, edit=ed, responsive=res["resp"], seconds=res["seconds"],
                peak_mb=res["peak_mb"], html_bytes=res["html_bytes"], error=res["error"],
                n_text_nodes=len(probe["texts"]), n_raster=len(probe["rasters"]))


def eval_real(br, adapter, name: str, out_dir: str, responsive=True) -> dict:
    shot = cv2.cvtColor(cv2.imread(os.path.join(REAL_DIR, name)), cv2.COLOR_BGR2RGB)
    res = _convert_and_render(br, adapter, "real-" + name.rsplit(".", 1)[0], shot, out_dir, responsive)
    H, W = shot.shape[:2]
    vis = M.visual_eval(shot, res["out_img"])
    ed = M.editability_eval(W, H, res["probe"]["rasters"], [])
    lines = []
    gp = os.path.join(REAL_GT, name.rsplit(".", 1)[0] + ".txt")
    if os.path.exists(gp):
        lines = [l.strip() for l in open(gp, encoding="utf-8") if l.strip()]
    nodes = [norm for norm in (M.norm(t["text"]) for t in res["probe"]["texts"] if not t.get("occluded")) if norm]
    cov = []
    for ln in lines:
        g = M.norm(ln)
        best = min((M.lev(g, n) / max(1, len(g)) for n in nodes), default=1.0)
        # also allow the line to be a part of a longer node / spread over a few nodes
        joined = "".join(nodes)
        if g and g in joined:
            best = 0.0
        cov.append(min(1.0, best))
    return dict(visual=vis, edit=ed, responsive=res["resp"], seconds=res["seconds"], peak_mb=res["peak_mb"],
                n_text_nodes=len(res["probe"]["texts"]), n_raster=len(res["probe"]["rasters"]), error=res["error"],
                transcript_lines=len(lines), transcript_found=sum(1 for c in cov if c < 0.5),
                transcript_cer=(sum(cov) / len(cov)) if cov else None, html_bytes=res["html_bytes"])


# --------------------------------------------------------------------------- aggregate
def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(sum(xs) / len(xs)) if xs else None


def aggregate(pages: dict) -> dict:
    P = list(pages.values())
    agg = {"pages": len(P)}
    # text (pooled over characters)
    for sc in ("fa", "en"):
        chars = sum(p["text"]["by_script"][sc]["chars"] for p in P)
        err = sum(p["text"]["by_script"][sc]["err"] for p in P)
        pres = sum(p["text"]["by_script"][sc]["present"] for p in P)
        placed = sum(p["text"]["by_script"][sc].get("placed", 0.0) for p in P)
        on = sum(p["text"]["by_script"][sc]["order_num"] for p in P)
        od = sum(p["text"]["by_script"][sc]["order_den"] for p in P)
        dok = sum(p["text"]["by_script"][sc]["dig_ok"] for p in P)
        dn = sum(p["text"]["by_script"][sc]["dig_n"] for p in P)
        agg[f"text_{sc}"] = dict(chars=chars, cer=(err / chars) if chars else None,
                                 present=(pres / chars) if chars else None, placed=(placed / chars) if chars else None,
                                 word_order=(on / od) if od else None, digit_script=(dok / dn) if dn else None)
    allc = sum(agg[f"text_{s}"]["chars"] for s in ("fa", "en"))
    agg["text_all"] = dict(
        placed=sum((agg[f"text_{s}"]["placed"] or 0) * agg[f"text_{s}"]["chars"] for s in ("fa", "en")) / max(1, allc),
        present=sum((agg[f"text_{s}"]["present"] or 0) * agg[f"text_{s}"]["chars"] for s in ("fa", "en")) / max(1, allc),
        cer=sum((agg[f"text_{s}"]["cer"] or 0) * agg[f"text_{s}"]["chars"] for s in ("fa", "en")) / max(1, allc))
    oc = sum(p["text"]["out_chars"] for p in P)
    agg["text_extra_chars"] = sum(p["text"]["extra_chars"] for p in P) / max(1, oc)
    fs = [x for p in P for x in p["text"]["fs_err"]]
    ce = [x for p in P for x in p["text"]["col_err"]]
    bo = [x for p in P for x in p["text"]["bold_ok"]]
    agg["style"] = dict(font_size_err_median=float(np.median(fs)) if fs else None,
                        color_dist_median=float(np.median(ce)) if ce else None, bold_acc=_mean(bo))
    # structure (pooled)
    roles = M.EVAL_ROLES + M.EXTRA_ROLES
    st = {}
    for r in roles:
        tp = sum(p["struct"][r]["tp"] for p in P)
        fp = sum(p["struct"][r]["fp"] for p in P)
        fn = sum(p["struct"][r]["fn"] for p in P)
        pr, rc, f1 = M.prf(tp, fp, fn)
        st[r] = dict(tp=tp, fp=fp, fn=fn, precision=pr, recall=rc, f1=f1, support=tp + fn)
    agg["structure"] = st
    agg["structure_macro_f1_spec"] = _mean([st[r]["f1"] for r in M.EVAL_ROLES if st[r]["support"]])
    agg["structure_macro_f1_all"] = _mean([st[r]["f1"] for r in roles if st[r]["support"]])
    # visual
    agg["visual"] = dict(ssim=_mean([p["visual"]["ssim"] for p in P]), pix_sim=_mean([p["visual"]["pix_sim"] for p in P]),
                         ssim_text=_mean([p["visual"].get("ssim_text") for p in P]),
                         legacy_match=_mean([p["visual"]["legacy_match"] for p in P]),
                         blank_page_legacy_match=_mean([p["visual"]["blank_match"] for p in P]))
    agg["editable"] = _mean([p["edit"]["editable"] for p in P])
    rr = [r["ok"] for p in P for r in p["responsive"]]
    agg["responsive_ok"] = _mean([1.0 if x else 0.0 for x in rr]) if rr else None
    agg["responsive_overflow"] = _mean([1.0 if r["overflow"] else 0.0 for p in P for r in p["responsive"]]) if rr else None
    agg["responsive_overlap_pages"] = _mean([1.0 if any(r["overlaps"] for r in p["responsive"]) else 0.0 for p in P]) if rr else None
    agg["seconds_per_page"] = _mean([p["seconds"] for p in P])
    agg["seconds_max"] = max(p["seconds"] for p in P)
    agg["peak_ram_mb"] = max(p["peak_mb"] for p in P)
    agg["errors"] = sum(1 for p in P if p["error"])
    return agg


def group_text(pages: dict, key: str) -> dict:
    out = {}
    for val in sorted({p["_meta"][key] for p in pages.values()}):
        sub = {k: p for k, p in pages.items() if p["_meta"][key] == val}
        a = aggregate(sub)
        out[val] = dict(pages=a["pages"], ssim=a["visual"]["ssim"], editable=a["editable"], text_all=a["text_all"],
                        f1=a["structure_macro_f1_spec"])
    return out


# --------------------------------------------------------------------------- printing
def fmt(x, pct=True, nd=1):
    if x is None:
        return "  -  "
    return f"{x * 100:.{nd}f}%" if pct else f"{x:.{nd}f}"


def print_table(res: dict, log=print):
    a = res["overall"]
    log("")
    log(f"=== mv2code benchmark :: engine={res['meta']['engine']} mode={res['meta']['mode']} pages={a['pages']} "
        f"set={res['meta']['set']} ===")
    for sc, lab in (("fa", "TEXT Persian"), ("en", "TEXT English")):
        t = a[f"text_{sc}"]
        log(f"{lab:14s} present {fmt(t['present'])}  CER {fmt(t['cer'])}  placed-in-its-box {fmt(t['placed'])}  word-order {fmt(t['word_order'])}  "
            f"digit-script {fmt(t['digit_script'])}  ({int(t['chars'])} chars)")
    log(f"TEXT all       present {fmt(a['text_all']['present'])}  CER {fmt(a['text_all']['cer'])}  "
        f"extra-text {fmt(a['text_extra_chars'])}")
    s = a["style"]
    log(f"STYLE          font-size error (median) {fmt(s['font_size_err_median'])}  colour dist (median) "
        f"{s['color_dist_median'] if s['color_dist_median'] is None else round(s['color_dist_median'], 1)}  bold acc {fmt(s['bold_acc'])}")
    log("STRUCTURE      IoU>=0.5      precision  recall   F1    (support)")
    for r in M.EVAL_ROLES + M.EXTRA_ROLES:
        x = a["structure"][r]
        tag = " " if r in M.EVAL_ROLES else "*"
        log(f"  {tag}{r:10s}                {fmt(x['precision']):>8s} {fmt(x['recall']):>8s} {fmt(x['f1']):>7s}   ({x['support']})")
    log(f"  macro-F1 (7 spec roles) {fmt(a['structure_macro_f1_spec'])}   all roles {fmt(a['structure_macro_f1_all'])}")
    v = a["visual"]
    log(f"VISUAL         SSIM {fmt(v['ssim'])}  pixel-sim {fmt(v['pix_sim'])}  SSIM-in-text-areas {fmt(v['ssim_text'])}  "
        f"| old 'similarity' {fmt(v['legacy_match'])} (blank page would get {fmt(v['blank_page_legacy_match'])})")
    log(f"EDITABILITY    real HTML/CSS share of page area {fmt(a['editable'])}")
    log(f"RESPONSIVE     ok at 360/768/1280: {fmt(a['responsive_ok'])}  (overflow {fmt(a['responsive_overflow'])}, pages with overlapping text {fmt(a['responsive_overlap_pages'])})")
    log(f"SPEED/RAM      {a['seconds_per_page']:.1f} s/page (max {a['seconds_max']:.1f})   peak RAM {a['peak_ram_mb']:.0f} MB   errors {a['errors']}")
    for name, r in res.get("real", {}).items():
        tl = r["transcript_lines"]
        log(f"REAL {name:10s} SSIM {fmt(r['visual']['ssim'])}  pixel-sim {fmt(r['visual']['pix_sim'])}  old-sim {fmt(r['visual']['legacy_match'])}  "
            f"editable {fmt(r['edit']['editable'])}  text nodes {r['n_text_nodes']}  transcript found {r['transcript_found']}/{tl}  "
            f"{r['seconds']:.1f}s")
    log("")


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return ""


def run(args) -> int:
    full = args.full
    specs = page_specs(True)
    if not full:
        specs = page_specs(False)
    if args.limit:
        specs = specs[:args.limit]
    if args.ids:
        want = set(args.ids.split(","))
        specs = [s for s in page_specs(True) if ("%s-%s-%s-%d" % s) in want]
    data_dir = os.path.abspath(args.data)
    out_dir = os.path.abspath(args.outdir or os.path.join(data_dir, "out-%s-%s" % (args.engine, args.mode)))
    t_all = time.time()
    ids = build_dataset(data_dir, specs)
    from .adapters import get_adapter
    adapter = get_adapter(args.engine, args.mode, ocr=args.ocr, refine=args.refine)
    pages = {}
    with Browser() as br:
        for i, pid in enumerate(ids, 1):
            r = eval_page(br, adapter, pid, data_dir, out_dir, responsive=not args.no_responsive)
            meta = json.load(open(os.path.join(data_dir, "gt", pid + ".json"), encoding="utf-8"))["meta"]
            r["_meta"] = meta
            pages[pid] = r
            v = r["visual"]
            print(f"[{i}/{len(ids)}] {pid:28s} ssim {v['ssim']:.3f} edit {r['edit']['editable']:.2f} "
                  f"{r['seconds']:.1f}s {'ERR' if r['error'] else ''}", flush=True)
            if r["error"] and args.verbose:
                print(r["error"])
        real = {}
        if not args.no_real and os.path.isdir(REAL_DIR):
            for name in sorted(os.listdir(REAL_DIR)):
                if name.lower().endswith((".png", ".jpg", ".jpeg")):
                    real[name] = eval_real(br, adapter, name, out_dir, responsive=not args.no_responsive)
                    print(f"[real] {name} ssim {real[name]['visual']['ssim']:.3f} {real[name]['seconds']:.1f}s", flush=True)
    result = dict(
        meta=dict(engine=args.engine, mode=args.mode, set="full" if full else "quick", date=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                  git=git_rev(), python=platform.python_version(), machine=platform.platform(), cpus=os.cpu_count(),
                  ocr=args.ocr, refine=args.refine, wall_seconds=round(time.time() - t_all, 1)),
        overall=aggregate(pages),
        by_lang=group_text(pages, "lang"), by_theme=group_text(pages, "theme"), by_kind=group_text(pages, "kind"),
        real=real, pages={k: {kk: vv for kk, vv in v.items()} for k, v in pages.items()})
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print_table(result)
    print("results ->", out)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bench", description="Honest benchmark for screenshot -> HTML/CSS")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the benchmark")
    g = r.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true", help="24 pages (default)")
    g.add_argument("--full", action="store_true", help="all 80 synthetic pages")
    r.add_argument("--engine", default="mv2code", choices=["mv2code", "legacy"])
    r.add_argument("--mode", default="responsive", choices=["responsive", "exact"])
    r.add_argument("--ocr", default="auto")
    r.add_argument("--refine", default="off", choices=["auto", "on", "off"],
                   help="browser feedback loop in the converter under test (default off = deterministic, the setting of the README tables)")
    r.add_argument("--data", default=DEFAULT_DATA, help="cache dir for generated pages/ground truth")
    r.add_argument("--outdir", default=None)
    r.add_argument("--out", default=None, help="results json path (default bench/results.json)")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--ids", default="")
    r.add_argument("--no-responsive", action="store_true")
    r.add_argument("--no-real", action="store_true")
    r.add_argument("--verbose", action="store_true")
    r.set_defaults(fn=run)
    gen = sub.add_parser("gen", help="only (re)generate the pages + ground truth")
    gen.add_argument("--full", action="store_true")
    gen.add_argument("--data", default=DEFAULT_DATA)
    gen.add_argument("--force", action="store_true")
    gen.set_defaults(fn=lambda a: (build_dataset(os.path.abspath(a.data), page_specs(a.full), a.force), 0)[1])
    rs = sub.add_parser("rescore", help="recompute the text metrics of an existing results file from the saved output pages")
    rs.add_argument("results")
    rs.add_argument("--data", default=DEFAULT_DATA)
    rs.add_argument("--outdir", required=True, help="folder with the saved <page id>.html outputs")
    rs.set_defaults(fn=lambda a: rescore(a.results, os.path.abspath(a.data), os.path.abspath(a.outdir)))
    cmp_ = sub.add_parser("compare", help="print a BEFORE/AFTER table from two results.json files")
    cmp_.add_argument("before")
    cmp_.add_argument("after")
    cmp_.add_argument("--md", action="store_true", help="markdown table")
    cmp_.set_defaults(fn=lambda a: compare(a.before, a.after, a.md))
    args = ap.parse_args(argv)
    return args.fn(args)


def rescore(results_path: str, data_dir: str, out_dir: str) -> int:
    """Re-render the saved outputs and recompute ONLY the text metrics (used when the text metric changed)."""
    res = json.load(open(results_path, encoding="utf-8"))
    with Browser() as br:
        for pid, page in res["pages"].items():
            gt = prep_gt(json.load(open(os.path.join(data_dir, "gt", pid + ".json"), encoding="utf-8")))
            w = gt["meta"]["width"]
            _, probe = br.render("file://" + os.path.join(out_dir, pid + ".html"), w, shot=False)
            page["text"] = M.text_eval(gt["texts"], probe["texts"])
    res["overall"] = aggregate(res["pages"])
    for key, name in (("by_lang", "lang"), ("by_theme", "theme"), ("by_kind", "kind")):
        res[key] = group_text(res["pages"], name)
    json.dump(res, open(results_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print_table(res)
    return 0


def _rows(bo: dict, ao: dict, co: dict | None = None):
    def g(d, *path):
        for k in path:
            d = d.get(k) if isinstance(d, dict) else None
        return d
    spec = [("Persian text present as real text", ("text_fa", "present"), True),
            ("Persian text also in the right place", ("text_fa", "placed"), True),
            ("Persian text error rate (CER)", ("text_fa", "cer"), True),
            ("Persian word order correct", ("text_fa", "word_order"), True),
            ("English text present as real text", ("text_en", "present"), True),
            ("English text error rate (CER)", ("text_en", "cer"), True),
            ("Structure macro-F1 (7 roles, IoU>=0.5)", ("structure_macro_f1_spec",), True),
            ("SSIM vs screenshot", ("visual", "ssim"), True),
            ("Pixel similarity (strict)", ("visual", "pix_sim"), True),
            ("Editable area (real HTML/CSS)", ("editable",), True),
            ("Responsive ok (360/768/1280 px)", ("responsive_ok",), True),
            ("Seconds per page", ("seconds_per_page",), False),
            ("Peak RAM (MB)", ("peak_ram_mb",), False)]
    rows = []
    for name, path, pct in spec:
        vals = [g(bo, *path), g(ao, *path)] + ([g(co, *path)] if co else [])
        rows.append((name, vals, pct))
    return rows


def compare(before: str, after: str, markdown: bool = False) -> int:
    """BEFORE (old engine) vs AFTER (new engine) on the SAME pages (+ AFTER on every page it ran)."""
    b, a = json.load(open(before, encoding="utf-8")), json.load(open(after, encoding="utf-8"))
    common = sorted(set(b["pages"]) & set(a["pages"]))
    bsub = aggregate({k: b["pages"][k] for k in common})
    asub = aggregate({k: a["pages"][k] for k in common})
    rows = _rows(bsub, asub, a["overall"])
    head = ["metric", f"BEFORE (old code, {len(common)} pages)", f"AFTER (same {len(common)} pages)", f"AFTER (all {a['overall']['pages']} pages)"]
    f = lambda v, pct: "-" if v is None else (f"{v * 100:.1f}%" if pct else f"{v:.1f}")
    if markdown:
        print("| " + " | ".join(head) + " |")
        print("|" + "---|" * len(head))
        for n, vals, pct in rows:
            print(f"| {n} | " + " | ".join(f(v, pct) for v in vals) + " |")
        for name in sorted(set(b.get("real", {})) & set(a.get("real", {}))):
            rb, ra = b["real"][name], a["real"][name]
            print(f"| real sample {name}: SSIM / editable | {rb['visual']['ssim'] * 100:.1f}% / {rb['edit']['editable'] * 100:.1f}% | "
                  f"{ra['visual']['ssim'] * 100:.1f}% / {ra['edit']['editable'] * 100:.1f}% | |")
    else:
        print(f"{head[0]:42s} {head[1]:>28s} {head[2]:>26s} {head[3]:>22s}")
        for n, vals, pct in rows:
            print(f"{n:42s} " + " ".join(f"{f(v, pct):>26s}" for v in vals))
    return 0
