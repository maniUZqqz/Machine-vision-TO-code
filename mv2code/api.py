"""Public API: ``convert(image, mode='responsive', lang='auto', ...)``."""
from __future__ import annotations

import time
from typing import Optional

from . import __version__
from .build import build_tree, postprocess_lines, read_text
from .emit import emit, font_assets
from .imaging import load_image
from .layout import layout_tree
from .ocr import Backend, CallbackBackend, get_backend, prepare_crop
from .pipeline import analyze
from . import review as _review


def convert(image, mode: str = "responsive", lang: str = "auto", ocr: str = "auto", refine="auto",
            browser=None, dpr: Optional[float] = None, ocr_lines=None, title: str = "Converted page",
            max_width: int = 2400, options: Optional[dict] = None, glossary=None) -> dict:
    """Convert a UI screenshot into HTML/CSS.

    Returns ``{"html", "css", "tree_json", "report", "assets"}``.  ``mode`` is ``responsive``
    (flex/table layout, semantic tags) or ``exact`` (absolute, pixel-faithful).
    """
    if mode not in ("responsive", "exact"):
        raise ValueError("mode must be 'responsive' or 'exact'")
    if lang not in ("auto", "fa", "en"):
        raise ValueError("lang must be 'auto', 'fa' or 'en'")
    opts = dict(options or {})
    t0 = time.perf_counter()
    rgb, info = load_image(image, max_width=max_width, dpr=dpr)
    t1 = time.perf_counter()
    an = analyze(rgb)
    t2 = time.perf_counter()
    if ocr_lines is not None:
        backend = get_backend("none")
    elif isinstance(ocr, Backend):
        backend = ocr
    elif callable(ocr):
        backend = CallbackBackend(ocr)
    else:
        backend = get_backend(ocr)
    lines, nontext = read_text(an, backend, lang, ocr_lines)
    t3 = time.perf_counter()
    pdig, page_dir = postprocess_lines(lines, glossary=glossary, repair=opts.get("repair", True) and ocr_lines is None)
    n_check = _review.flag_lines(lines, glossary, from_ocr=ocr_lines is None and backend.name != "none")
    if lang == "fa":
        page_dir = "rtl"
    elif lang == "en":
        page_dir = "ltr"
    rtl = page_dir == "rtl"
    root = build_tree(an, lines, nontext, page_dir, opts)
    layout_tree(root, rtl)
    out = emit(root, mode, rtl, an.size, opts)
    t4 = time.perf_counter()
    refine_report = None
    if refine in (True, "on", "auto"):
        refine = True if refine == "on" else refine
        refine_report, out = _maybe_refine(rgb, root, mode, rtl, an.size, opts, out, browser, refine)
    t5 = time.perf_counter()
    result = dict(html=out["html"], css=out["css"], tree_json=_tree_json(root, an.size, page_dir, info),
                  assets=[dict(path=a["path"], source=a["source"]) for a in font_assets()],
                  report=dict(version=__version__, mode=mode, lang=lang, page_direction=page_dir, ocr=backend.name, ocr_stats=getattr(backend, "stats", None),
                              digits="persian" if pdig == "fa" else "latin", size=list(an.size), scale=info["scale"],
                              regions=len(an.regions), text_lines=len(lines), nodes=sum(1 for _ in root.walk()),
                              review_lines=n_check, review=_review.report_items(lines),
                              seconds=dict(load=round(t1 - t0, 3), analyze=round(t2 - t1, 3), ocr=round(t3 - t2, 3),
                                           build=round(t4 - t3, 3), refine=round(t5 - t4, 3), total=round(t5 - t0, 3)),
                              refine=refine_report))
    if opts.get("return_root"):          # internal/debug: the live Node tree (not JSON serialisable)
        result["_root"] = root
    return result


def ocr_crops(image, dpr: Optional[float] = None, max_width: int = 2400) -> dict:
    """The text crops the converter would read, for engines that run elsewhere (a GPU machine, a service).

    Returns ``{"size": [w, h], "items": [{"box": [x, y, w, h], "crop": ndarray (dark text on white, grey)}]}`` - send the crops, get ``text`` back
    and pass ``ocr=PrecomputedBackend([{"box", "text", "conf"}, ...])`` to ``convert`` (same image, same ``dpr``/``max_width``)."""
    from .build import select_candidates
    rgb, _info = load_image(image, max_width=max_width, dpr=dpr)
    an = analyze(rgb)
    cands = select_candidates(an)
    return {"size": [int(rgb.shape[1]), int(rgb.shape[0])], "items": [{"box": [int(v) for v in c.box], "crop": prepare_crop(rgb, c.box)} for c in cands]}


def _maybe_refine(rgb, root, mode, rtl, size, opts, out, browser, refine):
    """Run the browser feedback loop when a headless browser is available (``refine='auto'``)."""
    from .emit import emit, font_assets
    own = None
    try:
        br = browser
        if br is None:
            try:
                from .browser import Browser
                own = Browser().__enter__()
                br = own
            except Exception as e:  # no playwright / no browser: not an error in 'auto' mode
                if refine is True:
                    raise RuntimeError(f"refine=True but no headless browser is available: {e}") from e
                return dict(skipped=f"no browser ({type(e).__name__})"), out
        from .refine import refine as run_refine
        best, rep = run_refine(rgb, root, lambda adj: emit(root, mode, rtl, size, opts, adj), br, font_assets(), rtl)
        return rep, best
    except RuntimeError:
        raise
    except Exception as e:  # the loop must never break a conversion
        return dict(skipped=f"{type(e).__name__}: {e}"), out
    finally:
        if own is not None:
            own.__exit__(None, None, None)


def _tree_json(root, size, page_dir, info):
    d = root.to_json()
    d.update(width=size[0], height=size[1], direction=page_dir, source_scale=info["scale"])
    return d
