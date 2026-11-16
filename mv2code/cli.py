"""Command line:  ``mv2code input.png -o out/ --mode responsive``"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="mv2code", description="Convert a UI screenshot into HTML/CSS (classical computer vision).")
    ap.add_argument("image", help="screenshot (png/jpg/webp)")
    ap.add_argument("-o", "--out", default="mv2code-out", help="output folder (default: mv2code-out)")
    ap.add_argument("--mode", default="responsive", choices=["responsive", "exact", "both"],
                    help="responsive = flex/table layout, semantic tags; exact = pixel-faithful absolute layout")
    ap.add_argument("--lang", default="auto", choices=["auto", "fa", "en"], help="text language / direction")
    ap.add_argument("--ocr", default="auto", help="OCR backend: auto, tesseract, easyocr, none")
    ap.add_argument("--refine", default="auto", choices=["auto", "on", "off"], help="browser feedback loop (needs playwright)")
    ap.add_argument("--dpr", type=float, default=None, help="device pixel ratio of the screenshot (2 for retina)")
    ap.add_argument("--score", action="store_true", help="render the result and print SSIM / editable share (needs playwright)")
    ap.add_argument("--zip", action="store_true", help="also write <out>/result.zip")
    ap.add_argument("--json", action="store_true", help="print the report as JSON on stdout")
    ap.add_argument("--version", action="version", version=f"mv2code {__version__}")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from . import ImageError, convert
    from .bundle import write_bundle, zip_bytes
    from .imaging import load_image
    try:
        rgb, _ = load_image(args.image, dpr=args.dpr)
    except ImageError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    modes = ["responsive", "exact"] if args.mode == "both" else [args.mode]
    reports = {}
    for mode in modes:
        try:
            res = convert(args.image, mode=mode, lang=args.lang, ocr=args.ocr, refine=args.refine, dpr=args.dpr)
        except ImageError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        except (RuntimeError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 3
        name = "index" if len(modes) == 1 else mode
        write_bundle(res, args.out, name, original=rgb)
        rep = dict(res["report"])
        if args.score:
            try:
                from .browser import Browser
                from .score import compare
                with Browser() as br:
                    rep["score"] = compare(rgb, res["html"], res["assets"], br)
            except Exception as e:
                rep["score"] = dict(error=f"{type(e).__name__}: {e}")
        reports[mode] = rep
        if args.zip:
            with open(os.path.join(args.out, "result.zip"), "wb") as f:
                f.write(zip_bytes(res, name))
        if not args.json:
            sec = rep["seconds"]["total"]
            print(f"[{mode}] {rep['page_direction']} | {rep['regions']} regions, {rep['text_lines']} text lines, {rep['nodes']} nodes | "
                  f"OCR: {rep['ocr']} | {sec:.1f}s -> {os.path.join(args.out, name + '.html')}")
            if "score" in rep:
                print("      score:", rep["score"])
    if args.json:
        print(json.dumps(reports, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
