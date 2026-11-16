# Changelog

All notable changes. Numbers come from `python -m bench` on the 80 generated pages (+ the 2 real samples), see [README.md](README.md).

## [2.0.0] - 2026-11-16

First tagged release. A rewrite of the converter around an honest benchmark; the v1 engine is kept (`MV2CODE_ENGINE=legacy`, `vision_engine/`, `code_generator/`).

### Added
- `bench/`: honest benchmark. 80 generated pages (10 kinds x fa/en x light/dark x 2 seeds, Vazirmatn/Inter, Persian digits) with exact ground truth + the 2 real samples; metrics for text (presence, CER, word order, digit script), structure (P/R/F1 at IoU>=0.5 per role), visual (SSIM, strict pixel similarity), editability, responsive re-flow at 360/768/1280, speed and peak RAM. `python -m bench run --quick|--full [--mode exact|responsive] [--refine on|off]`, `compare`, `rescore`.
- `mv2code/`: pip-installable package (`pip install -e .`), `convert(image, mode='responsive'|'exact', lang='auto'|'fa'|'en', ocr=..., refine=...)` -> `{html, css, tree_json, report, assets}`, `mv2code` CLI (`-o out/ --mode responsive|exact|both --score --zip`), `write_bundle` / `zip_bytes`.
  Pipeline: smooth-pixel region analysis -> ink-tight text lines -> Persian-aware OCR (Tesseract `fas+eng` default, EasyOCR optional, or your own `ocr_lines`) -> font/size/weight/colour by analysis-by-synthesis -> XY-cut layout (flex rows/columns, tables) -> HTML/CSS -> optional browser feedback loop.
- Component tree JSON (`docs/tree-json.md`) for site builders.
- Django web app: upload -> original and generated page side by side with honest scores -> ZIP; REST `POST /api/convert/`.
- 55 pytest tests (unit per stage, CLI and Django end-to-end, golden pages with score thresholds, browser stages), GitHub Actions workflow (`.github/workflows/ci.yml`), README in Persian and English, `docs/yolo.md`.
- Bundled fonts: Vazirmatn and Inter (SIL OFL 1.1, licences in `mv2code/fonts/`).

### Changed
- `qqz.md` rewritten: the v1 claim "91.4% / 95.5% pixel similarity" is **withdrawn** (that metric gives a blank page ~88%).
- Benchmark default is `--refine off` (deterministic: re-running reproduces the committed `bench/results*.json` exactly).
- `DJANGO_ENV=production` now refuses the development fallback `SECRET_KEY` (set a real one in the environment).

### Fixed (found in the final clean-room check)
- `POST /api/convert/` answered `403 CSRF verification failed` to plain `curl -F image=@x.png ...` (the Django test client does not enforce CSRF, so the tests did not see it). The REST endpoint is now CSRF-exempt (stateless; the HTML upload form stays protected) and accepts `languages=fa,en` as well as repeated fields. Test added with `enforce_csrf_checks=True`.
- `playwright install chromium` can be blocked on some networks: README/qqz.md now say to point `MV2CODE_CHROME` at an existing Chromium.

### Measured (v1 -> v2, same benchmark; v1 on a 6-page subset + the 2 real samples because EasyOCR needs ~2 min/page on 2 CPUs)
- Persian text present as real text 26.3% -> 82.0% (same 6 pages; 62.4% on all 80); Persian CER 64.0% -> 13.3% (31.3% on all 80).
- English CER 29.0% -> 6.9% (23.8% on all 80). Structure macro-F1 15.5% -> 85.1% (83.2% on all 80).
- Editable area 89.2% -> 100.0% (99.3% on all 80). Speed 118 s -> ~4 s per page, peak RAM 1.85 GB -> ~220 MB.
- Real photo-like sample `1122.png`: v1 SSIM 82.7% (by pasting crops, 12.8% editable) vs v2 52.2% (67.5% in exact mode, 59% editable) - v2 is worse there, honestly.
- Known limits: see README ("Known limits").

### Not included
- No licence file (the owner has not chosen one).

## [1.0.0] - 2026-02-26 (untagged)
- Initial version: Django project converting a screenshot to HTML/CSS with classical computer vision (EasyOCR, contour detection, absolute layout).
