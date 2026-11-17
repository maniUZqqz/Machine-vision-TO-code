# Changelog

All notable changes. Numbers come from `python -m bench` on the 80 generated pages (+ the 2 real samples), see [README.md](README.md).

## [2.1.1] - 2026-11-17

Same code as 2.1.0. 2.1.0 was committed with Windows line endings (CRLF) by mistake; the text files are LF again and `.gitattributes` keeps them so. Use 2.1.1 (the tag v2.1.0 stays as published).

## [2.1.0] - 2026-11-17

Persian text from real screenshots. Measured on a real landing page (chagher.ir at 1280 and 390 px, text known from the DOM, `python -m bench.line_eval`): lines read exactly 61.5% -> 72.5% (75.2% with the brand word), character error rate 18.9% -> 8.7% (8.5%), lines found 89.0% -> 93.6%; synthetic benchmark (80 pages) Persian CER 31.3% -> 14.1%, English 23.8% -> 6.6%.

### Added
- `mv2code/lexicon.py` + `mv2code/data/fa_words.tsv.gz` (28k Persian words with frequencies from `wordfreq`, CC BY-SA 4.0, see `mv2code/data/NOTICE.md`; rebuild with `tools/build_fa_words.py`): OCR words that are not in the list are repaired with a look-alike-letter cost model (dots: ب پ ت ث ن ی, ج چ ح خ, ف ق, ع غ ...), only towards a clearly more frequent close word; short words, brand words, words repeated 3+ times and rare real words are left alone; half-space restored (`میکند` -> `می‌کند`); `،` `؛` `؟` restored.
- `convert(..., glossary=[...])` (brand words: never changed, misread look-alikes become them), `convert(..., ocr=<callable | Backend>)`, `CallbackBackend`, `PrecomputedBackend` (results matched to crops by box, missing ones fall back to the local engine), `mv2code.ocr_crops(image)` (the crops the converter would read, for engines on another machine), `select_candidates`.
- `bench/line_eval.py` (OCR on real pages with DOM ground truth), `bench/ocr_compare.py --ids`, 16 new tests (`tests/test_ocr_v21.py`; 70 in total).

### Fixed
- A long one-line paragraph (30x wider than tall) was discarded as a rule/bar instead of read as text.
- A label inside a gradient button / pill was lost when the outline and glow merged with it (or two buttons became one candidate): the labels are found again inside the region against its own local background (`textdet._fix_leaky`).
- Icons next to a label were read as letters or digits and became words (`# امکانات`, `زج ورود`, `0 امنیت`): low-confidence Latin fragments and short low-confidence words at the end of a Persian line are dropped; weak lines without any Persian letter are re-read with the Persian-only model.
- Persian lines read with `,` `;` `?` now get `،` `؛` `؟`.

### Evaluated, not shipped
- EasyOCR `arabic` recogniser exported to ONNX (215 MB): real page 52.3% exact / 6.8% CER at 0.7 s per line on 2 CPUs; PP-OCRv5 `arabic_PP-OCRv5_mobile_rec` ONNX (8 MB): 48.6% / 19.3%. Both lose to Tesseract + repair on exact lines; see README.

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
