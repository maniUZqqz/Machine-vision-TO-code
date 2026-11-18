# Changelog

All notable changes. Numbers come from `python -m bench` on the 80 generated pages (+ the 2 real samples), see [README.md](README.md).

## [2.3.0] - 2026-11-18

Smooth colour bands are CSS, never one pasted picture; icon digits and the labels of gradient buttons on real pages.

### Fixed
- **A full-bleed gradient hero was pasted as ONE image** (the most common landing-page design: a hero band under the header whose end colour touches the page colour, e.g. `linear-gradient(135deg,#14264d,#0b1220)` on a `#0b1220` page).
  Header, hero and page background were one connected smooth area; no single gradient fitted it, so `find_regions` took the whole page for a "soft photo" (regions 0, text lines 0, 100 KB of base64, editable share 0-5%).
  Now the area is split into its flat and planar parts (`mv2code/gradfit.py`) - header, gradient hero, flat page - and the hero is a CSS gradient region with its text and buttons as real HTML on top.
- A big smooth area that no gradient model explains is approximated by the best gradient we can fit instead of becoming a picture (text and buttons on top stay real HTML).
- **The label of a button was lost when one text candidate spanned several buttons** (outline/glow merged them): the first button was fixed and the candidate was removed, the others never got their labels back (`شروع رایگان` on a gradient button). Every region is now judged against the original candidates (`textdet._fix_leaky`).
- Icons read as characters: a lone Latin digit at the end of a Persian label (`9 شروع رایگان`), one or two unsure characters set apart from the only word of a nav item (`هص وبلاگ`), a pin/phone icon read as a digit beside a line of words, `in` (a social icon), a glyph whose reading changes between the two passes (`۴` then `0`), two unsure digits in a tile (`88`), two unsure letters in a thin sliver.
- A big region that only looks round by its fill ratio (the quarter of a translucent decoration circle that a card clips) is no longer an ellipse.

### Added
- `mv2code/gradfit.py`: gradient models fitted from the pixels - linear with up to 5 stops at any angle, radial (any centre, `circle farthest-corner`), and a piecewise-planar split of a smooth area (flat colours and linear gradients; a flat colour wins ties so a header never leaks into a gradient that passes through its colour).
  `emit.css_gradient` writes them; `tree_json` carries `gradient.stops` / `kind` / `cx` / `cy` for multi-stop and radial gradients (the two-stop form `angle/from/to` is unchanged).
- Benchmark: `python -m bench run --hero` builds 16 full-bleed gradient hero pages (8 designs x fa/en: seam-dark, seam-light, diagonal dark/light, vivid, three-stop, radial, decorative circles; `bench/templates.py` `HERO_VARIANTS`);
  the page background touches the end colour of the hero in the designs that failed. `bench/results-hero-v2.3.0.json` has before/after per page.
- Tests: `tests/test_gradient_bands.py` (11 numpy tests + 3 browser tests on the generated hero pages in URL mode), `tests/test_ocr_v23.py` (7).

### Measured
Same machine, same pages, 2.2.1 -> 2.3.0.

| | 2.2.1 | **2.3.0** |
|---|---|---|
| 16 gradient hero pages: real HTML/CSS share (editable) | 67.8% | **100%** |
| ... of them the 8 designs that failed: editable / share of the page that became pictures | 35.5% / 64.5% | **100% / 0%** |
| ... regions found on those 8 | 4-6 | **11-16** |
| real chagher.ir 1280 px (54 DOM lines): exact / CER / found / junk lines | 87.0% / 2.3% / 98.1% / 5 | **88.9% / 1.3% / 100% / 1** (a quiet server: 90.7% / 1.1% / 100% / 0) |
| real chagher.ir 390 px (55 lines): exact / CER / found / junk lines | 83.6% / 5.2% / 92.7% / 5 | **92.7% / 0.6% / 100% / 0** |
| real chagher.ir 1280 px rendered output vs the screenshot (SSIM) | 81.6% | **88.7%** |
| 24-page quick set (clean fonts): SSIM / editable / Persian CER / English CER | 85.8% / 99.8% / 14.5% / 2.5% | 85.7% / 99.8% / 14.4% / 2.4% (no change) |
| 24-page quick set: Persian present / structure F1 / responsive ok | 84.0% / 86.8% / 68.1% | 84.3% / 87.0% / 69.4% |

(The chagher.ir numbers are the page as it was on 2026-11-18, both engines on the same screenshots, DOM text as ground truth, Tesseract on CPU, brand word not given; the second OCR pass is time-boxed, so a loaded server gives a slightly lower number than a quiet one.
The 24-page run took 42 s per page against 27 s because the server was busy with other work, not because of the engine: the hero pages convert in 1-8 s.)
Not fixed: about 1 line in 10 on the real page still has a wrong letter (`چفر` for `چغر`, a comma read as a colon), a decorative translucent circle on a gradient card is dropped (the main gradient shows there), and
for 100% exact text the page address (DOM text) is still the answer.

## [2.2.1] - 2026-11-17

Fixes for what the full 80-page benchmark showed right after 2.2.0: table / price / product pages with Persian numbers lost number lines (Persian digit lines that were read before were missing, the digit-script score fell).

### Fixed
- A run of numbers is a left-to-right run, Persian digits included: `۴۱ ۳۶۳` was reversed inside right-to-left lines, and a line that holds only numbers is ordered left to right.
- The "icon read as digits" rules need real Persian *letters* in the line (a line of numbers has none, its short digit tokens are numbers); the compact "icon" rule drops only two-digit readings below 60% confidence (3 digits and clear numbers stay, `۱۰` at 50% is still dropped).
- The second pass never replaces a reading by one the engine trusts clearly less (garbled variants used to win on width or votes), a number is never replaced by a variant with fewer digits, and the "width lost" rule only applies to lines of words.
- Synthetic 24-page quick benchmark (clean fonts), 2.1.1 -> 2.2.1: Persian present 81.2% -> 84.0%, Persian CER 17.0% -> 14.5%, structure macro-F1 85.7% -> 86.8%, SSIM unchanged; worse: Persian text in the right place 66.4% -> 64.9%, responsive re-flow ok 75.0% -> 68.1%, 4.8 s -> 26.6 s per page (`bench/results-v2.2.1-quick.json`).
- Measured on 4 number-heavy Persian synthetic pages (table, pricing, products, dashboard; 236 DOM lines, `bench.line_eval`): 2.1.1 -> 2.2.1 exact lines 72.9% -> 76.3%, CER 15.1% -> 11.3%, lines found 90.3% -> 94.1%; real chagher.ir page unchanged from 2.2.0 (87.2% exact, CER 3.7%).

## [2.2.0] - 2026-11-17

**Use 2.2.1.** 2.2.0 damaged lines of numbers (see below); the figures in this entry are those of 2.2.1.

A second look at the lines OCR is unsure about, and honest marking of what is still doubtful. Measured on the same screenshots as 2.1.1 (`python -m bench.line_eval`, Tesseract on CPU,
brand word given; DOM text is the ground truth): a real Persian landing page (chagher.ir, 1280 px + 390 px, 109 lines) and a synthetic Persian page in Tahoma 13-44 px (1280 + 390 px, 70 lines):

| | 2.1.1 | **2.2.0** |
|---|---|---|
| real page: lines read exactly | 75.2% | **87.2%** (1280 px 88.9%, 390 px 85.5%) |
| real page: character error rate | 8.6% | **3.7%** |
| real page: lines found at all | 93.6% | **95.4%** |
| Tahoma page: lines read exactly | 52.8% | **72.9%** |
| Tahoma page: character error rate | 13.7% | **6.7%** |
| Tahoma page: lines found at all | 85.7% | **92.9%** |
| all four screenshots: exact / CER / found | 66.4% / 10.6% / 90.5% | **81.6% / 4.8% / 94.4%** |
| time per page (1280 x 2561 real page) | ~6 s | ~48 s (bounded, see below) |

### Added
- `mv2code/rescan.py`: the doubtful lines (confidence < 90%, a Persian word the list does not know, nothing found) are read again from the original image through five cheap variants of the crop
  (64 px line height with the strokes thickened by erosion 5x5 / 3x3, gamma-darkened, 48 px gamma + erode, Persian-only model) and the reading the engine itself trusts most is kept:
  `confidence + 0.03 x (variants that agree) + 0.1 x word-list score - (width lost against the widest reading)`; a variant must win by a margin, so the first reading is never replaced by something worse.
  Measured on 172 single-line crops: the default crop 62.2% exact, the best single variant (thickened 5x5 at 64 px) 68.6%, best-of-five by confidence 74.4%, oracle 79.1%.
  Bounded: at most 90 lines and `MV2CODE_RESCAN_BUDGET` seconds (default 75; least confident lines first); `MV2CODE_RESCAN=off` switches it off. Only the Tesseract backend has it.
- Single doubtful Latin words / numbers inside a Persian line are re-read on their own with the English model (`PDF` read as `۳۵۴`, `100%` as `96100`); every candidate reading gets this before the readings are compared, so a variant that reads `PDF` as digits with a high confidence cannot win.
- `mv2code/review.py`: lines a person should look at again - OCR confidence below 80%, a Persian word the list does not know (brand words and words repeated 3+ times excluded), Persian letters glued to digits/Latin
  letters, a stray single character at an end. `convert()` returns them as `report["review"]` (`[{box, text, conf, why}]`, least confident first) and `report["review_lines"]`; a text node of the tree gets `conf` and `"review": true`.
  Text that did not come from OCR (`ocr_lines`, e.g. read from a browser's DOM) is never flagged.
- `lexicon.Lexicon.affix_known` / `is_known`: a known stem + a common suffix/prefix (`پیشرفتتان`, `کتابهایی`, `می‌کنم`) is a word, not an OCR slip.

### Fixed
- Word order of mixed lines: Tesseract returned `PDF خروجی` for the visual line `خروجی PDF`; a Persian line is now read right to left by word position and a run of Latin words/numbers inside it left to right.
- Number signs: on a right-to-left page `+500` is drawn as `500+` and `100%` as `%100` (OCR returns what it sees); the text order is restored (`fix_trailing_plus`).
- The word-list repair turned the valid word `پیشرفتتان` into the more frequent `پیشرفت‌های`; a comma read as a hamza (`شماء`) becomes `،`.
- Near-invisible shapes (a `#f3f5f8` watermark digit on white, soft shadows; colour distance < 22 from their background) are no longer text.
- Icons read as `۱۰`, `4a`, a lone digit or one/two low-confidence letters next to a Persian line (`2 چغر`, `ه وبلاگ`) are dropped.

### Evaluated, not shipped
- `tessdata_best` `fas`/`eng` models (3.3 MB / 15 MB instead of the packaged 0.4 MB / 4 MB "fast" ones): +2.2 points exact lines and 2x slower on the same pages; the second pass gives more and keeps the packaged models.
- A GPU OCR route: the best open Persian recognisers measured in 2.1.0 (EasyOCR `arabic`, PP-OCRv5 `arabic`) are still worse than Tesseract + repair on exact lines, so nothing waits for a GPU node; the plug-in points (`ocr=`, `ocr_crops`, `PrecomputedBackend`) are unchanged.

### Known limits
- Roughly one line in five on a real page still has a wrong letter or is missing: 13 px light text on dark, stat chips on gradients (`98%`), a few labels inside gradient buttons that the line finder does not find at all.
  The review flags mark the doubtful ones; for exact text give the page's address (DOM text) or an `ocr_lines` list.
- The line finder, not OCR, loses some short labels (they never become a candidate); `bench.line_eval` reports them as "found" < 100%.

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
