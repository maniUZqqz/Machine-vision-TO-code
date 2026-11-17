# Machine-vision-TO-code (mv2code v2)

[فارسی](README.fa.md) | English

Convert a **UI screenshot into real, editable HTML/CSS** with classical computer vision - no AI API, no GPU.
Persian (RTL) and English (LTR) pages, light and dark, desktop and mobile.

* **Real text** (Tesseract `fas+eng` by default, EasyOCR optional, any engine through `ocr=`), Persian letter/digit/ZWNJ normalisation, word-list repair of dot errors, correct RTL word order.
* **Real layout**: recursive XY-cut -> nested `flex` rows/columns, `<table>`s, full-bleed sections - not absolutely positioned image crops.
* **Real components**: buttons, inputs/selects/textareas, badges, checkboxes/radios/toggles, cards (border, radius, shadow, gradient), dividers, icons traced to inline SVG, photos kept as images.
* Two output modes: `responsive` (semantic tags, flex/table, CSS variables, re-flows at 360/768/1280 px) and `exact` (pixel-faithful, absolute).
* A **component tree as JSON** (`tree_json`, see [docs/tree-json.md](docs/tree-json.md)) for tools that want sections/blocks/text/colours instead of HTML.
* An **honest benchmark** (`python -m bench run`) that this README quotes - before (the v1 code) and after.

## Install

```bash
pip install -e .                       # core: numpy + opencv-headless + Pillow (light)
pip install -e ".[ocr-tesseract]"      # + pytesseract; also: sudo apt install tesseract-ocr tesseract-ocr-fas tesseract-ocr-eng
pip install -e ".[browser]" && python -m playwright install chromium    # feedback loop, --score, benchmark
pip install -e ".[web]"                # Django web app
pip install -e ".[ocr-easyocr]"        # optional, heavy (torch): ~2x more accurate on tiny Persian text
```

If `playwright install` cannot download Chromium (blocked/slow network), point `MV2CODE_CHROME` at an existing Chromium / chrome-headless-shell executable instead.

Pinned, known-good versions: `requirements.txt`. The fonts (Vazirmatn, Inter - SIL OFL 1.1) ship inside the package (`mv2code/fonts`).
Pillow built with `raqm` (the default Linux wheels) gives the most accurate text-width fitting without a browser.

## Use

```bash
mv2code shot.png -o out/ --mode responsive          # out/index.html + fonts/ + tree.json + report.json
mv2code shot.png -o out/ --mode both --score --zip  # also exact mode, honest scores (needs playwright), result.zip
```

```python
from mv2code import convert
r = convert("shot.png", mode="responsive", lang="auto")   # lang: auto | fa | en ; ocr: auto | tesseract | easyocr | none
r["html"], r["css"], r["tree_json"], r["report"]          # r["assets"] = fonts the CSS refers to (fonts/...)
from mv2code.bundle import write_bundle; write_bundle(r, "out/")
```

Web app (upload -> original and generated page side by side with the scores -> ZIP):
`python manage.py migrate && python manage.py runserver`, REST: `POST /api/convert/` (`image`, `mode`, `languages`).

Errors are explicit (`ImageError` for unreadable/too small/huge images, `ValueError` for bad options, exit codes 2/3 in the CLI).
Large images are downscaled (`max_width=2400`), `--dpr 2` handles retina screenshots, output is deterministic.

## How it works

`segment.py` (smooth-pixel region analysis: fills, gradients, borders, radius, shadows, photos, rules) -> `textdet.py` (ink-tight text line boxes)
-> `ocr.py` + `persian.py` (read the crops; Persian-aware clean-up) -> `textstyle.py` (font/weight/size/colour by analysis-by-synthesis with the bundled fonts)
-> `build.py` (roles) -> `layout.py` (XY-cut, tables) -> `emit.py` (HTML/CSS) -> `refine.py` (render, diff per element, correct, keep the best).

## Honest benchmark

The v1 project reported "91.4% / 95.5% pixel similarity". That number counts a pixel as equal when it is within 30 grey levels,
so **a completely blank page in the background colour already scores ~88%**, and a page built from image crops of the original scores
high while being useless as code. The new benchmark (`bench/`) renders 80 pages from its own templates (10 page kinds x fa/en x light/dark x 2 seeds,
Vazirmatn/Inter, Persian digits) plus the 2 real samples, keeps the exact ground truth of every element (role, box, text, style) and measures:

* **text**: share of the ground-truth text that exists as real selectable text, character error rate (Persian and English separately), RTL word order, digit script;
* **structure**: precision/recall/F1 at IoU>=0.5 per role (button, input, image, text block, card, table, nav; plus icon, badge, control);
* **visual**: SSIM and strict pixel similarity of the rendered output vs the screenshot (same size);
* **editability**: share of the page area produced as real HTML/CSS (photos may stay images, everything else may not);
* **responsive**: no horizontal overflow and no overlapping text at 360 / 768 / 1280 px;
* **speed / RAM** on this 2-CPU server.

`python -m bench run --quick` (24 pages) or `--full` (80 pages) prints the table and writes `bench/results.json` (pass `--out` to keep the committed file; `--mode exact` for the pixel-faithful output; `--refine on` to measure the feedback loop; default `--refine off`, the setting of every table below). It needs tesseract, playwright and a headless Chromium.
`python -m bench compare before.json after.json --md` prints the BEFORE/AFTER table below.

### Before vs after (same machine, same pages)

BEFORE is the original v1 code run through the very same benchmark (EasyOCR needs ~2 minutes per page on this server, so v1 was measured on a 6-page subset of the 80 pages plus the 2 real samples; AFTER is shown on the same 6 pages and on all pages).

**Responsive mode (new default) vs v1:**

| metric | BEFORE (old code, 6 pages) | AFTER (same 6 pages) | AFTER (all 80 pages) |
|---|---|---|---|
| Persian text present as real text | 26.3% | 82.0% | 62.4% |
| Persian text also in the right place | 26.3% | 58.0% | 47.0% |
| Persian text error rate (CER) | 64.0% | 13.3% | 31.3% |
| Persian word order correct | 100.0% | 100.0% | 98.1% |
| English text present as real text | 66.0% | 90.5% | 71.9% |
| English text error rate (CER) | 29.0% | 6.9% | 23.8% |
| Structure macro-F1 (7 roles, IoU>=0.5) | 15.5% | 85.1% | 83.2% |
| SSIM vs screenshot | 90.9% | 89.7% | 85.4% |
| Pixel similarity (strict) | 97.9% | 98.7% | 97.3% |
| Editable area (real HTML/CSS) | 89.2% | 100.0% | 99.3% |
| Responsive ok (360/768/1280 px) | 66.7% | 77.8% | 69.2% |
| Seconds per page | 117.8 | 3.4 | 4.4 |
| Peak RAM (MB) | 1853.8 | 213.2 | 220.4 |
| real sample 1122.png: SSIM / editable | 82.7% / 12.8% | 52.2% / 59.3% | |
| real sample 1524.png: SSIM / editable | 86.8% / 74.4% | 92.9% / 99.9% | |

**Exact (pixel-faithful) mode vs v1** (quick set of 24 pages):

| metric | BEFORE (old code, 6 pages) | AFTER (same 6 pages) | AFTER (all 24 pages) |
|---|---|---|---|
| Persian text present as real text | 26.3% | 89.2% | 73.5% |
| Persian text also in the right place | 26.3% | 89.1% | 73.1% |
| Persian text error rate (CER) | 64.0% | 8.0% | 22.1% |
| Persian word order correct | 100.0% | 100.0% | 99.5% |
| English text present as real text | 66.0% | 95.7% | 77.9% |
| English text error rate (CER) | 29.0% | 3.3% | 18.5% |
| Structure macro-F1 (7 roles, IoU>=0.5) | 15.5% | 85.1% | 85.6% |
| SSIM vs screenshot | 90.9% | 94.5% | 93.1% |
| Pixel similarity (strict) | 97.9% | 99.1% | 98.9% |
| Editable area (real HTML/CSS) | 89.2% | 100.0% | 100.0% |
| Responsive ok (360/768/1280 px) | 66.7% | 83.3% | 72.2% |
| Seconds per page | 117.8 | 3.4 | 4.5 |
| Peak RAM (MB) | 1853.8 | 193.1 | 202.6 |
| real sample 1122.png: SSIM / editable | 82.7% / 12.8% | 67.5% / 59.2% | |
| real sample 1524.png: SSIM / editable | 86.8% / 74.4% | 94.0% / 99.9% | |

Reading the tables: (1) v1's *responsive* score comes from shrinking the whole page with `transform: scale()` - nothing re-flows; in v2's *exact* mode the responsive row is meaningless (absolute layout). (2) On the 6-page subset v2-responsive has a slightly lower SSIM than v1 because v1 pastes crops of the screenshot (see editable area); v2-exact is higher than v1 on SSIM while still being real HTML. (3) The all-pages columns are lower than the 6-page ones because the 6 pages are easier than the average (dense Persian dashboards are the hardest).

### The browser feedback loop

`refine="on"` renders the page in a headless browser, compares every text with the pixels it came from, nudges font sizes and positions (<= 4.5 px) and keeps the best round by SSIM. `refine="auto"` (the library/CLI default) turns it on when a browser is installed. **Every other table in this README is measured with `--refine off`** (deterministic and reproducible: re-running `python -m bench run --full --refine off` / `--quick --mode exact --refine off` on the final code reproduced `bench/results.json` / `bench/results-exact-quick.json` number for number). Effect of switching the loop on, same pages, same machine:

| metric (80 pages, responsive) | feedback loop off | feedback loop on |
|---|---|---|
| SSIM vs screenshot | 85.4% | 86.3% |
| SSIM inside text areas | 30.6% | 44.4% |
| Pixel similarity (strict) | 97.3% | 97.5% |
| Persian text present as real text | 62.4% | 64.3% |
| Persian text error rate (CER) | 31.3% | 29.9% |
| English text error rate (CER) | 23.8% | 23.8% |
| Responsive ok (360/768/1280 px) | 69.2% | 65.8% |
| Seconds per page | 4.4 | 7.2 |
| real sample 1122.png: SSIM | 52.2% | 52.2% |
| real sample 1524.png: SSIM | 92.9% | 94.5% |

| metric (24 pages, exact) | feedback loop off | feedback loop on |
|---|---|---|
| SSIM vs screenshot | 93.1% | 95.0% |
| SSIM inside text areas | 31.8% | 60.1% |
| Pixel similarity (strict) | 98.9% | 99.1% |
| Persian text present as real text | 73.5% | 72.7% |
| Persian text error rate (CER) | 22.1% | 22.7% |
| English text error rate (CER) | 18.5% | 18.6% |
| Seconds per page | 4.5 | 7.5 |
| real sample 1122.png: SSIM | 67.5% | 67.5% |
| real sample 1524.png: SSIM | 94.0% | 95.6% |

Reading it: the loop brings the picture closer (more in exact mode: SSIM +1.9 points and text areas 31.8% -> 60.1%) and costs ~3 s per page; it does not improve the text recognition itself (Persian text present in exact mode 73.5% -> 72.7%, CER 22.1% -> 22.7%: within noise, slightly worse). In *responsive* mode the gain is small (SSIM +0.9) and the re-flow check gets slightly **worse** (69.2% -> 65.8% of pages without overflow/overlap): sizes tuned to match the 1280 px screenshot leave less room when the page is narrowed. So: `refine="off"` for responsive output that people will edit, `refine="on"` for exact output. Raw results: `bench/results-refine-full.json`, `bench/results-exact-quick-refine-on.json`.

### OCR engines compared (same crops, measured on this 2-CPU server)

Text lines are found by our own pixel analysis, so every engine reads the *same* crops. Character error rate against the exact ground truth:

| engine | CER | speed / weight | verdict |
|---|---|---|---|
| **Tesseract 4.1 `fas+eng`** (fast models, apt) - default | 0.9% Persian / ~0% English on six mixed pages; **14.1% on three Persian pages full of 11-14 px text** | ~3.5 s/page, no torch, ~200 MB RAM | **default** (best accuracy per resource here) |
| Tesseract with `tessdata_best` fas | 16.4% (same 3 pages) | same | no gain |
| Tesseract + extra pre-processing (heights 34..64 px, Lanczos, Otsu, sharpening, digit whitelist retry ladder) | 14.5-16% | +1-3 s | no gain; the retry ladder is kept only for weak lines |
| Tesseract best-of-6 variants (oracle) / re-ranked by re-rendering the text with Vazirmatn | 9.9% / 14.1% | | re-ranking did not find the better reading |
| **EasyOCR `fa+en`** (CPU, optional extra) | **6.2%** (2 of those 3 pages) | 100+ s/page under load, torch ~1.5 GB | ~2x more accurate on tiny Persian text, too heavy for this server -> optional backend (`ocr="easyocr"`) |
| RapidOCR / PaddleOCR (ONNX) | not evaluated | | no Persian recogniser model was available from sources reachable here; the backend interface (`mv2code/ocr.py`) makes adding one a small change |


### v2.1: Persian text read from a *real* page (before / after)

The synthetic pages above are clean; a real landing page is not (gradient buttons, icons next to labels, 12-14 px Persian). v2.1 is measured on a real
Persian landing page (chagher.ir, screenshot at 1280 px and 390 px) whose text is known exactly from the browser's DOM (`python -m bench.line_eval`,
109 DOM lines, 2 screenshots; DOM text nodes that share one pixel line are compared together, in reading order). Same machine, Tesseract `fas+eng`, no GPU:

| real page, DOM ground truth | v2.0.0 | **v2.1.0** | v2.1.0 + brand word (`glossary=["چغر"]`) |
|---|---|---|---|
| lines read exactly (letters/digits normalised, spaces and half-spaces ignored) | 61.5% | **72.5%** | **75.2%** |
| character error rate | 18.9% | **8.7%** | **8.5%** |
| lines found at all | 89.0% | **93.6%** | 93.6% |
| screenshot -> site-builder draft: DOM lines present as text (1280 / 390 px) | 42/54, 40/55 | **45/54, 42/55** | |
| screenshot -> site-builder draft: invented form blocks | 1 per page | **0** | |

(The brand word fixes `چخر / چفر / چعر` -> `چغر`; on this page the brand name is the only word that no word list knows.)
Synthetic benchmark (80 pages, responsive, `--refine off`, committed `bench/results.json`): Persian text present as real text 62.4% -> **83.3%**,
Persian CER 31.3% -> **14.1%**, English CER 23.8% -> **6.6%**, responsive-ok 69.2% -> 72.5%, structure macro-F1 83.2% -> 82.7%, SSIM unchanged (85.4%), 4.6 s/page, 234 MB.

What changed (all CPU, no model files, no torch):

* **Word-list repair** (`mv2code/lexicon.py`, `mv2code/data/fa_words.tsv.gz`): small Persian text loses its dots (`محفوظ` -> `محقوظ`, `پشتیبانی` -> `پشنیبانی`). A word that is not in the
  list (28k words, with frequencies) is compared letter by letter with the list words of the same length, with a small cost for look-alike letters (ب پ ت ث ن ی, ج چ ح خ, ف ق, ع غ ...) and a full cost for any other,
  and replaced only by a clearly more frequent word that is close. Short words (<= 3 letters), brand words, words repeated 3+ times on the page and rare real words are left alone; the standard half-space is restored
  (`میکند` -> `می‌کند`). Persian punctuation (`،` `؛` `؟`) is restored. `convert(..., glossary=["brand", ...])`: names that are not in any dictionary: never changed, and misread look-alikes become them.
* **Junk words**: low-confidence Latin fragments and icon fragments at the end of a Persian line (`# امکانات`, `als &`, `زج ورود`) are dropped.
* **Long one-line paragraphs** were thrown away as "rules" (a 700 x 24 px line is 29x wider than tall); now they are text.
* **Text inside gradient buttons and pills**: the outline and glow merged with the label (or two buttons were one candidate) and nothing was read; labels are now found again inside such a region against its own local background.
* **External engines**: `convert(..., ocr=<callable or Backend>)`, `PrecomputedBackend` (results read elsewhere are matched to crops by box; missing ones fall back to the local engine) and `ocr_crops(image)` (the crops the converter would read),
  so a GPU service can read the crops and the converter does the rest.

OCR engines on the same real page (same crops, same clean-up):

| engine | exact lines | CER | verdict |
|---|---|---|---|
| Tesseract 4.1 `fas+eng` + v2.1 repair (with the brand word, like the two rows below) | **75.2%** | 8.5% | default |
| Tesseract 4.1, v2.0.0 | 61.5% | 18.9% | |
| EasyOCR `arabic` recogniser exported to ONNX (215 MB, 0.7 s per line on 2 CPUs, same crops, same clean-up) | 52.3% | **6.8%** | lower raw error but drops spaces, reads icons as digits and confuses punctuation; 100x the weight: not worth shipping |
| PP-OCRv5 `arabic_PP-OCRv5_mobile_rec` (ONNX, 8 MB, 0.1 s per line) | 48.6% | 19.3% | short words are excellent (`قیمت`, `سوالات متداول`), long lines lose letters; thin Vazirmatn is not what it was trained on |

### v2.2: a second look at doubtful lines, and honest marks (before / after)

v2.1 left about one real-page line in four with a wrong letter. v2.2 (`python -m bench.line_eval`, same real chagher.ir screenshots + a synthetic Persian page in Tahoma, DOM text as ground truth, Tesseract on CPU,
brand word given) reads the doubtful lines again and marks the ones it is still unsure about:

| | v2.1.1 | **v2.2.0** |
|---|---|---|
| real page (1280 + 390 px, 109 lines): exact / CER / found | 75.2% / 8.6% / 93.6% | **87.2% / 3.7% / 95.4%** |
| Tahoma page (1280 + 390 px, 70 lines): exact / CER / found | 52.8% / 13.7% / 85.7% | **74.3% / 4.6% / 91.4%** |
| time for a 1280 x 2561 px page | ~6 s | ~48 s (the second pass is bounded: `MV2CODE_RESCAN_BUDGET`, default 75 s; `MV2CODE_RESCAN=off` skips it) |

* **Second pass** (`mv2code/rescan.py`): lines with a low confidence, a word the list does not know or nothing found are read again from the original image through five variants of the crop (strokes thickened by erosion, gamma-darkened, other height, Persian-only model);
  the variant the engine trusts most wins (`confidence + agreement + word-list score - width lost`), never by less than a margin, so a reading is only replaced by a better one. Single Latin words/numbers in a Persian line (`PDF`, `100%`) are re-read with the English model.
* **Word order and signs**: a Persian line is read right to left by position (Latin runs inside it left to right), `500+` -> `+500` and `%100` -> `100%` on right-to-left pages, a comma read as a hamza -> `،`, a known stem with a suffix is not "repaired" into another word.
* **Junk**: near-invisible shapes (colour distance < 22 from their background) and icons read as `۱۰`, `4a`, a lone digit or one/two letters at the end of a line are dropped.
* **Review marks** (`mv2code/review.py`): `convert()` returns `report["review"]` = `[{box, text, conf, why}]` (the lines a person should check: low confidence, an unknown Persian word, letters glued to digits, a stray character) and `report["review_lines"]`; text nodes of the tree get `conf` and `"review": true`. DOM text (`ocr_lines`) is never flagged.
  A UI can highlight these lines ("check this line") instead of pretending the text is perfect. **For exact text, pass the page's own text (`ocr_lines`, e.g. from the browser DOM): no OCR engine reads 13 px Persian perfectly.**


### What was wrong with v1 (found by reading and running it)

* Its "pixel similarity" counts a pixel as equal when it is within 30 grey levels: a **blank page scores ~88-94%** on it, so 91.4% / 95.5% said nothing about the quality of the code.
* Complex containers are embedded as **cropped images of the original** (`bg_image_data_uri`), and everything is `position:absolute` (+ a `transform: scale()` hack for narrow screens): high pixel score, useless as editable/responsive code.
* `text_detector.py` contains **corrections hard-coded for the two sample screenshots** (`'سوشمند' -> 'هوشمند'`, `'FLDW' -> 'FLOW'`, `'DOSIN' -> 'DOSING'` ...): they inflate the sample results and do nothing for any other image.
* EasyOCR (torch, ~1.5 GB) on every request: 118 s and 1.8 GB RAM per page on this server; Persian digits/letters, line order and mixed bidi were not handled; only blocks were detected, no tables/navs/inputs/badges (structure F1 15.5%).
* On photographic screenshots (`1122.png`, an AI-rendered SCADA picture) v1 gets a higher SSIM only because it pastes crops of the picture (12.8% editable); v2 keeps the photo areas as images and tries to rebuild the rest as code, and honestly scores lower there (SSIM 52%).


## Known limits (honest)

* **Synthetic benchmark.** The 80 pages come from our own templates; numbers show progress and regressions, not generality. On the two real samples only visual/editability/transcript checks exist (`1524.png` is a real Persian Bootstrap dashboard: SSIM 92.9%, 28 of 32 transcript lines found).
* **OCR is the weak link on tiny Persian text** (11-14 px): v2.1 repairs dots with a word list (real page: 18.9% -> 8.7% character errors, a quarter of the lines still have a wrong letter), so the generated text is real but sometimes wrong; read the first page of text before you publish. Names that are in no dictionary need `glossary=`.
* **Layout drift**: positions are rebuilt from measured gaps; some pages drift 10-30 px vertically (see "placed in its box" in the table) and 20-30% of the narrow-width renders still overflow or overlap. The browser feedback loop nudges text but does not repair structural drift.
* Rows only wrap **below ~93% of the source width**; tables need row rules or a header band to be recognised; repeated cards become flex rows (no CSS grid yet); overlapping elements fall back to small absolute "stacks".
* Controls: checkboxes/radios/toggles are found with recall ~25%; icons ~50% (single-colour ones become inline SVG, multi-colour ones raster crops).
* Only Vazirmatn and Inter are produced; other typefaces (serif, display) are substituted. Letter-spacing, opacity, blur, text-shadow, multi-stop gradients and background patterns are not modelled; box-shadows are approximated as black.
* Photographic or AI-rendered "screenshots" cannot be turned into structure: photo areas stay images and the rest is best effort.
* JPEG-compressed screenshots are handled by an adaptive noise threshold but are **not** part of the benchmark yet; sub-pixel (blurred) 1px borders are only partly recovered.
* Speed: 4-9 s/page on 2 CPUs (Tesseract dominates), ~220 MB RAM; the feedback loop adds a few seconds.


## Tests and CI

```bash
pip install -e ".[dev,web,ocr-tesseract]"     # pytest + Django + pytesseract (+ [browser] and a Chromium for the golden/browser tests)
python -m pytest -q
```

Unit tests per stage, CLI and Django API end-to-end tests, golden pages (need tesseract + a headless browser; skipped otherwise).
`.github/workflows/ci.yml` runs the quick tests on every push and the browser benchmark on a schedule / manually.

## YOLO (optional)

The v2 pipeline does not need it. The v1 detector code is kept and stays off without `ultralytics`; train on a GPU machine only: [docs/yolo.md](docs/yolo.md).

## Licence

This repository has **no licence file** yet - that is the owner's decision. Bundled fonts keep their own SIL OFL 1.1 licences (`mv2code/fonts/LICENSE-*.txt`);
the benchmark pages are generated by this repository's own code (no third-party screenshots).
The Persian word list `mv2code/data/fa_words.tsv.gz` is derived from the `wordfreq` data and keeps its **CC BY-SA 4.0** licence (`mv2code/data/NOTICE.md`); it is a separate data file that the code only reads.
