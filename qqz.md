# Machine Vision: Screenshot to HTML/CSS Converter

A pure computer-vision system that converts UI screenshots into pixel-accurate HTML/CSS — no AI APIs needed for the core pipeline. Built with Django, OpenCV, NumPy, and optional YOLO enhancement.

**Current benchmark:**
- `1122.png` (dark industrial SCADA UI): **91.4%** pixel similarity
- `1524.png` (Persian scheduling dashboard): **95.5%** pixel similarity

---

## Quick Start — Copy & Paste

Clone, install, and run in one go:

```bash
git clone https://github.com/maniUZqqz/Machine-vision-TO-code.git
cd Machine-vision-TO-code
pip install -r requirements.txt
echo "DEBUG=True" > .env
echo "SECRET_KEY=some-random-key" >> .env
echo "DJANGO_SETTINGS_MODULE=config.settings" >> .env
python manage.py migrate
python test_images.py test_samples/input/1524.png --feedback
```

Optional extras:

```bash
# Feedback loop (Playwright — pixel comparison with original)
pip install playwright
playwright install chromium

# YOLO enhancement (optional, works without it)
pip install ultralytics

# Run Django web UI
python manage.py runserver
# Open http://localhost:8000/
```

---

## Detailed Setup

### 1. Install Dependencies

```bash
# Python 3.9+ required
pip install -r requirements.txt

# For feedback loop (optional but recommended):
pip install playwright
playwright install chromium

# For YOLO enhancement (optional):
pip install ultralytics
```

### 2. Initialize Django

```bash
# Set environment
set DJANGO_SETTINGS_MODULE=config.settings    # Windows
export DJANGO_SETTINGS_MODULE=config.settings  # Linux/Mac

# Run migrations
python manage.py migrate
```

### 3. Run — CLI Mode

```bash
# Process all images in test_samples/input/
python test_images.py

# Process a single image
python test_images.py path/to/screenshot.png

# With quality feedback (requires Playwright)
python test_images.py path/to/screenshot.png --feedback

# JSON output (for scripts/API)
python test_images.py --json
```

Output HTML files are saved to `test_samples/output/`.

### 4. Run — Web UI

```bash
python manage.py runserver
# Open http://localhost:8000/
```

Upload a screenshot, get HTML/CSS back. Debug panel available at `/debug/<job_id>/`.

### 5. Run — REST API

```bash
# Convert an image
curl -X POST http://localhost:8000/api/convert/ \
  -F "image=@screenshot.png" \
  -F "languages=en,fa"

# Batch test all images in test_samples/
curl http://localhost:8000/api/test/
```

---

## How It Works

The system runs a **9-step computer vision pipeline** on each screenshot:

```
Input Image (PNG/JPG)
        |
   [1] ImagePreprocessor     — Resize to <=1600px, normalize
        |
   [2] ContourDetector        — 10 classical CV strategies (edges, color,
        |                       thresholds, hue-diversity, sidebars, etc.)
        |
  [2.5] YOLODetector          — (Optional) 11-class UI element detection
        |                       Merges with classical results via IoU
        |
   [3] TextDetector           — EasyOCR (Persian + English), CLAHE enhancement
        |
   [4] ColorAnalyzer          — K-means clustering, gradient detection
        |
   [5] TypographyAnalyzer     — Font size, weight, alignment estimation
        |
   [6] SpacingAnalyzer        — Padding/margin calculation
        |
   [7] LayoutAnalyzer         — Flex/grid/block layout detection
        |
   [8] HierarchyBuilder       — Tree assembly, child grouping, pruning
        |
   [9] ElementClassifier      — Type assignment (BUTTON, INPUT, IMAGE, etc.)
        |                       Data URI embedding, icon matching
        |
   CodeGenerator              — HTMLBuilder + CSSBuilder → final HTML document
        |
   FeedbackLoop (optional)    — Playwright render → pixel comparison
        |
   Output HTML/CSS
```

---

## Project Structure

```
Machine vision/
├── apps/converter/              # Django web app
│   ├── models.py                #   ConversionJob database model
│   ├── views.py                 #   Upload, preview, debug, API views
│   ├── services.py              #   Pipeline orchestration service
│   ├── urls.py                  #   URL routing
│   └── templates/converter/     #   HTML templates for web UI
│
├── code_generator/              # HTML/CSS output generation
│   ├── generator.py             #   Main orchestrator (CSS vars, CDN, assembly)
│   ├── html_builder.py          #   Semantic HTML rendering
│   ├── css_builder.py           #   CSS rules (position, color, layout)
│   └── rtl_handler.py           #   RTL/LTR text direction detection
│
├── vision_engine/               # Core computer vision pipeline
│   ├── pipeline.py              #   Pipeline orchestrator (9 steps)
│   ├── feedback_loop.py         #   Playwright-based quality comparison
│   ├── preprocessing/
│   │   └── image_loader.py      #   Step 1: Load, resize, normalize
│   ├── detection/
│   │   ├── contour_detector.py  #   Step 2: 10-strategy region detection
│   │   ├── yolo_detector.py     #   Step 2.5: YOLO UI element detection
│   │   └── text_detector.py     #   Step 3: OCR (EasyOCR)
│   ├── analysis/
│   │   ├── color_analyzer.py    #   Step 4: Color extraction (k-means)
│   │   ├── typography_analyzer.py # Step 5: Font analysis
│   │   ├── spacing_analyzer.py  #   Step 6: Padding/margin
│   │   ├── layout_analyzer.py   #   Step 7: Flex/grid detection
│   │   └── hierarchy_builder.py #   Step 8: Tree building
│   ├── classification/
│   │   ├── element_classifier.py #  Step 9: Semantic classification
│   │   └── icon_matcher.py      #   Bootstrap Icons CDN matching
│   ├── models/
│   │   ├── elements.py          #   BoundingBox, ColorInfo, DetectedElement
│   │   ├── layout.py            #   LayoutMode, LayoutInfo
│   │   └── page.py              #   PageStructure
│   └── utils/
│       ├── geometry.py          #   NMS, box merging, containment
│       └── image_utils.py       #   Crop, luminance, variance helpers
│
├── ml_models/yolo_ui/           # YOLO model training
│   ├── train.py                 #   Training script
│   ├── synthetic_generator.py   #   Generate synthetic training data
│   ├── auto_labeler.py          #   Auto-label images using pipeline
│   ├── self_trainer.py          #   Pseudo-labeling for self-training
│   └── dataset/                 #   Training images & labels
│
├── config/                      # Django configuration
│   ├── settings/base.py         #   Main settings
│   └── urls.py                  #   Root URL router
│
├── test_samples/                # Test data
│   ├── input/                   #   Input screenshots
│   └── output/                  #   Generated HTML (gitignored)
│
├── test_images.py               # CLI entry point
├── manage.py                    # Django management
├── requirements.txt             # Python dependencies
└── .env                         # Environment variables (gitignored)
```

---

## Development Guide

### Which files to edit for what

| Goal | Files |
|------|-------|
| **Improve element detection** | `vision_engine/detection/contour_detector.py` (add/tune strategies) |
| **Improve text recognition** | `vision_engine/detection/text_detector.py` (OCR params, post-processing) |
| **Fix color extraction** | `vision_engine/analysis/color_analyzer.py` (k-means, gradient, border) |
| **Fix layout detection** | `vision_engine/analysis/layout_analyzer.py` (flex/grid/block rules) |
| **Fix element classification** | `vision_engine/classification/element_classifier.py` (type rules, thresholds) |
| **Fix HTML output** | `code_generator/html_builder.py` (tags, attributes, structure) |
| **Fix CSS output** | `code_generator/css_builder.py` (position, color, border, layout props) |
| **Fix tree nesting** | `vision_engine/analysis/hierarchy_builder.py` (parent-child assignment) |
| **Train YOLO model** | `ml_models/yolo_ui/train.py` (training params, data augmentation) |
| **Add new pipeline step** | `vision_engine/pipeline.py` (register new step in `create_default_pipeline()`) |
| **Change web UI** | `apps/converter/views.py` + `apps/converter/templates/` |
| **Change API** | `apps/converter/views.py` (APIConvertView) + `apps/converter/urls.py` |

### How to develop and test

```bash
# 1. Make your code change

# 2. Test on a single image (fast iteration)
python test_images.py test_samples/input/1524.png --feedback

# 3. Check the output HTML
#    → opens test_samples/output/1524_output.html in browser

# 4. Check the diff map (shows red areas where output differs from original)
#    → test_samples/output/1524_diff.png

# 5. Test on all images
python test_images.py --feedback
```

### Key design decisions

- **All-absolute positioning**: Every element uses `position: absolute` with pixel coordinates from the original screenshot. This gives pixel-accurate placement but means no responsive reflow.
- **JPEG data URIs**: Complex visual regions (photos, gauges, charts) are embedded as JPEG base64 data URIs rather than trying to recreate them in CSS.
- **No AI APIs**: The core pipeline is pure OpenCV + NumPy. YOLO is optional and trained locally. EasyOCR runs locally. No cloud API calls.
- **Feedback loop**: Playwright renders the generated HTML and compares it pixel-by-pixel with the original screenshot. This gives an objective quality score.

### Important thresholds and constants

| What | Where | Value | Notes |
|------|-------|-------|-------|
| Max image dimension | `image_loader.py` | 1600px | Larger images are resized to prevent OOM |
| OCR confidence | `text_detector.py` | 0.35 | Lower = more text detected (but more noise) |
| NMS IoU threshold | `geometry.py` | 0.5 | Higher = less aggressive deduplication |
| YOLO confidence | `yolo_detector.py` | 0.3 | Minimum confidence for YOLO detections |
| JPEG quality | `element_classifier.py` | 95 | For embedded data URIs |
| K-means clusters | `color_analyzer.py` | 5 | Number of color clusters per element |
| Icon max size | `element_classifier.py` | 60x60 | Larger = IMAGE, smaller = ICON |
| bg_image lum_std | `element_classifier.py` | 40 | Threshold for embedding bg as JPEG |
| Feedback similarity threshold | `feedback_loop.py` | 30 gray levels | Pixel tolerance for "matching" |

---

## YOLO Training

The optional YOLO detector recognizes 11 UI element classes:
`button`, `input`, `select`, `card`, `image`, `icon`, `text_block`, `header_bar`, `sidebar`, `navbar`, `separator`

### Train from scratch

```bash
# Generate synthetic data + train
python ml_models/yolo_ui/train.py --epochs 50

# Or use Django management command
python manage.py train_yolo --epochs 50
```

### Self-training (continuous improvement)

```bash
# Uses pipeline output as pseudo-labels
python manage.py train_yolo --self-train
```

### Hardware notes

- CPU-only training works but needs `batch=2, imgsz=416, workers=0` on 8GB RAM
- GPU recommended for batch=8+ or imgsz=640+
- Current model: YOLO11n (2.6M params, 6.5 GFLOPs)
- Model saved at: `ml_models/yolo_ui/best_ui_detector.pt`

---

## Known Issues & Limitations

1. **Large images (>1600px)**: Resized internally; coordinates are scaled back, but some precision is lost.
2. **Complex overlapping elements**: The contour detector can merge adjacent elements if they share similar colors.
3. **Dynamic content**: Animations, hover states, dropdowns are not captured — only the static screenshot.
4. **Font matching**: Font families are guessed (sans-serif/serif), not detected precisely.
5. **JPEG artifacts**: Embedded JPEG data URIs have compression artifacts at element boundaries.
6. **EasyOCR memory**: Can crash on very large images (mitigated by 1600px resize limit).

---

## Roadmap

### Short-term (next improvements)

- [ ] **Image overflow clipping**: Some `<img>` elements render larger than their CSS dimensions. Need `object-fit: cover` + proper clipping or re-cropping data URIs to exact bbox dimensions.
- [ ] **Sidebar gradient matching**: The detected sidebar gradient doesn't always match the original (especially around logo areas). Need per-segment gradient detection.
- [ ] **PNG encoding for high-detail regions**: Switch from JPEG to PNG for small, high-frequency regions (gauges, charts) to eliminate compression artifacts.
- [ ] **Font detection via OCR metrics**: Use character-level bounding boxes to estimate serif vs sans-serif.
- [ ] **Multi-page support**: Handle scrollable pages by stitching multiple screenshots.

### Medium-term

- [ ] **Responsive output**: Generate flexible CSS (%, vw, vh) instead of fixed pixels for mobile-friendly output.
- [ ] **Component library matching**: Detect Bootstrap/Tailwind/Material Design components and generate framework-specific code.
- [ ] **Interactive elements**: Detect and generate working form elements (dropdowns, checkboxes, radio buttons with proper behavior).
- [ ] **CSS Grid improvements**: Better grid detection for complex table-like layouts.
- [ ] **Batch processing API**: Queue-based processing for multiple screenshots.

### Long-term

- [ ] **Real-time preview**: WebSocket-based live preview as the pipeline processes.
- [ ] **Plugin system**: Allow custom pipeline steps to be added without modifying core code.
- [ ] **Export formats**: React/Vue/Angular component generation.
- [ ] **Figma/Sketch import**: Direct design file import instead of screenshots.
- [ ] **Accessibility compliance**: Generate ARIA labels, proper heading hierarchy, form labels.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Django 4.2+ |
| Vision | OpenCV, NumPy |
| OCR | EasyOCR (local, multi-language) |
| Object Detection | YOLO11 via Ultralytics (optional) |
| Quality Testing | Playwright (headless Chromium) |
| Database | SQLite |
| Frontend | Vanilla HTML/CSS/JS |
| Icons | Bootstrap Icons CDN |
| Fonts | Google Fonts (Vazirmatn for Persian) |

---

## Environment Variables (`.env`)

```env
DEBUG=True
SECRET_KEY=your-secret-key
ALLOWED_HOSTS=localhost,127.0.0.1
DJANGO_SETTINGS_MODULE=config.settings
```

---

## API Reference

### POST `/api/convert/`

Upload a screenshot and get HTML/CSS back.

**Request:**
```
Content-Type: multipart/form-data
- image: file (PNG/JPG)
- languages: string (comma-separated, e.g. "en,fa")
```

**Response:**
```json
{
  "job_id": "uuid",
  "html": "<!DOCTYPE html>...",
  "css": "body { ... }",
  "similarity": 0.94,
  "processing_time_ms": 15000
}
```

### GET `/api/test/`

Batch-process all images in `test_samples/input/`.

**Response:**
```json
{
  "results": [
    {
      "file": "1524.png",
      "similarity": 0.955,
      "total_elements": 104,
      "processing_time_ms": 40000
    }
  ]
}
```

---

## License

This project is under development. License TBD.
