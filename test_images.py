#!/usr/bin/env python
"""
CLI test script: process images from test_samples/ folder and generate reports.

Usage:
    python test_images.py                    # Process all images in test_samples/
    python test_images.py path/to/image.png  # Process a single image
    python test_images.py --json             # Output JSON report (for API consumption)

Output:
    - For each image, generates HTML output in test_samples/output/
    - Prints a summary report with per-element statistics
    - With --json flag, outputs machine-readable JSON to stdout
"""
import os
import sys
import json
import time
import glob
import argparse

# Setup Django before any Django imports
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

import django
django.setup()

import numpy as np
from PIL import Image

from vision_engine.pipeline import create_default_pipeline
from vision_engine.models.page import PageStructure
from vision_engine.models.elements import ElementType, BoundingBox
from code_generator.generator import CodeGenerator


def _rescale_to_original(context):
    """Scale element bboxes from working-image space back to original image space.

    When ImagePreprocessor resizes a large image (>1600px), all element bboxes
    are in the resized coordinate space. The feedback loop renders HTML at the
    ORIGINAL image dimensions, so bboxes must be scaled back to match.
    IMAGE element data URIs are re-cropped from the original image for fidelity.
    """
    import base64
    import cv2 as _cv2

    orig_w = getattr(context, 'original_width', 0)
    orig_h = getattr(context, 'original_height', 0)
    if not orig_w or not orig_h:
        return
    if orig_w == context.width and orig_h == context.height:
        return  # No resize — nothing to scale

    sx = orig_w / context.width
    sy = orig_h / context.height

    orig_img = context.original_image  # Original (unresized) image

    def _make_data_uri(img, bbox):
        """Re-crop IMAGE data URI from original image at scaled coords."""
        x1 = max(0, bbox.x)
        y1 = max(0, bbox.y)
        x2 = min(orig_w, bbox.x2)
        y2 = min(orig_h, bbox.y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = img[y1:y2, x1:x2]
        bgr = _cv2.cvtColor(crop, _cv2.COLOR_RGB2BGR)
        ok, buf = _cv2.imencode('.jpg', bgr, [_cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            return None
        b64 = base64.b64encode(buf.tobytes()).decode('ascii')
        return f'data:image/jpeg;base64,{b64}'

    def _scale(elements):
        for e in elements:
            b = e.bbox
            new_bbox = BoundingBox(
                max(0, int(round(b.x * sx))),
                max(0, int(round(b.y * sy))),
                max(1, int(round(b.width * sx))),
                max(1, int(round(b.height * sy))),
            )
            e.bbox = new_bbox
            # Re-crop data URI from original image at correct (scaled) position
            if (e.element_type.value == 'image'
                    and getattr(e, 'image_data_uri', None)):
                e.image_data_uri = _make_data_uri(orig_img, new_bbox)
            # Re-crop bg_image data URI: was cropped from resized image, must
            # be re-cropped from original at the scaled bbox for fidelity.
            if getattr(e, 'bg_image_data_uri', None):
                e.bg_image_data_uri = _make_data_uri(orig_img, new_bbox)
            _scale(e.children)

    _scale(context.regions)
    _scale(context.text_elements)
    context.width = orig_w
    context.height = orig_h


def process_image(image_path, pipeline, generator):
    """Process a single image and return analysis results."""
    start = time.monotonic()

    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    h, w = img_array.shape[:2]

    # Run vision pipeline
    context = pipeline.run(img_array)

    # Scale element bboxes back to original image space if image was resized
    _rescale_to_original(context)

    # Build page structure (same logic as ConversionService)
    orphan_texts = [t for t in context.text_elements if t.parent_id is None]
    all_root = context.regions + orphan_texts
    root_elements = sorted(all_root, key=lambda e: e.bbox.area, reverse=True)

    # Detect dark top-band: if the original image has a uniform dark strip
    # at y=0 (browser chrome artifact) AND the body background is light,
    # record it so the body gets a matching border-top in the generated CSS.
    # Skip if the page is already dark-themed (body bg lum < 80) — in that
    # case the dark top rows blend naturally with the dark background.
    top_border = None
    bg_r, bg_g, bg_b = context.page_background
    page_bg_lum = 0.2126 * bg_r + 0.7152 * bg_g + 0.0722 * bg_b
    if page_bg_lum >= 80 and img_array.shape[0] > 10:
        top_rows = img_array[:5, :, :]  # First 5 rows
        row_means = top_rows.mean(axis=(1, 2))  # Mean brightness per row
        # Find how many consecutive rows from y=0 are dark (mean < 100)
        band_h = 0
        for rm in row_means:
            if rm < 100:
                band_h += 1
            else:
                break
        if band_h >= 2:
            band_color = tuple(int(c) for c in top_rows[:band_h, :, :].mean(axis=(0, 1)))
            top_border = (band_h, band_color)

    page = PageStructure(
        width=context.width,
        height=context.height,
        background_color=context.page_background,
        root_elements=root_elements,
        text_direction=getattr(context, 'page_direction', 'ltr'),
        top_border=top_border,
    )

    # Generate code
    result = generator.generate(page)

    elapsed_ms = int((time.monotonic() - start) * 1000)

    # Count element types
    type_counts = {}
    for elem in page.walk():
        t = elem.element_type.value
        type_counts[t] = type_counts.get(t, 0) + 1

    # Count layout modes
    layout_counts = {}
    for elem in page.walk():
        layout = getattr(elem, 'layout', None)
        if layout:
            mode = layout.mode.value if hasattr(layout.mode, 'value') else str(layout.mode)
            layout_counts[mode] = layout_counts.get(mode, 0) + 1

    # Count semantic tags
    tag_counts = {}
    for elem in page.walk():
        tag = getattr(elem, 'semantic_tag', 'div') or 'div'
        tag_counts[tag] = tag_counts.get(tag, 0) + 1

    # Flex/grid usage
    has_flex = any(
        getattr(e, 'layout', None) and
        getattr(e.layout, 'mode', None) and
        e.layout.mode.value in ('flex_row', 'flex_column')
        for e in page.walk()
    )
    has_grid = any(
        getattr(e, 'layout', None) and
        getattr(e.layout, 'mode', None) and
        e.layout.mode.value == 'grid'
        for e in page.walk()
    )

    # Image elements with data URIs
    images_with_data = sum(
        1 for e in page.walk()
        if e.element_type == ElementType.IMAGE and getattr(e, 'image_data_uri', None)
    )

    return {
        'file': os.path.basename(image_path),
        'path': image_path,
        'image_size': {'w': w, 'h': h},
        'processing_time_ms': elapsed_ms,
        'total_elements': sum(type_counts.values()),
        'regions': len(context.regions),
        'text_elements': len(context.text_elements),
        'orphan_texts': len(orphan_texts),
        'element_types': type_counts,
        'layout_modes': layout_counts,
        'semantic_tags': tag_counts,
        'background_color': list(context.page_background) if context.page_background else None,
        'has_flex': has_flex,
        'has_grid': has_grid,
        'images_with_data_uri': images_with_data,
        'html': result['combined'],
        'css': result['css'],
    }


def save_output(result, output_dir):
    """Save generated HTML to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(result['file'])[0]
    html_path = os.path.join(output_dir, f'{base_name}_output.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(result['html'])
    return html_path


def print_report(result):
    """Print human-readable report for a single image."""
    print(f"\n{'='*60}")
    print(f"  {result['file']}")
    print(f"{'='*60}")
    print(f"  Image size:     {result['image_size']['w']}x{result['image_size']['h']}")
    print(f"  Processing:     {result['processing_time_ms']}ms")
    print(f"  Total elements: {result['total_elements']}")
    print(f"  Regions:        {result['regions']}")
    print(f"  Text elements:  {result['text_elements']}")
    print(f"  Orphan texts:   {result['orphan_texts']}")

    if result['background_color']:
        r, g, b = result['background_color']
        print(f"  Background:     #{r:02x}{g:02x}{b:02x}")

    print(f"\n  Element Types:")
    for etype, count in sorted(result['element_types'].items()):
        bar = '#' * min(count, 30)
        print(f"    {etype:12s} {count:3d}  {bar}")

    print(f"\n  Layout Modes:")
    for mode, count in sorted(result['layout_modes'].items()):
        print(f"    {mode:15s} {count:3d}")

    print(f"\n  Semantic Tags:")
    for tag, count in sorted(result['tag_counts'].items()) if 'tag_counts' in result else sorted(result['semantic_tags'].items()):
        print(f"    <{tag}>{' '*(12-len(tag))} {count:3d}")

    features = []
    if result['has_flex']:
        features.append('Flexbox')
    if result['has_grid']:
        features.append('Grid')
    if result['images_with_data_uri'] > 0:
        features.append(f"Images({result['images_with_data_uri']})")
    print(f"\n  CSS Features:   {', '.join(features) if features else 'Absolute only'}")

    # Feedback loop results
    if 'similarity' in result:
        sim = result['similarity']
        bar_len = int(sim * 30)
        bar = '#' * bar_len + '-' * (30 - bar_len)
        print(f"\n  Similarity:     {sim:.1%} [{bar}]")
        print(f"  Diff regions:   {result.get('diff_regions_count', 0)}")
        if 'diff_map_path' in result:
            print(f"  Diff map:       {result['diff_map_path']}")


def main():
    parser = argparse.ArgumentParser(
        description='Test vision pipeline on images'
    )
    parser.add_argument(
        'images', nargs='*',
        help='Image paths to process (default: all in test_samples/)'
    )
    parser.add_argument(
        '--json', action='store_true',
        help='Output JSON report to stdout'
    )
    parser.add_argument(
        '--output-dir', default=None,
        help='Output directory (default: test_samples/output/)'
    )
    parser.add_argument(
        '--feedback', action='store_true',
        help='Run feedback loop: render output, compare with input, report similarity'
    )
    args = parser.parse_args()

    # Find images to process
    project_root = os.path.dirname(os.path.abspath(__file__))
    samples_dir = os.path.join(project_root, 'test_samples')

    if args.images:
        image_paths = args.images
    else:
        if not os.path.isdir(samples_dir):
            os.makedirs(samples_dir, exist_ok=True)
            print(f"Created test_samples/ directory.")
            print(f"Put your test images there and run again.")
            print(f"  Path: {samples_dir}")
            return

        extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.webp')
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(samples_dir, ext)))

        if not image_paths:
            print(f"No images found in test_samples/")
            print(f"Put .png/.jpg images there and run again.")
            print(f"  Path: {samples_dir}")
            return

    output_dir = args.output_dir or os.path.join(samples_dir, 'output')

    # Initialize pipeline once (expensive)
    print("Initializing pipeline (loading OCR models)...")
    pipeline = create_default_pipeline()
    generator = CodeGenerator()
    print("Pipeline ready.\n")

    results = []
    for path in sorted(image_paths):
        print(f"Processing: {os.path.basename(path)} ...", end=' ', flush=True)
        try:
            result = process_image(path, pipeline, generator)
            html_path = save_output(result, output_dir)
            result['output_html_path'] = html_path

            # Run feedback loop if requested
            if args.feedback:
                print("comparing...", end=' ', flush=True)
                from vision_engine.feedback_loop import FeedbackLoop
                feedback = FeedbackLoop()
                fb_img = np.array(Image.open(path).convert('RGB'))
                fb_result = feedback.compare_only(fb_img, result['html'])
                result['similarity'] = fb_result.get('similarity', 0.0)
                diff_count = len(fb_result.get('diff_regions', []))
                result['diff_regions_count'] = diff_count

                # Save diff map if available
                if fb_result.get('diff_map') is not None:
                    import cv2
                    diff_path = os.path.join(
                        output_dir,
                        f'{os.path.splitext(result["file"])[0]}_diff.png'
                    )
                    cv2.imwrite(diff_path, fb_result['diff_map'])
                    result['diff_map_path'] = diff_path

            results.append(result)
            print(f"OK ({result['processing_time_ms']}ms)")
        except Exception as e:
            print(f"FAILED: {e}")
            results.append({
                'file': os.path.basename(path),
                'path': path,
                'error': str(e),
            })

    if args.json:
        # JSON output (strip HTML to keep it small)
        for r in results:
            r.pop('html', None)
            r.pop('css', None)
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        # Human-readable reports
        for result in results:
            if 'error' not in result:
                print_report(result)
            else:
                print(f"\n{'='*60}")
                print(f"  {result['file']} — ERROR: {result['error']}")
                print(f"{'='*60}")

        print(f"\n{'='*60}")
        print(f"  Summary: {len([r for r in results if 'error' not in r])}/{len(results)} images processed")
        print(f"  Output directory: {output_dir}")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
