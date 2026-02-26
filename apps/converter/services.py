import os
import sys
import time
import logging
import traceback
import base64

import cv2
import numpy as np
from PIL import Image

from .models import ConversionJob
from vision_engine.pipeline import create_default_pipeline
from vision_engine.models.page import PageStructure
from vision_engine.models.elements import BoundingBox
from code_generator.generator import CodeGenerator

logger = logging.getLogger(__name__)

# Fix Windows encoding for EasyOCR
os.environ['PYTHONIOENCODING'] = 'utf-8'


def _rescale_to_original(context):
    """Scale element bboxes from working-image space back to original image space.

    When ImagePreprocessor resizes a large image (>1600px), all element bboxes
    are in the resized coordinate space. The feedback loop renders HTML at the
    ORIGINAL image dimensions, so bboxes must be scaled back to match.
    IMAGE element data URIs are re-cropped from the original image for fidelity.
    """
    orig_w = getattr(context, 'original_width', 0)
    orig_h = getattr(context, 'original_height', 0)
    if not orig_w or not orig_h:
        return
    if orig_w == context.width and orig_h == context.height:
        return

    sx = orig_w / context.width
    sy = orig_h / context.height
    orig_img = context.original_image

    def _make_data_uri(img, bbox):
        x1 = max(0, bbox.x)
        y1 = max(0, bbox.y)
        x2 = min(orig_w, bbox.x2)
        y2 = min(orig_h, bbox.y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = img[y1:y2, x1:x2]
        bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
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
            if (e.element_type.value == 'image'
                    and getattr(e, 'image_data_uri', None)):
                e.image_data_uri = _make_data_uri(orig_img, new_bbox)
            if getattr(e, 'bg_image_data_uri', None):
                e.bg_image_data_uri = _make_data_uri(orig_img, new_bbox)
            _scale(e.children)

    _scale(context.regions)
    _scale(context.text_elements)
    context.width = orig_w
    context.height = orig_h


# Singleton pipeline (expensive to recreate due to EasyOCR model loading)
_pipeline = None
_generator = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        logger.info("Initializing vision pipeline (first request may be slow)...")
        _pipeline = create_default_pipeline()
    return _pipeline


def _get_generator():
    global _generator
    if _generator is None:
        _generator = CodeGenerator()
    return _generator


class ConversionService:
    """
    Orchestrates the full conversion: image -> vision pipeline -> code generator.
    This is the ONLY place where Django code touches the vision engine.
    """

    def process_job(self, job: ConversionJob) -> None:
        job.status = ConversionJob.Status.PROCESSING
        job.save(update_fields=['status'])

        start_time = time.monotonic()

        try:
            # Load image
            image = Image.open(job.original_image.path).convert('RGB')
            image_array = np.array(image)
            logger.info(f"Processing image: {image_array.shape}")

            # Run vision pipeline
            pipeline = _get_pipeline()
            context = pipeline.run(image_array)
            logger.info(
                f"Pipeline done: {len(context.regions)} regions, "
                f"{len(context.text_elements)} text elements"
            )

            # Scale element bboxes back to original image space if image was resized
            _rescale_to_original(context)

            # Collect orphan text elements (not assigned to any region)
            # These are standalone text like page titles, headings, etc.
            orphan_texts = [
                t for t in context.text_elements if t.parent_id is None
            ]

            # Sort root elements by area DESCENDING so that large background
            # containers get lower z-index than smaller UI elements.
            # This prevents a large mis-detected container from covering smaller
            # correctly-detected elements (e.g. the yellow stat card being hidden
            # behind a wide teal container that spans the whole stats bar).
            all_root = context.regions + orphan_texts
            root_elements = sorted(all_root, key=lambda e: e.bbox.area, reverse=True)
            # Detect dark top-band (browser chrome artifact) on light pages
            top_border = None
            bg_r, bg_g, bg_b = context.page_background
            page_bg_lum = 0.2126 * bg_r + 0.7152 * bg_g + 0.0722 * bg_b
            if page_bg_lum >= 80 and image_array.shape[0] > 10:
                top_rows = image_array[:5, :, :]
                row_means = top_rows.mean(axis=(1, 2))
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

            # Generate HTML/CSS
            generator = _get_generator()
            result = generator.generate(page)

            # Run feedback loop for quality comparison
            similarity = None
            try:
                from vision_engine.feedback_loop import FeedbackLoop
                feedback = FeedbackLoop()
                fb_result = feedback.compare_only(image_array, result['combined'])
                similarity = fb_result.get('similarity')
                logger.info(f"Feedback similarity: {similarity:.1%}")
            except Exception as fb_err:
                logger.warning(f"Feedback loop skipped: {fb_err}")

            # Save results
            job.generated_html = result['html']
            job.generated_css = result['css']
            job.combined_output = result['combined']
            job.status = ConversionJob.Status.COMPLETED
            job.processing_time_ms = int(
                (time.monotonic() - start_time) * 1000
            )
            job.analysis_metadata = {
                'total_regions': sum(1 for _ in page.walk()),
                'text_elements': len(context.text_elements),
                'similarity': similarity,
            }
            job.save()
            logger.info(f"Job {job.id} completed in {job.processing_time_ms}ms")

        except Exception as e:
            logger.error(f"Job {job.id} failed: {e}\n{traceback.format_exc()}")
            job.status = ConversionJob.Status.FAILED
            job.error_message = f"{type(e).__name__}: {e}"
            job.save(update_fields=['status', 'error_message'])
