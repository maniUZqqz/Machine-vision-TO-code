"""Glue between the Django app and the ``mv2code`` package.

The web app is a thin shell: it stores the upload, calls ``mv2code.convert`` and keeps the
result (HTML, component tree, report and - when a headless browser is available - honest
scores) on the job.  The original v1 engine (``vision_engine`` + ``code_generator``) is still
importable for the old debug views; set ``MV2CODE_ENGINE=legacy`` to convert with it.
"""
import io
import logging
import os
import time
import traceback

import numpy as np
from PIL import Image

from .models import ConversionJob

logger = logging.getLogger(__name__)

os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

# --------------------------------------------------------------------------- legacy (v1) bits
_pipeline = None


def _get_pipeline():
    """The v1 vision pipeline (EasyOCR based). Only used by the legacy debug views."""
    global _pipeline
    if _pipeline is None:
        from vision_engine.pipeline import create_default_pipeline
        _pipeline = create_default_pipeline()
    return _pipeline


def engine_name() -> str:
    return os.environ.get('MV2CODE_ENGINE', 'mv2code').strip().lower()


# --------------------------------------------------------------------------- scoring helper
def _score(rgb, result):
    """Honest scores (SSIM, pixel similarity, editable share) when a headless browser exists."""
    try:
        from mv2code.browser import Browser
        from mv2code.score import compare
    except Exception as e:  # playwright not installed
        return {'unavailable': f'{type(e).__name__}: {e}'}
    try:
        photos = [n['box'] for n in _walk(result['tree_json']) if n.get('role') == 'image']
        with Browser() as br:
            return compare(rgb, result['html'], result['assets'], br, photo_boxes=photos)
    except Exception as e:
        return {'unavailable': f'{type(e).__name__}: {e}'}


def _walk(node):
    yield node
    for c in node.get('children', []) or []:
        yield from _walk(c)


class ConversionService:
    """Orchestrates one conversion job."""

    def process_job(self, job: ConversionJob) -> None:
        job.status = ConversionJob.Status.PROCESSING
        job.save(update_fields=['status'])
        start = time.monotonic()
        try:
            if engine_name() == 'legacy':
                self._process_legacy(job)
            else:
                self._process_v2(job)
            job.processing_time_ms = int((time.monotonic() - start) * 1000)
            job.status = ConversionJob.Status.COMPLETED
            job.save()
            logger.info('Job %s completed in %sms', job.id, job.processing_time_ms)
        except Exception as e:
            logger.error('Job %s failed: %s\n%s', job.id, e, traceback.format_exc())
            job.status = ConversionJob.Status.FAILED
            job.error_message = f'{type(e).__name__}: {e}'
            job.save(update_fields=['status', 'error_message'])

    # ---- v2
    def _process_v2(self, job: ConversionJob) -> None:
        import mv2code
        from mv2code.imaging import load_image
        langs = list(job.languages or [])
        lang = 'auto'
        if langs == ['fa']:
            lang = 'fa'
        elif langs == ['en']:
            lang = 'en'
        mode = getattr(job, 'mode', 'responsive') or 'responsive'
        rgb, _ = load_image(job.original_image.path)
        res = mv2code.convert(job.original_image.path, mode=mode, lang=lang, refine='off')
        scores = _score(rgb, res)
        job.generated_html = res['html']
        job.generated_css = res['css']
        job.combined_output = res['html']
        rep = res['report']
        job.analysis_metadata = {
            'engine': 'mv2code',
            'mode': mode,
            'total_regions': rep['regions'],
            'text_elements': rep['text_lines'],
            'nodes': rep['nodes'],
            'ocr': rep['ocr'],
            'direction': rep['page_direction'],
            'scores': scores,
            'similarity': scores.get('ssim') if isinstance(scores, dict) else None,
            'tree': res['tree_json'],
            'seconds': rep['seconds'],
        }

    # ---- v1 (kept for comparison)
    def _process_legacy(self, job: ConversionJob) -> None:
        from code_generator.generator import CodeGenerator
        from vision_engine.models.page import PageStructure
        image = Image.open(job.original_image.path).convert('RGB')
        arr = np.array(image)
        context = _get_pipeline().run(arr)
        orphan = [t for t in context.text_elements if t.parent_id is None]
        roots = sorted(context.regions + orphan, key=lambda e: e.bbox.area, reverse=True)
        page = PageStructure(width=context.width, height=context.height, background_color=context.page_background,
                             root_elements=roots, text_direction=getattr(context, 'page_direction', 'ltr'))
        out = CodeGenerator().generate(page)
        job.generated_html = out['html']
        job.generated_css = out['css']
        job.combined_output = out['combined']
        job.analysis_metadata = {'engine': 'legacy', 'total_regions': sum(1 for _ in page.walk()),
                                 'text_elements': len(context.text_elements)}


def build_zip(job: ConversionJob) -> bytes:
    """ZIP with index.html, fonts/, tree.json, report.json and original.png for a finished job."""
    from mv2code.bundle import zip_bytes
    from mv2code.emit import font_assets
    meta = job.analysis_metadata or {}
    result = {
        'html': job.combined_output,
        'tree_json': meta.get('tree', {}),
        'report': {k: v for k, v in meta.items() if k != 'tree'},
        'assets': [dict(path=a['path'], source=a['source']) for a in font_assets()],
    }
    buf = io.BytesIO()
    Image.open(job.original_image.path).convert('RGB').save(buf, 'PNG')
    return zip_bytes(result, 'index', buf.getvalue())
