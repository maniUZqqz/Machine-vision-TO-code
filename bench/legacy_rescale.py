"""Verbatim copy of ``_rescale_to_original`` from the v1 ``apps/converter/services.py``.

The v1 service was replaced by the mv2code glue; the benchmark's ``legacy`` engine still needs this
helper to reproduce exactly what the v1 web app did (scale boxes back to the original size and
re-crop embedded image data URIs).
"""
import base64

import cv2

from vision_engine.models.elements import BoundingBox


def _rescale_to_original(context):
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
