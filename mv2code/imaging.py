"""Image loading, colour helpers and box geometry (pure numpy/OpenCV/Pillow)."""
from __future__ import annotations

import io
import os
from typing import Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]  # x, y, w, h

MAX_PIXELS = 40_000_000  # refuse absurd inputs (decompression bombs)


class ImageError(ValueError):
    """The input could not be read as an image (clear message for the user)."""


def load_image(src, max_width: int = 2400, dpr: float | None = None):
    """Load ``src`` (path, bytes, PIL image or ndarray) as RGB uint8.

    Returns ``(rgb, info)``; ``info`` has ``scale`` (processing px per source px) and the
    source size.  Wide images are downscaled (``max_width``); ``dpr`` (e.g. 2 for retina
    screenshots) divides the size so that 1 processed px == 1 CSS px.
    """
    from PIL import Image, ImageFile
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    try:
        if isinstance(src, np.ndarray):
            arr = src
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, -1)
            if arr.shape[2] == 4:
                arr = _flatten_alpha(arr)
            rgb = np.ascontiguousarray(arr[:, :, :3]).astype(np.uint8)
        else:
            if isinstance(src, (bytes, bytearray)):
                im = Image.open(io.BytesIO(bytes(src)))
            elif isinstance(src, (str, os.PathLike)):
                if not os.path.exists(src):
                    raise ImageError(f"file not found: {src}")
                im = Image.open(src)
            else:
                im = src
            im.load()
            if im.mode in ("RGBA", "LA", "P"):
                im = im.convert("RGBA")
                bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
                im = Image.alpha_composite(bg, im)
            rgb = np.array(im.convert("RGB"))
    except ImageError:
        raise
    except Exception as e:  # corrupt / unsupported file
        raise ImageError(f"cannot read the image: {e}") from e
    h, w = rgb.shape[:2]
    if h < 16 or w < 16:
        raise ImageError(f"image too small ({w}x{h}px)")
    if h * w > MAX_PIXELS:
        raise ImageError(f"image too large ({w}x{h}px, limit {MAX_PIXELS // 1_000_000} megapixels)")
    src_size = (w, h)
    scale = 1.0
    if dpr and dpr > 1:
        scale = 1.0 / dpr
    if w * scale > max_width:
        scale = max_width / w
    if abs(scale - 1.0) > 1e-6:
        rgb = cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    return rgb, dict(scale=scale, source_size=src_size, size=(rgb.shape[1], rgb.shape[0]))


def _flatten_alpha(arr: np.ndarray) -> np.ndarray:
    a = arr[:, :, 3:4].astype(np.float32) / 255.0
    return (arr[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)


# --------------------------------------------------------------------------- colour
def hex_of(c) -> str:
    return "#%02x%02x%02x" % tuple(int(max(0, min(255, round(v)))) for v in c[:3])


def rgb_of(h: str):
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def luminance(c) -> float:
    r, g, b = [v / 255.0 for v in c[:3]]
    f = lambda v: v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def cdist(a, b) -> float:
    return float(max(abs(int(a[i]) - int(b[i])) for i in range(3)))


def mode_color(pix: np.ndarray, max_samples: int = 60000):
    """Most frequent colour of an (N,3) pixel array (exact RGB)."""
    pix = pix.reshape(-1, 3)
    if len(pix) > max_samples:
        pix = pix[:: max(1, len(pix) // max_samples)]
    if len(pix) == 0:
        return (255, 255, 255)
    key = (pix[:, 0].astype(np.uint32) << 16) | (pix[:, 1].astype(np.uint32) << 8) | pix[:, 2].astype(np.uint32)
    vals, cnt = np.unique(key, return_counts=True)
    k = int(vals[cnt.argmax()])
    return (k >> 16 & 255, k >> 8 & 255, k & 255)


# --------------------------------------------------------------------------- boxes
def box_area(b: Box) -> int:
    return max(0, b[2]) * max(0, b[3])


def box_x2(b: Box) -> int:
    return b[0] + b[2]


def box_y2(b: Box) -> int:
    return b[1] + b[3]


def box_union(a: Box, b: Box) -> Box:
    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
    x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
    return (x0, y0, x1 - x0, y1 - y0)


def box_inter(a: Box, b: Box) -> int:
    iw = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    ih = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return max(0, iw) * max(0, ih)


def box_iou(a: Box, b: Box) -> float:
    i = box_inter(a, b)
    u = box_area(a) + box_area(b) - i
    return i / u if u > 0 else 0.0


def box_contains(outer: Box, inner: Box, tol: int = 0) -> bool:
    return (outer[0] - tol <= inner[0] and outer[1] - tol <= inner[1]
            and outer[0] + outer[2] + tol >= inner[0] + inner[2]
            and outer[1] + outer[3] + tol >= inner[1] + inner[3])


def box_center(b: Box):
    return (b[0] + b[2] / 2.0, b[1] + b[3] / 2.0)


def center_in(inner: Box, outer: Box, tol: float = 0.0) -> bool:
    cx, cy = box_center(inner)
    return outer[0] - tol <= cx <= outer[0] + outer[2] + tol and outer[1] - tol <= cy <= outer[1] + outer[3] + tol


def clip_box(b: Box, W: int, H: int) -> Box:
    x0, y0 = max(0, b[0]), max(0, b[1])
    x1, y1 = min(W, b[0] + b[2]), min(H, b[1] + b[3])
    return (x0, y0, max(0, x1 - x0), max(0, y1 - y0))


def crop(img: np.ndarray, b: Box, pad: int = 0) -> np.ndarray:
    H, W = img.shape[:2]
    x0, y0 = max(0, b[0] - pad), max(0, b[1] - pad)
    x1, y1 = min(W, b[0] + b[2] + pad), min(H, b[1] + b[3] + pad)
    return img[y0:y1, x0:x1]
