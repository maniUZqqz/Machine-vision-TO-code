import numpy as np
from ..models.elements import BoundingBox


def crop_region(image: np.ndarray, bbox: BoundingBox,
                margin: int = 0) -> np.ndarray:
    """Safely crop a region from an image with optional inward margin."""
    h, w = image.shape[:2]
    x1 = max(0, bbox.x + margin)
    y1 = max(0, bbox.y + margin)
    x2 = min(w, bbox.x2 - margin)
    y2 = min(h, bbox.y2 - margin)
    if x2 <= x1 or y2 <= y1:
        return np.array([])
    return image[y1:y2, x1:x2]


def luminance(rgb: tuple) -> float:
    """Perceived luminance (ITU-R BT.709)."""
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def color_variance(crop: np.ndarray) -> float:
    """Mean per-channel standard deviation of pixel colors.
    High values (>45) indicate images/icons; low values (<15) indicate solid fills."""
    if crop.size == 0:
        return 0.0
    pixels = crop.reshape(-1, 3).astype(np.float32)
    return float(pixels.std(axis=0).mean())
