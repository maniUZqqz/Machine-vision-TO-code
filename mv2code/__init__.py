"""mv2code - screenshot to HTML/CSS with classical computer vision."""
__version__ = "2.2.1"

from .api import convert, ocr_crops  # noqa: E402,F401
from .imaging import ImageError  # noqa: E402,F401

__all__ = ["convert", "ocr_crops", "ImageError", "__version__"]
