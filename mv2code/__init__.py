"""mv2code - screenshot to HTML/CSS with classical computer vision."""
__version__ = "2.0.0"

from .api import convert  # noqa: E402,F401
from .imaging import ImageError  # noqa: E402,F401

__all__ = ["convert", "ImageError", "__version__"]
