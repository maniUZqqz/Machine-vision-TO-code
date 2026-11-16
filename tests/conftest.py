"""Shared fixtures.  Slow / optional parts are marked and skipped when the tool is missing."""
import os
import shutil
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _have_browser() -> bool:
    try:
        from mv2code.browser import Browser
        with Browser():
            return True
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    have_tess = shutil.which("tesseract") is not None
    browser_ok = None
    for it in items:
        if "ocr" in it.keywords and not have_tess:
            it.add_marker(pytest.mark.skip(reason="tesseract binary not installed"))
        if "browser" in it.keywords:
            if browser_ok is None:
                browser_ok = _have_browser()
            if not browser_ok:
                it.add_marker(pytest.mark.skip(reason="no playwright / headless browser"))


def synthetic_ui(w: int = 640, h: int = 420, dark: bool = False) -> np.ndarray:
    """A small UI-like picture drawn with numpy: page bg, a bordered card, a filled button, a gradient bar."""
    import cv2
    bg = (18, 24, 38) if dark else (246, 248, 251)
    card = (30, 41, 59) if dark else (255, 255, 255)
    img = np.empty((h, w, 3), np.uint8)
    img[:] = bg
    # card with 1px border and rounded corners (drawn with cv2)
    cv2.rectangle(img, (40, 40), (w - 40, h - 40), (51, 65, 85) if dark else (227, 231, 238), -1)
    cv2.rectangle(img, (41, 41), (w - 41, h - 41), card, -1)
    # button
    cv2.rectangle(img, (70, 260), (220, 304), (37, 99, 235), -1)
    # gradient strip
    for x in range(260, w - 70):
        t = (x - 260) / float(w - 330)
        img[80:120, x] = (int(30 + 200 * t), 90, int(220 - 150 * t))
    return img


@pytest.fixture
def ui_image():
    return synthetic_ui()


@pytest.fixture
def ui_png(tmp_path):
    import cv2
    p = tmp_path / "ui.png"
    cv2.imwrite(str(p), cv2.cvtColor(synthetic_ui(), cv2.COLOR_RGB2BGR))
    return str(p)
