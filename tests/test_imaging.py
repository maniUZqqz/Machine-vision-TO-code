import io

import numpy as np
import pytest
from PIL import Image

from mv2code.imaging import ImageError, load_image, box_iou, box_contains, mode_color, hex_of, rgb_of


def test_load_rgba_is_flattened_on_white():
    im = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    rgb, info = load_image(buf.getvalue())
    assert rgb.shape == (40, 40, 3) and rgb.mean() == 255
    assert info["scale"] == 1.0


def test_large_image_is_downscaled():
    big = np.full((300, 5000, 3), 200, np.uint8)
    rgb, info = load_image(big, max_width=2400)
    assert rgb.shape[1] == 2400 and info["scale"] == pytest.approx(0.48)
    assert info["source_size"] == (5000, 300)


def test_dpr_divides_the_size():
    rgb, info = load_image(np.full((200, 400, 3), 9, np.uint8), dpr=2)
    assert rgb.shape[:2] == (100, 200)


@pytest.mark.parametrize("bad", [b"not an image", b"", "does/not/exist.png"])
def test_clear_errors(bad):
    with pytest.raises(ImageError):
        load_image(bad)


def test_too_small():
    with pytest.raises(ImageError):
        load_image(np.zeros((5, 5, 3), np.uint8))


def test_box_helpers():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert box_iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0
    assert box_contains((0, 0, 100, 100), (10, 10, 20, 20))
    assert mode_color(np.array([[1, 2, 3]] * 5 + [[9, 9, 9]], np.uint8)) == (1, 2, 3)
    assert rgb_of(hex_of((12, 34, 56))) == (12, 34, 56)
