import numpy as np

from mv2code.imaging import hex_of
from mv2code.pipeline import analyze
from mv2code import segment as sg
from tests.conftest import synthetic_ui


def test_regions_cards_buttons_gradient():
    an = analyze(synthetic_ui())
    fills = {hex_of(r.fill): r for r in an.regions}
    # white card with a 1px border
    card = next(r for r in an.regions if hex_of(r.fill) == "#ffffff")
    assert card.border and len(card.border) >= 3
    assert abs(card.box[0] - 40) <= 2 and abs(card.box[2] - 560) <= 4
    # blue button
    btn = next(r for r in an.regions if hex_of(r.fill) == "#2563eb")
    assert abs(btn.box[2] - 151) <= 3 and abs(btn.box[3] - 45) <= 3
    # gradient strip is one region with a 2-stop gradient
    grads = [r for r in an.regions if r.grad]
    assert grads and abs(grads[0].box[3] - 40) <= 3


def test_page_background_and_threshold():
    img = synthetic_ui(dark=True)
    an = analyze(img)
    assert hex_of(an.bg) == "#121826"
    assert an.threshold == 3


def test_corner_radius_probe():
    import cv2
    img = np.full((120, 200, 3), 255, np.uint8)
    # rounded rectangle r=16 drawn with cv2
    cv2.rectangle(img, (20 + 16, 20), (180 - 16, 100), (30, 30, 30), -1)
    cv2.rectangle(img, (20, 20 + 16), (180, 100 - 16), (30, 30, 30), -1)
    for cx, cy in ((36, 36), (164, 36), (36, 84), (164, 84)):
        cv2.circle(img, (cx, cy), 16, (30, 30, 30), -1)
    r = sg.corner_radius(img, (20, 20, 160, 80), (30, 30, 30))
    assert 10 <= r <= 22


def test_flat_images_have_no_photo():
    assert sg.find_photos(synthetic_ui(), sg.smooth_mask(synthetic_ui())[0]) == []


def test_noise_is_a_photo():
    rng = np.random.RandomState(0)
    img = np.full((300, 300, 3), 240, np.uint8)
    img[60:220, 60:220] = rng.randint(0, 255, (160, 160, 3))
    ph = sg.find_photos(img, sg.smooth_mask(img)[0])
    assert len(ph) == 1 and abs(ph[0].box[2] - 160) <= 6
