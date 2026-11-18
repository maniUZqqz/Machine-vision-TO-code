"""v2.3: smooth gradient bands (hero, CTA) are CSS backgrounds, never one pasted picture.

The pictures are drawn with numpy only (no browser): a header, a full-bleed gradient hero whose end colour touches the page background (the
case that made 2.2.x paste the whole page as ONE image), a few cards and a gradient call-to-action band.
"""
import numpy as np
import pytest

from mv2code import gradfit as gf
from mv2code import segment as sg
from mv2code.emit import css_gradient
from mv2code.imaging import hex_of
from mv2code.pipeline import analyze

W, H = 960, 720


def _grad(w, h, c1, c2, angle=180, stops=None):
    """CSS-style linear gradient as an array (angle: 0 = to top, 90 = to right, 180 = to bottom, 135 = to bottom right)."""
    a = np.radians(angle)
    dx, dy = np.sin(a), -np.cos(a)
    L = abs(w * dx) + abs(h * dy)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    t = ((xx + 0.5 - w / 2) * dx + (yy + 0.5 - h / 2) * dy) / L + 0.5
    t = np.clip(t, 0, 1)[..., None]
    if stops is None:
        stops = [(0.0, c1), (1.0, c2)]
    ts = np.array([s[0] for s in stops])
    out = np.zeros((h, w, 3))
    for k in range(3):
        out[..., k] = np.interp(t[..., 0], ts, [s[1][k] for s in stops])
    return np.round(out).astype(np.uint8)


def page(hero=(20, 38, 77), hero_end=(11, 18, 32), body=(11, 18, 32), angle=180, stops=None, hero_h=300):
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = body
    img[0:56] = (16, 26, 48)                                   # header
    img[56:56 + hero_h] = _grad(W, hero_h, hero, hero_end, angle, stops)
    for k in range(3):                                         # three flat cards below the hero
        x0 = 40 + k * 300
        img[400:520, x0:x0 + 280] = (22, 33, 58)
    img[560:680, 40:W - 40] = _grad(W - 80, 120, (109, 40, 217), (37, 99, 235), 135)   # CTA band
    return img


def _editable_share(an):
    """Share of the page that is NOT a pasted picture (the converter's own photo list)."""
    area = sum(p.box[2] * p.box[3] for p in an.photos)
    return 1.0 - area / float(W * H)


@pytest.mark.parametrize("angle", [180, 135, 90])
def test_hero_whose_end_colour_equals_the_page_background_is_a_gradient_region_not_a_picture(angle):
    an = analyze(page(angle=angle))
    assert an.photos == [], "the page must not be pasted as a picture"
    assert len(an.regions) >= 4
    heroes = [r for r in an.regions if r.grad and r.box[2] >= 0.95 * W and r.box[3] >= 250]
    assert heroes, [(r.box, bool(r.grad)) for r in an.regions]
    hero = heroes[0]
    assert abs(hero.box[1] - 56) <= 3 and abs(hero.box[3] - 300) <= 16        # the last rows are within 3 levels of the page colour: they stay flat page
    # the CTA band is a gradient region too
    assert any(r.grad and 100 <= r.box[3] <= 130 and r.box[2] >= 800 for r in an.regions)


def test_the_header_keeps_its_own_colour_and_does_not_leak_into_the_gradient():
    an = analyze(page(angle=135))
    header = [r for r in an.regions if hex_of(r.fill) == "#101a30"]
    assert header and header[0].box[1] <= 1 and 54 <= header[0].box[3] <= 60
    hero = max((r for r in an.regions if r.grad and r.box[3] >= 250), key=lambda r: r.box[2] * r.box[3])
    assert hero.box[1] >= 54


def test_a_light_gradient_band_that_ends_in_white_is_still_a_region():
    img = np.full((H, W, 3), 255, np.uint8)
    img[0:56] = (16, 26, 48)
    img[56:356] = _grad(W, 300, (224, 231, 255), (255, 255, 255), 180)
    img[420:540, 40:300] = (248, 250, 252)
    an = analyze(img)
    assert an.photos == []
    assert any(r.grad and r.box[3] >= 250 for r in an.regions)


def test_three_stop_linear_gradient_is_fitted_with_its_middle_stop():
    img = np.full((300, 800, 3), 255, np.uint8)
    stops = [(0.0, (13, 148, 136)), (0.55, (15, 118, 110)), (1.0, (17, 94, 89))]
    img[:] = _grad(800, 300, None, None, 120, stops)
    S, T = sg.smooth_mask(img)
    g = gf.fit_general(img, S)
    assert g is not None and g["kind"] == "linear"
    assert g["angle"] % 180 in range(112, 129)                  # 120 deg, or 300 deg with the colours swapped: the same picture
    css = css_gradient(g)
    assert css.startswith("linear-gradient(") and css.count("%") >= 3


def test_radial_gradient_is_fitted_from_pixels():
    h, w = 320, 640
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cx, cy = 0.7 * w, 0.3 * h
    r = np.hypot(xx - cx, yy - cy)
    rfc = max(np.hypot(px - cx, py - cy) for px, py in ((0, 0), (w, 0), (0, h), (w, h)))
    t = np.clip(r / (0.7 * rfc), 0, 1)[..., None]
    c1, c2 = np.array([59, 130, 246.0]), np.array([11, 18, 32.0])
    img = np.round(c1 * (1 - t) + c2 * t).astype(np.uint8)
    m = np.ones((h, w), bool)
    g = gf.fit_radial(img, m)
    assert g is not None and g["kind"] == "radial"
    assert abs(g["cx"] - 0.7) < 0.12 and abs(g["cy"] - 0.3) < 0.2
    assert css_gradient(g).startswith("radial-gradient(circle farthest-corner at ")


def test_planes_split_a_flat_header_from_a_gradient_that_passes_through_its_colour():
    h, w = 400, 600
    img = _grad(w, h, (30, 60, 120), (10, 20, 40), 180)
    img[0:60] = (20, 40, 80)                                     # a flat header; the gradient crosses (20,40,80) further down
    parts = gf.split_planes(img, np.ones((h, w), bool))
    assert parts is not None and len(parts) >= 2
    flat = [p for p in parts if p["flat"]]
    assert flat and int(flat[0]["mask"][:60].sum()) >= 0.9 * 60 * w


def test_css_gradient_two_stop_and_unknown_keys_are_backward_compatible():
    old = dict(angle=135, c1=(109, 40, 217), c2=(37, 99, 235))
    assert css_gradient(old) == "linear-gradient(135deg,#6d28d9,#2563eb)"


def test_a_noisy_photo_is_still_a_photo_and_a_flat_page_still_has_no_gradient():
    rng = np.random.RandomState(1)
    img = np.full((400, 600, 3), 245, np.uint8)
    img[80:320, 100:500] = rng.randint(0, 255, (240, 400, 3))
    an = analyze(img)
    assert len(an.photos) == 1
    flat = analyze(np.full((300, 500, 3), 240, np.uint8))
    assert flat.photos == [] and not any(r.grad for r in flat.regions)


def test_tree_json_carries_stops_for_a_multi_stop_gradient():
    from mv2code.model import Node
    n = Node(role="box", box=(0, 0, 100, 50), grad=dict(kind="linear", angle=120, c1=(1, 2, 3), c2=(9, 9, 9),
                                                        stops=[(0.0, (1, 2, 3)), (0.5, (5, 5, 5)), (1.0, (9, 9, 9))]))
    d = n.to_json()["gradient"]
    assert d["angle"] == 120 and d["from"] == "#010203" and d["to"] == "#090909"
    assert [s["at"] for s in d["stops"]] == [0.0, 0.5, 1.0]


# ----------------------------------------------------------------------------- the generated hero pages of the benchmark, through a real browser
@pytest.mark.browser
@pytest.mark.parametrize("variant,lang", [(2, "fa"), (0, "en"), (7, "fa")])
def test_hero_pages_become_css_bands_with_real_text_in_url_mode(tmp_path, variant, lang):
    """URL mode (text lines read from the page itself) on the benchmark's full-bleed gradient hero pages: no picture, the title is real HTML."""
    from bench.render import Browser, ensure_fonts
    from bench.templates import HERO_VARIANTS, make_page
    from mv2code import convert
    pg = make_page("hero", lang, "dark" if HERO_VARIANTS[variant][2] else "light", variant)
    ensure_fonts(str(tmp_path))
    f = tmp_path / "p.html"
    f.write_text(pg["html"], encoding="utf-8")
    with Browser() as br:
        img, probe = br.render(f.as_uri(), pg["width"], probe=True)
    lines = []
    for t in probe["texts"]:
        if t.get("occluded"):
            continue
        for ln in t.get("lines") or [{"text": t["text"], "box": t["box"]}]:
            lines.append({"box": [float(v) for v in ln["box"]], "text": " ".join(ln["text"].split()), "conf": 1.0})
    res = convert(img, mode="responsive", lang=lang, ocr_lines=lines, refine="off")
    assert "-gradient(" in res["html"]
    assert "<img" not in res["html"]
    assert res["report"]["regions"] >= 5
    title = max((t for t in probe["texts"] if t["text"]), key=lambda t: t["fs"])["text"]       # the h1
    assert title.replace(" ", "") in res["html"].replace(" ", "").replace("&nbsp;", "")
