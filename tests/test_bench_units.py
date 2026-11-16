"""The benchmark's own code is tested too: generator determinism and metric maths."""
import numpy as np
import pytest

pytest.importorskip("rapidfuzz")

from bench import metrics as M
from bench.templates import KINDS, make_page, page_specs


def test_generator_is_deterministic_and_complete():
    a = make_page("dashboard", "fa", "dark", 1)
    b = make_page("dashboard", "fa", "dark", 1)
    assert a["html"] == b["html"]
    assert len(page_specs(True)) == 80 and len(set(page_specs(True))) == 80
    assert {k for k, *_ in page_specs(True)} == set(KINDS)
    assert 'dir="rtl"' in a["html"] and 'data-gt="button"' in a["html"]
    assert make_page("mobile", "en", "light", 0)["width"] == 390


def test_text_normalisation_for_cer():
    assert M.norm("ثبت‌نام ۱۲") == M.norm("ثبتنام 12")        # ZWNJ, spaces and digit script are ignored
    assert M.norm("كتاب") == M.norm("کتاب")
    assert M.script_of("سلام") == "fa" and M.script_of("hello") == "en"


def test_struct_eval_counts():
    gt = [dict(role="button", box=[0, 0, 100, 40]), dict(role="button", box=[200, 0, 100, 40])]
    pred = [dict(role="button", box=[2, 1, 100, 40]), dict(role="card", box=[0, 0, 100, 40])]
    r = M.struct_eval(gt, pred, roles=["button", "card"])
    assert r["button"] == dict(tp=1, fp=0, fn=1) and r["card"] == dict(tp=0, fp=1, fn=0)
    p, rc, f = M.prf(1, 0, 1)
    assert (p, round(rc, 2)) == (1.0, 0.5)


def test_text_eval_penalises_image_of_text():
    gt = [dict(text="Hello world", box=[10, 10, 100, 20], dir="ltr", fs=16, color="rgb(0,0,0)", fw=400)]
    good = [dict(text="Hello world", box=[10, 10, 100, 20], occluded=False, fs=16, color="rgb(0,0,0)", fw=400)]
    none = []
    assert M.text_eval(gt, good)["by_script"]["en"]["err"] == 0
    assert M.text_eval(gt, none)["by_script"]["en"]["err"] == pytest.approx(len("Helloworld"))


def test_editability_blank_vs_crop():
    gt = [dict(role="image", box=[10, 10, 50, 50])]
    page_crop = [dict(box=[0, 0, 200, 200])]                  # one big <img> of the whole page
    assert M.editability_eval(200, 200, page_crop, gt)["editable"] < 0.1
    ok = [dict(box=[10, 10, 50, 50])]                         # only the photo is a raster
    assert M.editability_eval(200, 200, ok, gt)["editable"] == pytest.approx(1.0)


def test_visual_metric_control_shows_the_old_claim_is_cheap():
    img = np.full((100, 100, 3), 240, np.uint8)
    img[40:60, 40:60] = 20
    blank = np.full((100, 100, 3), 240, np.uint8)
    v = M.visual_eval(img, blank)
    assert v["legacy_match"] > 0.9 and v["ssim"] < 0.99       # blank page scores high on the old measure
