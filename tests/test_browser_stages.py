"""Feedback loop, scoring and responsive behaviour in a real headless browser (skipped without one)."""
import numpy as np
import pytest

import mv2code
from tests.conftest import synthetic_ui

pytestmark = pytest.mark.browser


def test_refine_never_makes_it_worse_and_reports_steps():
    img = synthetic_ui()
    lines = [dict(box=(80, 150, 200, 20), text="Hello dashboard", conf=1.0, tighten=False),
             dict(box=(80, 190, 120, 18), text="Second line", conf=1.0, tighten=False)]
    plain = mv2code.convert(img, ocr_lines=lines, refine="off")
    res = mv2code.convert(img, ocr_lines=lines, refine="on")
    rep = res["report"]["refine"]
    assert rep and "ssim_steps" in rep and rep["best_ssim"] >= rep["ssim_steps"][0] - 1e-6
    assert "Hello dashboard" in res["html"] and "Hello dashboard" in plain["html"]


def test_score_reports_editable_share_and_ssim():
    from mv2code.browser import Browser
    from mv2code.score import compare
    img = synthetic_ui()
    res = mv2code.convert(img, ocr="none", refine="off")
    with Browser() as br:
        s = compare(img, res["html"], res["assets"], br)
    assert 0.5 < s["ssim"] <= 1.0 and s["editable_share"] > 0.9 and s["pixel_similarity"] > 0.9


def test_responsive_output_reflows_without_overflow():
    from mv2code.browser import Browser
    img = np.full((500, 1200, 3), 245, np.uint8)
    lines = [dict(box=(100 + 300 * i, 100, 200, 22), text=f"Card number {i}", conf=1.0, tighten=False) for i in range(3)]
    res = mv2code.convert(img, mode="responsive", ocr_lines=lines, refine="off")
    import os, tempfile, shutil
    d = tempfile.mkdtemp()
    try:
        res_path = os.path.join(d, "p.html")
        open(res_path, "w", encoding="utf-8").write(res["html"])
        with Browser() as br:
            for w in (360, 768, 1200):
                _, probe = br.render("file://" + res_path.replace("\\", "/") if os.name != "nt" else "file:///" + res_path.replace("\\", "/"), w, shot=False)
                assert probe["W"] <= w + 1, f"horizontal overflow at {w}px"
    finally:
        shutil.rmtree(d, ignore_errors=True)
