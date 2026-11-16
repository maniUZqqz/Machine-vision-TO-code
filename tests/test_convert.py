import json
import os
import subprocess
import sys

import numpy as np
import pytest

import mv2code
from mv2code.imaging import ImageError


def test_convert_returns_all_parts(ui_image):
    res = mv2code.convert(ui_image, mode="responsive", ocr="none", refine="off")
    assert set(res) >= {"html", "css", "tree_json", "report", "assets"}
    assert res["tree_json"]["role"] == "page" and res["tree_json"]["width"] == 640
    roles = {n["role"] for n in _walk(res["tree_json"])}
    assert "card" in roles
    assert res["report"]["mode"] == "responsive"
    assert any(a["path"].startswith("fonts/") for a in res["assets"])


def _walk(n):
    yield n
    for c in n.get("children", []):
        yield from _walk(c)


def test_both_modes_work_and_exact_is_absolute(ui_image):
    r = mv2code.convert(ui_image, mode="responsive", ocr="none", refine="off")
    e = mv2code.convert(ui_image, mode="exact", ocr="none", refine="off")
    assert "position:absolute" not in r["css"]
    assert "position:absolute" in e["css"]


def test_deterministic(ui_image):
    a = mv2code.convert(ui_image, ocr="none", refine="off")
    b = mv2code.convert(ui_image, ocr="none", refine="off")
    assert a["html"] == b["html"]
    assert json.dumps(a["tree_json"], sort_keys=True) == json.dumps(b["tree_json"], sort_keys=True)


def test_external_ocr_lines_become_real_text(ui_image):
    lines = [dict(box=(80, 150, 200, 20), text="سلام دنیا ۱۲۳", conf=0.99, tighten=False)]
    res = mv2code.convert(ui_image, ocr_lines=lines, refine="off")
    assert "سلام دنیا" in res["html"] and res["report"]["page_direction"] == "rtl"
    texts = [n for n in _walk(res["tree_json"]) if n["role"] == "text"]
    assert texts and texts[0]["direction"] == "rtl"


def test_clear_errors():
    with pytest.raises(ValueError):
        mv2code.convert(np.zeros((50, 50, 3), np.uint8), mode="weird")
    with pytest.raises(ImageError):
        mv2code.convert(b"garbage")
    with pytest.raises(ValueError):
        mv2code.convert(np.zeros((50, 50, 3), np.uint8), lang="de")


def test_unknown_ocr_backend_message(ui_image):
    with pytest.raises(ValueError, match="unknown OCR backend"):
        mv2code.convert(ui_image, ocr="nope")


def test_large_image_handling():
    big = np.full((900, 4800, 3), 250, np.uint8)
    big[100:700, 200:4600] = 30
    res = mv2code.convert(big, ocr="none", refine="off")
    assert res["report"]["size"][0] == 2400 and res["report"]["scale"] == 0.5


def test_cli_end_to_end(tmp_path, ui_png):
    out = tmp_path / "out"
    cp = subprocess.run([sys.executable, "-m", "mv2code", ui_png, "-o", str(out), "--ocr", "none", "--refine", "off", "--zip"],
                        capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert cp.returncode == 0, cp.stderr
    for name in ("index.html", "tree.json", "report.json", "result.zip", "original.png"):
        assert (out / name).exists(), name
    assert (out / "fonts").is_dir()
    assert json.loads((out / "tree.json").read_text(encoding="utf-8"))["role"] == "page"


def test_cli_errors(tmp_path):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cp = subprocess.run([sys.executable, "-m", "mv2code", str(tmp_path / "missing.png")], capture_output=True, text=True, cwd=root)
    assert cp.returncode == 2 and "not found" in cp.stderr


@pytest.mark.ocr
def test_tesseract_reads_persian_and_english(tmp_path):
    """Render Persian + English text with the bundled fonts, convert, expect the words back."""
    from PIL import Image, ImageDraw, ImageFont
    from mv2code import textstyle
    if not textstyle.HAS_RAQM:
        pytest.skip("Pillow without raqm cannot shape Persian")
    im = Image.new("RGB", (700, 160), (255, 255, 255))
    d = ImageDraw.Draw(im)
    fa = ImageFont.truetype(os.path.join(textstyle.FONT_DIR, "vazirmatn-arabic-400-normal.ttf"), 30)
    en = ImageFont.truetype(os.path.join(textstyle.FONT_DIR, "inter-latin-400-normal.ttf"), 28)
    d.text((660, 30), "سلام دنیا", font=fa, fill=(20, 20, 20), direction="rtl", language="fa", anchor="ra")
    d.text((40, 90), "Hello world dashboard", font=en, fill=(20, 20, 20))
    res = mv2code.convert(np.array(im), ocr="tesseract", refine="off")
    html = res["html"]
    assert "Hello" in html and "dashboard" in html
    assert "سلام" in html
