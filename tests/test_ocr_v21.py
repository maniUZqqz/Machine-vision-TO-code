"""v2.1: word-list repair of Persian OCR errors, junk words, long text lines, text inside gradient buttons, external OCR (crops out, text in)."""
import os

import cv2
import numpy as np
import pytest

import mv2code
from mv2code import lexicon, persian
from mv2code.ocr import CallbackBackend, OcrResult, PrecomputedBackend, TesseractBackend

FONT = os.path.join(os.path.dirname(mv2code.__file__), "fonts")


def _vazir():
    for n in sorted(os.listdir(FONT)):
        if n.lower().startswith("vazirmatn") and n.lower().endswith((".ttf", ".otf")):
            return os.path.join(FONT, n)
    pytest.skip("bundled Vazirmatn font not found")


def _persian_font(size, bold=False):
    from PIL import ImageFont, features
    if not features.check("raqm"):
        pytest.skip("Pillow without raqm cannot shape Persian")
    return ImageFont.truetype(_vazir(), size, layout_engine=ImageFont.Layout.RAQM)


def _draw(img, xy, text, size, fill, anchor="ra"):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    ImageDraw.Draw(im).text(xy, text, font=_persian_font(size), fill=fill, anchor=anchor, direction="rtl", language="fa")
    return np.array(im)


# ----------------------------------------------------------------------------- word list
@pytest.mark.parametrize("bad,good", [("محقوظ", "محفوظ"), ("پشنیبانی", "پشتیبانی")])
def test_dot_errors_are_repaired_from_the_word_list(bad, good):
    assert lexicon.correct_text(bad, conf=0.6) == good
    assert lexicon.correct_text("تمامی حقوق " + bad + " است", conf=0.8) == "تمامی حقوق " + good + " است"


def test_good_words_rare_words_and_short_words_are_left_alone():
    lex = lexicon.load()
    assert lex is not None and len(lex.zipf) > 20000
    for w in ("کاربر", "سلحشور", "همدان", "اعتمادیه", "چغر", "جغر", "چفر", "نو", "من"):
        assert lexicon.correct_text(w, conf=0.5) == w, w


def test_half_space_is_restored_when_the_list_knows_the_spelling():
    assert lexicon.correct_text("میکند", 0.8) == "می‌کند"
    assert lexicon.correct_text("ثبتنام", 0.8) == "ثبت‌نام"
    assert lexicon.correct_text("می‌کند", 0.8) == "می‌کند"


def test_brand_words_attract_look_alikes_and_are_never_changed():
    assert lexicon.correct_text("چخر چفر چعر", 0.6, glossary=["چغر"]) == "چغر چغر چغر"
    assert lexicon.correct_text("چغر", 0.6, glossary=["چغر"]) == "چغر"
    assert lexicon.correct_text("کاربر", 0.6, glossary=["چغر"]) == "کاربر"       # a known word is not turned into the brand


def test_words_repeated_on_the_page_are_protected():
    rep = lexicon.repeated_words(["محقوظ زیبا", "محقوظ", "باز محقوظ"])
    assert "محقوظ" in rep
    assert lexicon.correct_text("محقوظ", 0.6, protected=rep) == "محقوظ"


def test_punctuation_and_untouched_scripts():
    assert persian.fix_punctuation("شود, نه نوشتن؟") == "شود، نه نوشتن؟"
    assert persian.fix_punctuation("a, b?") == "a, b?"
    assert persian.normalize_line("شود, نه", "fa") == "شود، نه"
    assert persian.normalize_line("Hello, world", "en") == "Hello, world"
    assert lexicon.correct_text("hello wrld ۱۲۳ 45%", 0.5) == "hello wrld ۱۲۳ 45%"


# ----------------------------------------------------------------------------- junk words
def _ws(*items):
    return [(1, 1, 1, i * 10, t, c) for i, (t, c) in enumerate(items)]


def test_low_confidence_latin_fragments_in_a_persian_line_are_dropped():
    ws = _ws(("امکانات", 91), ("#", 60), ("als", 41), ("&", 30))
    kept = [w[4] for w in TesseractBackend._drop_junk_words(ws)]
    assert kept == ["امکانات"]
    ws = _ws(("خروجی", 90), ("PDF", 93), ("قابل", 88))
    assert [w[4] for w in TesseractBackend._drop_junk_words(ws)] == ["خروجی", "PDF", "قابل"]
    ws = _ws(("Hello", 40), ("wrld", 35))                                          # a Latin line is not touched
    assert len(TesseractBackend._drop_junk_words(ws)) == 2
    ws = _ws(("۲۵", 30), ("کاربر", 80))                                              # digits stay
    assert len(TesseractBackend._drop_junk_words(ws)) == 2


def test_an_icon_read_as_a_short_word_at_the_end_of_a_line_is_dropped():
    # (line, par, block, left, text, conf, width): icon fragment far to the right of the label
    ws = [(1, 1, 1, 0, "چیه؟", 90, 40), (1, 1, 1, 48, "چغر", 92, 30), (1, 1, 1, 84, "ورود", 91, 36), (1, 1, 1, 150, "عد", 55, 14)]
    assert [w[4] for w in TesseractBackend._drop_edge_fragments(ws)] == ["چیه؟", "چغر", "ورود"]
    ws = [(1, 1, 1, 0, "چیه؟", 90, 40), (1, 1, 1, 48, "چغر", 92, 30), (1, 1, 1, 84, "ورود", 91, 36), (1, 1, 1, 126, "ما", 55, 14)]      # normal word gap: kept
    assert len(TesseractBackend._drop_edge_fragments(ws)) == 4
    ws = [(1, 1, 1, 0, "چیه؟", 90, 40), (1, 1, 1, 48, "چغر", 92, 30), (1, 1, 1, 84, "ورود", 91, 36), (1, 1, 1, 150, "ما", 90, 14)]      # confident: kept
    assert len(TesseractBackend._drop_edge_fragments(ws)) == 4


# ----------------------------------------------------------------------------- long line, gradient button
def test_a_long_one_line_paragraph_is_text_not_a_rule():
    img = np.full((120, 1280, 3), 255, np.uint8)
    text = "همین حالا اولین برنامه اختصاصی تان را با چغر بسازید و ببینید نوشتن برنامه چقدر کوتاه تر می شود اگر قواعد را بدهید"
    img = _draw(img, (1200, 40), text, 20, (30, 30, 40))
    from mv2code.build import read_text, select_candidates
    from mv2code.pipeline import analyze
    an = analyze(img)
    cands = select_candidates(an)
    assert cands, "the line must be found"
    widest = max(cands, key=lambda c: c.box[2])
    assert widest.box[2] / widest.box[3] > 22, widest.box           # this is the case the old aspect rule threw away
    lines, _ = read_text(an, CallbackBackend(lambda crops, lang: [(text, 0.9)] * len(crops)), "fa")
    assert any(len(ln.text) > 60 for ln in lines)


def _gradient_pill(w=1280, h=300):
    img = np.full((h, w, 3), (244, 248, 255), np.uint8)
    x0, y0, bw, bh = 520, 100, 240, 70
    for x in range(bw):
        t = x / float(bw - 1)
        col = (int(21 + (18 - 21) * t), int(120 + (60 - 120) * t), int(140 + (110 - 140) * t))
        img[y0:y0 + bh, x0 + x] = col
    mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(mask, (x0, y0), (x0 + bw, y0 + bh), 255, -1)
    glow = cv2.GaussianBlur(mask, (0, 0), 14).astype(np.float32) / 255.0
    base = np.full_like(img, (244, 248, 255)).astype(np.float32)
    shadow = base * (1 - 0.25 * glow[..., None])
    sh = (np.roll(shadow, 10, axis=0)).astype(np.uint8)
    out = sh.copy()
    pill = np.zeros((h, w), np.uint8)
    cv2.rectangle(pill, (x0 + 30, y0), (x0 + bw - 30, y0 + bh), 255, -1)
    cv2.circle(pill, (x0 + 30, y0 + bh // 2), bh // 2, 255, -1)
    cv2.circle(pill, (x0 + bw - 30, y0 + bh // 2), bh // 2, 255, -1)
    pill = pill > 0
    out[pill] = img[pill]
    return out, (x0, y0, bw, bh)


def test_label_inside_a_gradient_pill_is_found_as_its_own_line():
    img, (x0, y0, bw, bh) = _gradient_pill()
    img = _draw(img, (x0 + bw - 40, y0 + bh // 2), "شروع رایگان", 22, (255, 255, 255), anchor="rm")
    from mv2code.build import select_candidates
    from mv2code.pipeline import analyze
    an = analyze(img)
    inside = [c for c in select_candidates(an) if y0 - 4 <= c.box[1] and c.box[1] + c.box[3] <= y0 + bh + 4]
    assert inside, [c.box for c in an.cands]
    for c in inside:
        assert c.box[0] >= x0 - 4 and c.box[0] + c.box[2] <= x0 + bw + 4, ("a candidate spills out of the button", c.box)
    lab = max(inside, key=lambda c: c.box[2])
    assert 50 <= lab.box[2] <= 170 and 12 <= lab.box[3] <= 40, lab.box


# ----------------------------------------------------------------------------- external OCR
def _page():
    img = np.full((260, 900, 3), 250, np.uint8)
    img = _draw(img, (860, 40), "عنوان صفحه", 30, (20, 20, 30))
    img = _draw(img, (860, 120), "متن نمونه برای آزمون", 20, (40, 40, 50))
    img = _draw(img, (860, 180), "همهٔ حقوق محفوظ است", 16, (90, 90, 100))
    return img


def test_ocr_crops_give_the_crops_the_converter_would_read():
    img = _page()
    req = mv2code.ocr_crops(img)
    assert req["size"] == [900, 260]
    assert len(req["items"]) >= 3
    for it in req["items"]:
        assert len(it["box"]) == 4 and it["crop"].ndim == 2 and it["crop"].dtype == np.uint8 and it["crop"].shape[0] >= 12


def test_precomputed_results_are_matched_by_box_and_the_rest_falls_back():
    img = _page()
    req = mv2code.ocr_crops(img)
    first = req["items"][0]
    res = [{"box": first["box"], "text": "عنوان از گره دور", "conf": 0.97}]

    class Fallback:
        name = "fb"

        def read(self, crops, lang="auto"):
            return [OcrResult("بازگشتی", 0.8, 1) for _ in crops]
    be = PrecomputedBackend(res, fallback=Fallback())
    out = mv2code.convert(img, ocr=be, refine="off")
    assert "عنوان از گره دور" in out["html"]
    assert "بازگشتی" in out["html"]
    assert out["report"]["ocr"] == "external"
    assert out["report"]["ocr_stats"]["external"] == 1 and out["report"]["ocr_stats"]["fallback"] >= 1


def test_a_callable_is_accepted_as_ocr_and_glossary_reaches_the_repair():
    img = _page()
    seen = []

    def fn(crops, lang):
        seen.append(len(crops))
        return [("چخر" if i == 0 else "متن", 0.6) for i in range(len(crops))]
    out = mv2code.convert(img, ocr=fn, refine="off", glossary=["چغر"])
    assert seen and "چغر" in out["html"] and "چخر" not in out["html"]


def test_mismatched_external_count_is_an_error_not_silent_garbage():
    be = CallbackBackend(lambda crops, lang: [("x", 0.9)])
    with pytest.raises(RuntimeError, match="external OCR returned"):
        be.read([np.zeros((10, 10), np.uint8)] * 3)
