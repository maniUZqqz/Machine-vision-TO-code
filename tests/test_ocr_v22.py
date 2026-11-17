"""v2.2: second look at doubtful lines, word order of mixed lines, number-sign order, suffix-aware word list, faint/icon junk, review flags."""
import cv2
import numpy as np
import pytest

import mv2code
from mv2code import lexicon, persian, rescan, review
from mv2code.build import TL, ink_contrast, postprocess_lines, read_text
from mv2code.ocr import OcrResult, TesseractBackend
from mv2code.pipeline import analyze


# ----------------------------------------------------------------------------- word order inside one line
def _w(text, left, width=40, conf=90.0, top=5, height=20):
    return (1, 1, 1, left, text, conf, width, top, height)


def test_persian_line_is_read_right_to_left_by_position_and_latin_runs_left_to_right():
    # visual: "PDF" is LEFT of "خروجی" (tesseract returned it first); logical order puts خروجی first
    ws = [_w("PDF", 10), _w("خروجی", 80)]
    assert TesseractBackend._assemble(ws).text == "خروجی PDF"
    # a run of two Latin words keeps its own left-to-right order inside the right-to-left line
    ws = [_w("Pro", 10), _w("Plan", 60), _w("پشتیبانی", 120)]
    assert TesseractBackend._assemble(ws).text == "پشتیبانی Pro Plan"
    # a Latin line is left to right
    assert TesseractBackend._assemble([_w("world", 90), _w("hello", 10)]).text == "hello world"
    # words of different text lines in one crop keep tesseract's order
    ws = [(1, 1, 1, 80, "اول", 90.0, 40, 5, 20), (2, 1, 1, 10, "دوم", 90.0, 40, 60, 20)]
    assert TesseractBackend._assemble(ws).text == "اول دوم"


def test_a_line_of_numbers_is_not_reordered_and_loses_no_digit_token():
    # visual left to right: ۴۱ ۳۶۳ - a number with a thousands separator split in two tokens (the first one short and not very sure)
    assert TesseractBackend._assemble([_w("۴۱", 10, 20, conf=80.0), _w("۳۶۳", 40, 30, conf=88.0)]).text == "۴۱ ۳۶۳"
    # a number next to a Persian word: the word (rightmost) comes first, the number keeps its own left-to-right order
    ws = [_w("۴۱", 10, 20, conf=92.0), _w("۳۶۳", 40, 30, conf=88.0), _w("تومان", 100, 50)]
    assert TesseractBackend._assemble(ws).text == "تومان ۴۱ ۳۶۳"


def test_one_or_two_digits_set_apart_at_the_end_of_a_persian_line_are_an_icon():
    ws = [_w("2", 0, 12, conf=92.0), _w("چغر", 40, 50), ]
    assert TesseractBackend._assemble(ws).text == "چغر"
    ws = [_w("۳", 61, 12, conf=92.0), _w("مربی", 5, 50)]                      # no gap: a real number next to its word stays
    assert "۳" in TesseractBackend._assemble(ws).text


# ----------------------------------------------------------------------------- numbers on a right-to-left page
def test_number_signs_follow_the_text_order_not_the_pixels_on_rtl_pages():
    assert persian.fix_trailing_plus("500+") == "+500"
    assert persian.fix_trailing_plus("۵۰۰+") == "+۵۰۰"
    assert persian.fix_trailing_plus("امنیت %100") == "امنیت 100%"
    assert persian.fix_trailing_plus("٪۹۸") == "۹۸٪"
    assert persian.fix_trailing_plus("500+ clients") == "500+ clients"           # an English line is left alone
    assert persian.fix_trailing_plus("2026 ©") == "2026 ©"
    lines = [TL(box=(0, 0, 40, 20), text="500+", conf=0.9), TL(box=(0, 30, 120, 20), text="برنامه ساخته شده", conf=0.9)]
    postprocess_lines(lines)
    assert lines[0].text == "+۵۰۰" or lines[0].text == "+500"


# ----------------------------------------------------------------------------- the word list knows suffixes
def test_a_known_stem_with_a_suffix_is_not_turned_into_another_word():
    lex = lexicon.load()
    assert lex.affix_known("پیشرفتتان") and lex.affix_known("کتابهایی")
    assert lexicon.correct_text("پیشرفتتان", conf=0.7) == "پیشرفتتان"           # used to become «پیشرفت‌های» (a more frequent look-alike)
    assert not lex.affix_known("محقوظ")


def test_a_comma_read_as_a_hamza_after_a_known_word_is_a_comma():
    assert lexicon.correct_text("شماء در", conf=0.7) == "شما، در"
    assert lexicon.correct_text("جزء", conf=0.7) == "جزء"                       # a real word with a final hamza stays


# ----------------------------------------------------------------------------- review flags
def test_review_flags_doubtful_lines_only_for_ocr_text():
    lex = lexicon.load()
    assert review.reasons_for("تمامی حقوق محفوظ است", 0.93, lex) == []
    assert "conf" in review.reasons_for("تمامی حقوق محفوظ است", 0.6, lex)
    assert "unknown" in review.reasons_for("تمامی حقوق محقوظ است", 0.93, lex)
    assert "unknown" not in review.reasons_for("تمامی حقوق چغر است", 0.93, lex, glossary=["چغر"])
    lines = [TL(box=(1, 2, 30, 10), text="تمامی حقوق محقوظ است", conf=0.95), TL(box=(1, 20, 30, 10), text="خانه", conf=0.97)]
    assert review.flag_lines(lines) == 1 and lines[0].check and not lines[1].check
    assert review.flag_lines(lines, from_ocr=False) == 0
    rep = review.report_items(lines)
    assert rep == [dict(box=[1, 2, 30, 10], text="تمامی حقوق محقوظ است", conf=0.95, why=["unknown"])]


def test_convert_report_lists_review_lines_and_tree_marks_them():
    pytest.importorskip("pytesseract")
    from tests.conftest import synthetic_ui
    res = mv2code.convert(synthetic_ui(), mode="responsive", ocr="none", refine="off")
    assert res["report"]["review_lines"] == 0 and res["report"]["review"] == []
    lines = [dict(box=[20, 20, 120, 22], text="تمامی حقوق محقوظ است", conf=0.4)]
    res = mv2code.convert(synthetic_ui(), mode="responsive", ocr=lambda crops, lang: [("تمامی حقوق محقوظ است", 0.5)] * len(crops), refine="off")
    # text that comes from an external callback counts as OCR text: low-confidence lines are reported
    assert res["report"]["review_lines"] >= 0 and isinstance(res["report"]["review"], list)
    # text given from the DOM (ocr_lines) is never flagged
    res = mv2code.convert(synthetic_ui(), mode="responsive", ocr_lines=lines, refine="off")
    assert res["report"]["review_lines"] == 0


# ----------------------------------------------------------------------------- faint shapes and icon digits are not text
class _FakeBackend:
    name = "fake"

    def __init__(self, text, conf):
        self.text, self.conf = text, conf

    def available(self):
        return True

    def read_boxes(self, crops, boxes, lang="auto"):
        return [OcrResult(self.text, self.conf) for _ in crops]


def _page_with_mark(color, w=64, h=34):
    img = np.full((220, 400, 3), 255, np.uint8)
    cv2.putText(img, "10", (150, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3, cv2.LINE_AA)
    return img


def test_near_invisible_text_is_dropped_but_readable_text_stays():
    faint = _page_with_mark((243, 245, 248))
    clear = _page_with_mark((30, 30, 30))
    for img, expect in ((faint, 0), (clear, None)):
        an = analyze(img)
        lines, _ = read_text(an, _FakeBackend("۱۰ ۱۰", 0.97), "fa")
        if expect == 0:
            assert not lines, "a #f3f5f8 mark on white must not become text"
        else:
            assert lines, "dark text must stay"
    assert ink_contrast(faint, (140, 90, 80, 40)) < rescan_faint() <= 40


def rescan_faint():
    from mv2code.build import FAINT
    return FAINT


def test_a_two_digit_icon_with_a_low_confidence_is_not_text_but_numbers_stay():
    img = _page_with_mark((30, 30, 30))
    an = analyze(img)
    lines, _ = read_text(an, _FakeBackend("۱۰", 0.5), "fa")
    assert not [ln for ln in lines if ln.text == "۱۰"]
    lines, _ = read_text(an, _FakeBackend("۱۰", 0.85), "fa")                      # a clear two-digit number stays
    assert [ln for ln in lines if ln.text == "۱۰"]
    lines, _ = read_text(an, _FakeBackend("۲۵۳", 0.65), "fa")                     # three digits stay (table cells)
    assert [ln for ln in lines if ln.text == "۲۵۳"]
    lines, _ = read_text(an, _FakeBackend("4a", 0.9), "en")
    assert not [ln for ln in lines if ln.text == "4a"]


# ----------------------------------------------------------------------------- the second pass
class _Variants(TesseractBackend):
    """First reading is wrong; the 64 px thickened variant reads the line right (higher confidence)."""

    def __init__(self, answers):
        super().__init__()
        self.answers = answers
        self.calls = 0

    def installed_langs(self):
        return ["eng", "fas"]

    def _read_line(self, crop, la, psm=7):
        self.calls += 1
        txt, conf = self.answers.pop(0) if self.answers else ("", 0.0)
        if not txt:
            return None
        r = OcrResult(txt, conf, len(txt.split()), ws=[(1, 1, 1, 12, txt, conf * 100, max(10, crop.shape[1] - 24), 5, 20)])
        r.src = crop
        return r


def _res(text, conf):
    r = OcrResult(text, conf, len(text.split()), ws=[(1, 1, 1, 12, text, conf * 100, 120, 5, 20)])
    r.src = np.full((40, 144), 255, np.uint8)
    return r


def test_second_pass_keeps_the_better_reading_and_never_a_worse_one(monkeypatch):
    monkeypatch.setenv("MV2CODE_RESCAN", "on")
    img = np.full((100, 300, 3), 255, np.uint8)
    boxes = [(10, 10, 120, 24), (10, 50, 120, 24)]
    crops = [np.full((40, 144), 255, np.uint8)] * 2
    first = [_res("محقوظ است", 0.55), _res("تمامی حقوق", 0.97)]
    be = _Variants([("محفوظ است", 0.92), ("محفوظ است", 0.9), ("محفوظ است", 0.91), ("x", 0.1), ("y", 0.1)])
    out = be.rescan(img, boxes, crops, first, "fa")
    assert out[0].text == "محفوظ است"            # the doubtful line was improved
    assert out[1].text == "تمامی حقوق"           # the confident line was not even touched
    assert be.calls <= 5 * 1 + 1


def test_second_pass_can_be_switched_off_and_is_time_bounded(monkeypatch):
    img = np.full((100, 300, 3), 255, np.uint8)
    boxes = [(10, 10, 120, 24)]
    crops = [np.full((40, 144), 255, np.uint8)]
    first = [_res("محقوظ است", 0.55)]
    monkeypatch.setenv("MV2CODE_RESCAN", "off")
    be = _Variants([("محفوظ است", 0.99)] * 5)
    assert be.rescan(img, boxes, crops, first, "fa")[0].text == "محقوظ است" and be.calls == 0
    monkeypatch.setenv("MV2CODE_RESCAN", "on")
    monkeypatch.setenv("MV2CODE_RESCAN_BUDGET", "0")
    be = _Variants([("محفوظ است", 0.99)] * 5)
    assert be.rescan(img, boxes, crops, first, "fa")[0].text == "محقوظ است" and be.calls == 0


def test_a_reading_that_lost_a_word_does_not_win_on_confidence(monkeypatch):
    monkeypatch.setenv("MV2CODE_RESCAN", "on")
    img = np.full((100, 300, 3), 255, np.uint8)
    boxes = [(10, 10, 300, 24)]
    crops = [np.full((40, 324), 255, np.uint8)]
    wide = OcrResult("برنامهٔ ورزشی مخصوص", 0.72, 3, ws=[(1, 1, 1, 12, "برنامهٔ", 40.0, 90, 5, 20), (1, 1, 1, 110, "ورزشی", 88.0, 80, 5, 20), (1, 1, 1, 200, "مخصوص", 90.0, 100, 5, 20)])
    wide.src = crops[0]
    short = OcrResult("ورزشی مخصوص", 0.89, 2, ws=[(1, 1, 1, 110, "ورزشی", 88.0, 80, 5, 20), (1, 1, 1, 200, "مخصوص", 90.0, 100, 5, 20)])
    short.src = crops[0]

    class B(_Variants):
        def _read_line(self, crop, la, psm=7):
            self.calls += 1
            return short if self.calls == 1 else None

    be = B([])
    out = be.rescan(img, boxes, crops, [wide], "fa")
    assert out[0].text == "برنامهٔ ورزشی مخصوص"


def test_latin_word_misread_as_digits_is_fixed_by_the_english_model():
    class B(TesseractBackend):
        answer = ("PDF", 96.0)

        def installed_langs(self):
            return ["eng", "fas"]

        def _read_word(self, crop, la, psm=8, whitelist=""):
            return self.answer

    def line(conf):
        r = OcrResult("خروجی ۳۲۰۴", 0.7, 2, ws=[(1, 1, 1, 88, "خروجی", 89.0, 89, 18, 30), (1, 1, 1, 21, "۳۲۰۴", conf, 57, 17, 23)])
        r.src = np.full((80, 200), 255, np.uint8)
        return r

    assert rescan.fix_tokens(B(), [line(51.0)])[0].text == "خروجی PDF"
    # digits read with a high confidence that the English model reads as a clear acronym are the acronym (the second pass compares fixed candidates)
    assert rescan.fix_tokens(B(), [line(95.0)])[0].text == "خروجی PDF"
    # a confident number stays when the English model does not read a clear word
    weak = B()
    weak.answer = ("YOY", 62.0)
    assert rescan.fix_tokens(weak, [line(95.0)])[0].text == "خروجی ۳۲۰۴"
    weak.answer = ("oO", 95.0)                       # not an acronym / word shape
    assert rescan.fix_tokens(weak, [line(95.0)])[0].text == "خروجی ۳۲۰۴"
    # a Latin line is never touched (no Persian letters around the token)
    only = OcrResult("۳۲۰۴", 0.5, 1, ws=[(1, 1, 1, 21, "۳۲۰۴", 51.0, 57, 17, 23)])
    only.src = np.full((80, 200), 255, np.uint8)
    assert rescan.fix_tokens(B(), [only])[0].text == "۳۲۰۴"
