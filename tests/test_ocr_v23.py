"""v2.3: icon glyphs read as digits/letters, labels of several buttons under one merged candidate."""
import numpy as np

from mv2code.build import _lone_icon_reading, _rtl_page
from mv2code.ocr import TesseractBackend
from mv2code.segment import Region
from mv2code.textdet import Cand, _fix_leaky


def _w(text, left, width=40, conf=90.0, top=5, height=20):
    return (1, 1, 1, left, text, conf, width, top, height)


# ----------------------------------------------------------------------------- word level
def test_a_lone_latin_digit_at_the_end_of_a_persian_label_is_the_icon_of_the_button():
    # the real case: «9 شروع رایگان» (a paper-plane icon read as 9, 7 px from the label)
    ws = [_w("9", 159, 28, conf=90.0), _w("شروع", 91, 61, conf=92.0), _w("رایگان", 21, 62, conf=87.0)]
    assert TesseractBackend._assemble(ws).text == "شروع رایگان"
    # a Persian digit is a number written on purpose
    ws = [_w("۹", 159, 28, conf=90.0), _w("شروع", 91, 61, conf=92.0), _w("رایگان", 21, 62, conf=87.0)]
    assert "۹" in TesseractBackend._assemble(ws).text
    # a very sure reading stays
    ws = [_w("5", 159, 28, conf=98.0), _w("روز", 91, 61, conf=92.0), _w("رایگان", 21, 62, conf=87.0)]
    assert "5" in TesseractBackend._assemble(ws).text
    # a number inside the line is not at its end
    ws = [_w("با", 159, 28, conf=90.0), _w("3", 120, 20, conf=90.0), _w("رایگان", 21, 62, conf=87.0)]
    assert "3" in TesseractBackend._assemble(ws).text


def test_one_or_two_unsure_characters_set_apart_from_the_only_word_are_the_icon_of_a_nav_item():
    ws = [_w("هص", 158, 18, conf=36.0), _w("وبلاگ", 36, 95, conf=91.0)]          # icon + «وبلاگ», the gap is wider than a space
    assert TesseractBackend._assemble(ws).text == "وبلاگ"
    ws = [_w("و", 110, 10, conf=44.0), _w("وبلاگ", 36, 66, conf=91.0)]           # a normal gap: the conjunction «و» stays
    assert TesseractBackend._assemble(ws).text == "و وبلاگ"


# ----------------------------------------------------------------------------- candidate level
def _cands(boxes):
    return [Cand(box=b, n_comp=1, ink=0.3) for b in boxes]


def test_a_lone_digit_beside_a_line_of_words_on_a_persian_page_is_an_icon_but_pagination_is_not():
    cands = _cands([(417, 2282, 14, 18), (300, 2282, 100, 18)])
    texty = [False, True]
    assert _lone_icon_reading("9", "9", 1, ("9", 0.94), 0.94, True, 0, cands, texty, 18)
    # the same digit on an English page, or a Persian digit, stays
    assert not _lone_icon_reading("9", "9", 1, ("9", 0.94), 0.94, False, 0, cands, texty, 18)
    assert not _lone_icon_reading("۹", "۹", 1, ("۹", 0.94), 0.94, True, 0, cands, texty, 18)
    # pagination: the neighbours are digits, not lines of words
    pag = _cands([(100, 50, 12, 18), (130, 50, 12, 18), (160, 50, 12, 18)])
    assert not _lone_icon_reading("2", "2", 1, ("2", 0.95), 0.95, True, 1, pag, [False, False, False], 18)


def test_two_lowercase_latin_letters_alone_are_a_social_icon_on_a_persian_page():
    cands = _cands([(187, 2360, 16, 16)])
    assert _lone_icon_reading("in", "in", 2, ("in", 0.97), 0.97, True, 0, cands, [False], 16)
    assert not _lone_icon_reading("AI", "AI", 2, ("AI", 0.97), 0.97, True, 0, cands, [False], 16)          # a real badge
    assert not _lone_icon_reading("in", "in", 2, ("in", 0.97), 0.97, False, 0, cands, [False], 16)         # English page


def test_a_glyph_whose_reading_changes_between_the_two_passes_is_not_a_character():
    cands = _cands([(249, 976, 43, 48)])
    assert _lone_icon_reading("0", "0", 1, ("۴", 0.52), 0.95, True, 0, cands, [False], 48)
    assert not _lone_icon_reading("0", "0", 1, ("0", 0.9), 0.95, True, 0, cands, [False], 48)


def test_page_direction_for_the_icon_rules():
    assert _rtl_page(["سلام دنیا", "x"], "auto") and not _rtl_page(["hello world", "سلام"], "auto")
    assert _rtl_page([], "fa") and not _rtl_page(["سلام دنیا"], "en")


# ----------------------------------------------------------------------------- one merged candidate over several buttons
def test_one_candidate_that_spans_two_buttons_gives_both_labels_back():
    img = np.full((140, 520, 3), 245, np.uint8)
    regs = []
    for i, x in enumerate((20, 280)):
        img[30:100, x:x + 220] = (20, 90, 126)                    # a dark teal button
        img[56:74, x + 60:x + 160] = (255, 255, 255)              # its white label (a bar is enough ink for the finder)
        regs.append(Region(id=i, box=(x, 30, 220, 70), inner=(x, 30, 220, 70), fill=(20, 90, 126), radius=14.0, area=220 * 70))
    merged = Cand(box=(10, 20, 500, 90), n_comp=4, ink=0.2)       # glow + outline merged the two buttons into one candidate
    out = _fix_leaky(img, [merged], regs)
    labels = [c for c in out if c.box[2] >= 80 and 50 <= c.box[1] <= 60]
    assert len(labels) == 2, [c.box for c in out]
    assert {c.box[0] // 200 for c in labels} == {0, 1}            # one label in each button
    assert not any(c.box[2] >= 400 for c in out)                  # the merged candidate is gone
