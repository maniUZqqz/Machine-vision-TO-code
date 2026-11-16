from mv2code import persian as P


def test_letter_normalisation():
    assert P.normalize_letters("كتاب ي") == "کتاب ی"          # Arabic kaf / yeh -> Persian
    assert P.normalize_letters("مَدرِسه") == "مدرسه"           # harakat removed
    assert P.normalize_letters("ـــ") == ""                    # tatweel removed


def test_digits_follow_the_page():
    assert P.arabic_indic_to_persian("٤٥٦") == "۴۵۶"
    # only ambiguous glyphs (1 2 3): the page decides
    assert P.fix_digits("123", "fa", True) == "۱۲۳"
    assert P.fix_digits("123", "fa", False) == "123"          # Latin line keeps Latin digits
    # unambiguous Latin glyph wins over the page
    assert P.fix_digits("4 users", "fa", False) == "4 users"
    assert P.fix_digits("۴ کاربر", "en", True) == "۴ کاربر"


def test_page_digit_decision():
    assert P.decide_page_digits(["۴۵ کاربر", "سلام"]) == "fa"
    assert P.decide_page_digits(["45 users", "hello"]) == "en"


def test_direction_and_script():
    assert P.is_rtl("سلام دنیا")
    assert not P.is_rtl("Hello world")
    assert P.is_rtl("خرید iPhone")                               # majority Persian letters
    assert P.page_direction(["سلام", "دنیا", "ok"]) == "rtl"


def test_zwnj_restoration():
    assert P.restore_zwnj("می شود") == "می‌شود"
    assert P.restore_zwnj("کتاب ها") == "کتاب‌ها"
    assert P.restore_zwnj("سلام دنیا") == "سلام دنیا"             # unrelated text untouched


def test_normalize_line_is_idempotent():
    s = "ثبت‌نام ۱۲۳ كاربر"
    once = P.normalize_line(s, "fa")
    assert P.normalize_line(once, "fa") == once
