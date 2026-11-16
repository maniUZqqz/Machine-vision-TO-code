"""Synthetic page generator for the benchmark.

Every page is produced by OUR OWN templates (no third-party screenshots), so the
ground truth (box, role, text, style of every element) is exact: the templates
tag elements with ``data-gt="<role>"`` and ``bench.render`` reads the boxes and
computed styles back from the browser.

Pages are deterministic: ``make_page(kind, lang, theme, seed)`` always returns the
same HTML.  Text is synthetic (random words from small generic word lists) - no
real people, no real brands.
"""
from __future__ import annotations

import base64
import html as _html
import random
from dataclasses import dataclass

import cv2
import numpy as np

# --------------------------------------------------------------------------- words
FA_WORDS = (
    "خدمات محصول سفارش کاربر حساب گزارش تنظیمات پشتیبانی قیمت خرید فروش مشتری پروژه برنامه "
    "داشبورد تحلیل آموزش تمرین سلامت کیفیت سریع ساده امن رایگان ویژه جدید بهترین مدیریت "
    "ثبت‌نام ورود جستجو اطلاعات پرداخت تحویل فاکتور موجودی دسته‌بندی نمایش ذخیره انصراف "
    "ادامه بازگشت تماس درباره وبلاگ مقاله نظرات امتیاز تخفیف سبد کالا ارسال شرکت تیم هدف "
    "نتیجه روزانه هفتگی ماهانه سالانه تغذیه ورزش مربی بدنسازی رژیم برنامه‌ریزی پیشرفت "
    "اعلان پیام یادآوری تقویم نمودار آمار کارکرد عملکرد دسترسی کاربران مشاهده ویرایش حذف"
).split()
FA_TITLES = [
    "مدیریت هوشمند", "شروع کنید", "ساده، سریع و امن", "همه چیز در یک جا", "پیشنهاد ویژه",
    "گزارش عملکرد", "تنظیمات حساب", "ورود به حساب", "ثبت‌نام رایگان", "پرفروش‌ترین‌ها",
    "سوالات متداول", "تماس با ما", "آخرین مقالات", "برنامه‌های ما", "نمای کلی",
]
FA_NAMES = ["علی رضایی", "مریم احمدی", "حسین کریمی", "زهرا محمدی", "رضا موسوی", "سارا نوری",
            "امیر حسینی", "نگار صادقی"]
EN_WORDS = (
    "service product order user account report settings support price buy sale customer project "
    "plan dashboard analysis learning training health quality fast simple secure free premium new "
    "best manage signup login search details payment delivery invoice stock category show save "
    "cancel continue back contact about blog article review rating discount cart item shipping "
    "company team goal result daily weekly monthly yearly nutrition sport coach progress alert "
    "message reminder calendar chart stats activity access members view edit delete"
).split()
EN_TITLES = [
    "Smart management", "Get started today", "Simple, fast and secure", "Everything in one place",
    "Special offer", "Performance report", "Account settings", "Sign in to your account",
    "Create a free account", "Best sellers", "Frequently asked questions", "Contact us",
    "Latest articles", "Our plans", "Overview",
]
EN_NAMES = ["Ava Miller", "Noah Brown", "Mia Johnson", "Liam Davis", "Emma Wilson", "Lucas Moore",
            "Zoe Clark", "Owen Hall"]

FA_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")

KINDS = ["landing", "dashboard", "table", "form", "pricing", "login", "products", "blog",
         "mobile", "settings"]
LANGS = ["fa", "en"]
THEMES = ["light", "dark"]

FONT_FACE_CSS = """
@font-face{font-family:Vazirmatn;font-weight:400;font-style:normal;src:url(fonts/vazirmatn-arabic-400-normal.woff2) format('woff2');unicode-range:U+0600-06FF,U+0750-077F,U+FB50-FDFF,U+FE70-FEFC,U+200C-200E,U+066A-066C}
@font-face{font-family:Vazirmatn;font-weight:500;font-style:normal;src:url(fonts/vazirmatn-arabic-500-normal.woff2) format('woff2');unicode-range:U+0600-06FF,U+0750-077F,U+FB50-FDFF,U+FE70-FEFC,U+200C-200E,U+066A-066C}
@font-face{font-family:Vazirmatn;font-weight:700;font-style:normal;src:url(fonts/vazirmatn-arabic-700-normal.woff2) format('woff2');unicode-range:U+0600-06FF,U+0750-077F,U+FB50-FDFF,U+FE70-FEFC,U+200C-200E,U+066A-066C}
@font-face{font-family:Vazirmatn;font-weight:400;font-style:normal;src:url(fonts/vazirmatn-latin-400-normal.woff2) format('woff2');unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD}
@font-face{font-family:Vazirmatn;font-weight:500;font-style:normal;src:url(fonts/vazirmatn-latin-500-normal.woff2) format('woff2');unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD}
@font-face{font-family:Vazirmatn;font-weight:700;font-style:normal;src:url(fonts/vazirmatn-latin-700-normal.woff2) format('woff2');unicode-range:U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+2000-206F,U+2074,U+20AC,U+2122,U+2191,U+2193,U+2212,U+2215,U+FEFF,U+FFFD}
@font-face{font-family:Inter;font-weight:400;font-style:normal;src:url(fonts/inter-latin-400-normal.woff2) format('woff2')}
@font-face{font-family:Inter;font-weight:500;font-style:normal;src:url(fonts/inter-latin-500-normal.woff2) format('woff2')}
@font-face{font-family:Inter;font-weight:600;font-style:normal;src:url(fonts/inter-latin-600-normal.woff2) format('woff2')}
@font-face{font-family:Inter;font-weight:700;font-style:normal;src:url(fonts/inter-latin-700-normal.woff2) format('woff2')}
"""

# --------------------------------------------------------------------------- helpers


def esc(s: str) -> str:
    return _html.escape(s, quote=True)


def hsl_to_hex(h: float, s: float, l: float) -> str:
    import colorsys
    r, g, b = colorsys.hls_to_rgb(h / 360.0, l, s)
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))


def photo(rng: random.Random, w: int, h: int, hue: float | None = None) -> str:
    """Procedural 'photo' (blobs + noise + gradients), JPEG data-URI."""
    nrng = np.random.RandomState(rng.randrange(1 << 30))
    hue = rng.random() if hue is None else hue
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.zeros((h, w, 3), np.float32)
    # smooth vertical gradient
    g = (yy / h)[..., None]
    c1 = np.array([0.35 + 0.3 * hue, 0.5, 0.75 - 0.3 * hue])
    c2 = np.array([0.9 - 0.4 * hue, 0.6 + 0.2 * hue, 0.4])
    base += c1 * (1 - g) + c2 * g
    for _ in range(rng.randint(5, 9)):
        cx, cy = nrng.rand() * w, nrng.rand() * h
        r = (0.1 + 0.25 * nrng.rand()) * max(w, h)
        col = nrng.rand(3) * 0.8
        d = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)))[..., None]
        base = base * (1 - 0.7 * d) + col * 0.7 * d
    base += nrng.randn(h, w, 3).astype(np.float32) * 0.03
    img = np.clip(base * 255, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 72])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


ICON_PATHS = {
    "home": '<path d="M3 11l9-8 9 8v9a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1z"/>',
    "user": '<circle cx="12" cy="8" r="4"/><path d="M4 21c0-4 4-6 8-6s8 2 8 6z"/>',
    "search": '<circle cx="10" cy="10" r="6" fill="none" stroke="currentColor" stroke-width="2.4"/><path d="M15 15l6 6" stroke="currentColor" stroke-width="2.4"/>',
    "bell": '<path d="M6 16V11a6 6 0 0 1 12 0v5l2 2H4z"/><circle cx="12" cy="21" r="2"/>',
    "gear": '<circle cx="12" cy="12" r="8" fill="none" stroke="currentColor" stroke-width="4" stroke-dasharray="3 2.3"/><circle cx="12" cy="12" r="3.2"/>',
    "check": '<path d="M4 12l5 5L20 6" fill="none" stroke="currentColor" stroke-width="3"/>',
    "heart": '<path d="M12 21C4 14 3 9 6.5 6.5 9 4.8 11.4 6 12 8c.6-2 3-3.2 5.5-1.5C21 9 20 14 12 21z"/>',
    "star": '<path d="M12 2l3 7 7.5.6-5.7 5 1.8 7.4L12 18l-6.6 4 1.8-7.4-5.7-5L9 9z"/>',
    "menu": '<path d="M3 6h18M3 12h18M3 18h18" stroke="currentColor" stroke-width="2.6" fill="none"/>',
    "chart": '<path d="M4 20V10M10 20V4M16 20v-8M22 20H2" stroke="currentColor" stroke-width="3" fill="none"/>',
}


def icon(name: str, size: int = 20, color: str = "currentColor") -> str:
    return (f'<svg data-gt="icon" width="{size}" height="{size}" viewBox="0 0 24 24" fill="{color}" '
            f'style="color:{color};flex:none">{ICON_PATHS[name]}</svg>')


@dataclass
class Ctx:
    rng: random.Random
    fa: bool
    dark: bool
    width: int
    p: dict  # palette/params

    # ---- text
    def n(self, k: int | None = None, lo: int = 1, hi: int = 99999) -> str:
        v = self.rng.randint(lo, hi) if k is None else k
        s = f"{v:,}"
        if self.fa:
            return s.replace(",", "٬").translate(FA_DIGITS)
        return s

    def words(self, lo: int, hi: int) -> str:
        pool = FA_WORDS if self.fa else EN_WORDS
        k = self.rng.randint(lo, hi)
        ws = [self.rng.choice(pool) for _ in range(k)]
        s = " ".join(ws)
        return s if self.fa else s.capitalize()

    def sentence(self, lo: int = 5, hi: int = 12) -> str:
        s = self.words(lo, hi)
        return s + ("." if not self.fa else ".")

    def title(self) -> str:
        return self.rng.choice(FA_TITLES if self.fa else EN_TITLES)

    def name(self) -> str:
        return self.rng.choice(FA_NAMES if self.fa else EN_NAMES)

    def price(self) -> str:
        if self.fa:
            return self.n(lo=10, hi=900) + "٬۰۰۰ تومان"
        return "$" + str(self.rng.randint(5, 199)) + ".00"

    def mixed(self) -> str:
        """A Persian line containing Latin words / numbers (bidi stress)."""
        if self.fa:
            lat = self.rng.choice(["AI", "iPhone", "Pro", "VIP", "SMS", "PDF", "API", "Wi-Fi"])
            return f"{self.words(1, 2)} {lat} {self.words(1, 2)} {self.n(lo=2, hi=60)}"
        return self.words(2, 4)

    # ---- elements (every element is GT-tagged)
    def t(self, s: str, tag: str = "p", cls: str = "", style: str = "") -> str:
        c = f' class="{cls}"' if cls else ""
        st = f' style="{style}"' if style else ""
        return f'<{tag} data-gt="text"{c}{st}>{esc(s)}</{tag}>'

    def btn(self, label: str, kind: str = "primary", style: str = "") -> str:
        st = f' style="{style}"' if style else ""
        return f'<button data-gt="button" class="btn btn-{kind}"{st}><span>{esc(label)}</span></button>'

    def inp(self, ph: str, typ: str = "text") -> str:
        return f'<input data-gt="input" class="input" type="{typ}" placeholder="{esc(ph)}">'

    def select(self, label: str) -> str:
        return (f'<div data-gt="input" class="input select"><span>{esc(label)}</span>'
                f'<svg width="12" height="12" viewBox="0 0 12 12"><path d="M2 4l4 4 4-4" fill="none" '
                f'stroke="currentColor" stroke-width="1.6"/></svg></div>')

    def field(self, label: str, ph: str, typ: str = "text") -> str:
        return f'<div class="field">{self.t(label, "label", "lbl")}{self.inp(ph, typ)}</div>'

    def badge(self, s: str, kind: str = "ok") -> str:
        return f'<span data-gt="badge" class="badge badge-{kind}"><span>{esc(s)}</span></span>'

    def check(self, label: str, on: bool = True, kind: str = "check") -> str:
        box = (f'<span data-gt="control" class="ctl {kind}{" on" if on else ""}"></span>')
        return f'<label class="row-ctl">{box}{self.t(label, "span")}</label>'

    def toggle(self, label: str, on: bool = True) -> str:
        return (f'<label class="row-ctl"><span data-gt="control" class="tog{" on" if on else ""}">'
                f'<i></i></span>{self.t(label, "span")}</label>')

    def img(self, w: int, h: int, cls: str = "", radius: int | None = None) -> str:
        st = f"width:{w}px;height:{h}px;" + (f"border-radius:{radius}px;" if radius is not None else "")
        return f'<img data-gt="image" class="photo {cls}" src="{photo(self.rng, w, h)}" style="{st}" alt="">'

    def avatar(self, size: int = 40) -> str:
        return (f'<img data-gt="image" class="avatar" src="{photo(self.rng, size * 2, size * 2)}" '
                f'style="width:{size}px;height:{size}px" alt="">')

    def card(self, inner: str, cls: str = "", style: str = "") -> str:
        st = f' style="{style}"' if style else ""
        return f'<div data-gt="card" class="card {cls}"{st}>{inner}</div>'

    def nav(self, links: list[str], logo: str, extra: str = "") -> str:
        items = "".join(self.t(s, "a", "nav-a") for s in links)
        return (f'<header data-gt="nav" class="topnav"><div class="wrap navrow">'
                f'{self.t(logo, "div", "logo")}<nav class="navlinks">{items}</nav>{extra}</div></header>')

    def footer(self) -> str:
        cols = "".join(
            f'<div class="fcol">{self.t(self.words(1, 2), "h4", "fh")}'
            + "".join(self.t(self.words(1, 2), "a", "fa") for _ in range(3)) + "</div>"
            for _ in range(3))
        return (f'<footer data-gt="nav" class="footer"><div class="wrap frow">{cols}</div>'
                f'<div class="wrap">{self.t(self.sentence(3, 6), "p", "copy")}</div></footer>')


# --------------------------------------------------------------------------- css


def make_css(ctx: Ctx) -> str:
    p = ctx.p
    dark = ctx.dark
    if dark:
        bg, fg, muted = hsl_to_hex(p["hue"], 0.25, 0.09), "#e8ebf0", "#9aa4b2"
        card, border, soft = hsl_to_hex(p["hue"], 0.22, 0.15), hsl_to_hex(p["hue"], 0.18, 0.24), hsl_to_hex(p["hue"], 0.2, 0.19)
    else:
        bg, fg, muted = ("#ffffff" if p["whitebg"] else hsl_to_hex(p["hue"], 0.3, 0.97)), "#1d2433", "#667085"
        card, border, soft = "#ffffff", "#e3e7ee", hsl_to_hex(p["hue"], 0.35, 0.95)
    primary = hsl_to_hex(p["hue"], 0.78, 0.47 if not dark else 0.58)
    primary2 = hsl_to_hex((p["hue"] + 40) % 360, 0.75, 0.5)
    primary_fg = "#ffffff" if not dark else "#0b1020"
    font = "Vazirmatn, sans-serif" if ctx.fa else p["latin_font"]
    r = p["radius"]
    shadow = "0 6px 18px rgba(16,24,40,.10)" if (p["shadow"] and not dark) else ("0 6px 18px rgba(0,0,0,.45)" if p["shadow"] else "none")
    cardborder = f"1px solid {border}" if p["cardborder"] or not p["shadow"] else "1px solid transparent"
    fs = p["fs"]
    return FONT_FACE_CSS + f"""
:root{{--bg:{bg};--fg:{fg};--muted:{muted};--card:{card};--border:{border};--soft:{soft};--primary:{primary};--primary2:{primary2};--pfg:{primary_fg};--r:{r}px;--font:{font};--fs:{fs}px}}
*{{box-sizing:border-box;margin:0;padding:0}}
html{{background:var(--bg)}}
body{{font-family:var(--font);background:var(--bg);color:var(--fg);font-size:var(--fs);line-height:1.65}}
a{{color:inherit;text-decoration:none}}
.wrap{{max-width:1160px;margin:0 auto;padding:0 {p['gutter']}px}}
.topnav{{background:var(--card);border-bottom:1px solid var(--border)}}
.navrow{{display:flex;align-items:center;gap:32px;height:68px}}
.logo{{font-weight:700;font-size:22px;color:var(--primary)}}
.navlinks{{display:flex;gap:26px;flex:1}}
.nav-a{{color:var(--muted);font-weight:500;font-size:15px}}
.btn{{border:0;cursor:pointer;font-family:inherit;font-size:15px;font-weight:600;padding:{p['btnpad'][0]}px {p['btnpad'][1]}px;border-radius:var(--r);display:inline-flex;align-items:center;justify-content:center;line-height:1.4}}
.btn-primary{{background:var(--primary);color:var(--pfg)}}
.btn-grad{{background:linear-gradient(135deg,var(--primary),var(--primary2));color:#fff}}
.btn-outline{{background:transparent;color:var(--primary);border:1.5px solid var(--primary)}}
.btn-ghost{{background:var(--soft);color:var(--fg)}}
.btn-danger{{background:#d92d20;color:#fff}}
.input{{display:flex;align-items:center;width:100%;height:44px;padding:0 14px;border:1px solid var(--border);border-radius:var(--r);background:var(--card);color:var(--fg);font-family:inherit;font-size:15px}}
.input::placeholder{{color:var(--muted);opacity:1}}
.select{{justify-content:space-between;color:var(--muted)}}
.field{{display:flex;flex-direction:column;gap:7px}}
.lbl{{font-size:14px;font-weight:500;color:var(--fg)}}
.card{{background:var(--card);border:{cardborder};border-radius:calc(var(--r) + 2px);box-shadow:{shadow};padding:{p['cardpad']}px}}
.badge{{display:inline-flex;align-items:center;padding:2px 11px;border-radius:99px;font-size:12.5px;font-weight:600;line-height:1.6}}
.badge-ok{{background:#12b76a22;color:#12b76a}}
.badge-warn{{background:#f7900922;color:#dc6803}}
.badge-bad{{background:#f0443822;color:#f04438}}
.badge-info{{background:color-mix(in srgb,var(--primary) 16%,transparent);color:var(--primary)}}
.row-ctl{{display:flex;align-items:center;gap:10px;font-size:15px}}
.ctl{{width:20px;height:20px;border:1.5px solid var(--border);background:var(--card);display:inline-block;position:relative;flex:none}}
.ctl.check{{border-radius:5px}}
.ctl.radio{{border-radius:50%}}
.ctl.on{{background:var(--primary);border-color:var(--primary)}}
.ctl.on::after{{content:"";position:absolute;inset:5px;background:#fff;border-radius:{ '1px' if True else '0' }}}
.ctl.radio.on::after{{border-radius:50%}}
.tog{{width:42px;height:24px;border-radius:99px;background:var(--border);display:inline-block;position:relative;flex:none}}
.tog.on{{background:var(--primary)}}
.tog i{{position:absolute;top:3px;inset-inline-start:3px;width:18px;height:18px;border-radius:50%;background:#fff}}
.tog.on i{{inset-inline-start:21px}}
.photo{{display:block;object-fit:cover;border-radius:var(--r)}}
.avatar{{border-radius:50%;display:block;object-fit:cover}}
h1{{font-size:{p['h1']}px;line-height:1.3;font-weight:700}}
h2{{font-size:{p['h2']}px;line-height:1.35;font-weight:700}}
h3{{font-size:19px;line-height:1.4;font-weight:600}}
.muted{{color:var(--muted)}}
.small{{font-size:13.5px}}
.sec{{padding:{p['secpad']}px 0}}
.grid{{display:grid;gap:{p['gap']}px}}
.footer{{background:var(--soft);padding:36px 0 22px;margin-top:40px;border-top:1px solid var(--border)}}
.frow{{display:flex;gap:60px;padding-bottom:20px}}
.fcol{{display:flex;flex-direction:column;gap:8px}}
.fh{{font-size:15px;font-weight:700;margin-bottom:4px}}
.fa{{font-size:14px;color:var(--muted)}}
.copy{{font-size:13px;color:var(--muted)}}
table{{border-collapse:collapse;width:100%}}
th{{text-align:start;font-size:13px;font-weight:600;color:var(--muted);padding:12px 16px;background:var(--soft);border-bottom:1px solid var(--border)}}
td{{padding:13px 16px;border-bottom:1px solid var(--border);font-size:14.5px}}
.hero{{padding:{p['secpad']+20}px 0}}
.sidebar{{width:250px;background:var(--card);border-inline-end:1px solid var(--border);padding:20px 14px;display:flex;flex-direction:column;gap:6px;flex:none}}
.side-a{{display:flex;align-items:center;gap:12px;padding:10px 14px;border-radius:var(--r);color:var(--muted);font-weight:500;font-size:15px}}
.side-a.act{{background:color-mix(in srgb,var(--primary) 14%,transparent);color:var(--primary)}}
.shell{{display:flex;min-height:100vh}}
.main{{flex:1;padding:26px 30px;display:flex;flex-direction:column;gap:22px;min-width:0}}
.stat{{display:flex;flex-direction:column;gap:6px}}
.stat-n{{font-size:30px;font-weight:700;line-height:1.2}}
.hr{{height:1px;background:var(--border)}}
.list{{display:flex;flex-direction:column;gap:14px}}
.pad-center{{display:flex;align-items:center;justify-content:center}}
.price-n{{font-size:38px;font-weight:700;line-height:1.2}}
"""


# --------------------------------------------------------------------------- page kinds


def _shell(ctx: Ctx, body: str, title: str = "") -> str:
    direction = "rtl" if ctx.fa else "ltr"
    lang = "fa" if ctx.fa else "en"
    return (f'<!doctype html><html lang="{lang}" dir="{direction}"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{esc(title or "bench")}</title><style>{make_css(ctx)}</style></head>'
            f'<body>{body}</body></html>')


def k_landing(c: Ctx) -> str:
    p = c.p
    links = [c.words(1, 1) for _ in range(4)]
    nav = c.nav(links, c.rng.choice(["NM", "Chagher", "Acme", "Nova"]) if not c.fa else c.rng.choice(["نوا", "آسمان", "پارس", "ایده"]),
                c.btn(c.words(1, 2), "primary"))
    if p["hero_grad"]:
        g1, g2 = hsl_to_hex(p["hue"], 0.7, 0.42), hsl_to_hex((p["hue"] + 50) % 360, 0.7, 0.35)
        hero_bg = f"background:linear-gradient(135deg,{g1},{g2});color:#fff"
        ghost = "ghost"
    else:
        hero_bg, ghost = "", "outline"
    hero_text = (f'<div style="flex:1;display:flex;flex-direction:column;gap:18px">'
                 f'{c.t(c.title(), "h1")}{c.t(c.sentence(10, 16), "p", "muted", "font-size:17px" if not p["hero_grad"] else "font-size:17px;opacity:.92")}'
                 f'<div style="display:flex;gap:12px;margin-top:6px">{c.btn(c.words(1, 2), "primary" if not p["hero_grad"] else "ghost")}{c.btn(c.words(1, 2), ghost)}</div></div>')
    hero_img = c.img(480, 300, radius=18)
    hero = (f'<section class="hero" style="{hero_bg}"><div class="wrap" style="display:flex;align-items:center;gap:48px">'
            f'{hero_text}{hero_img}</div></section>')
    cols = c.rng.choice([3, 3, 4])
    feats = ""
    for i in range(cols):
        ic = c.rng.choice(list(ICON_PATHS))
        feats += c.card(f'<div style="display:flex;flex-direction:column;gap:12px">'
                        f'<div style="width:46px;height:46px;border-radius:12px;background:var(--soft);display:flex;align-items:center;justify-content:center;color:var(--primary)">{icon(ic, 22, "var(--primary)")}</div>'
                        f'{c.t(c.words(1, 3), "h3")}{c.t(c.sentence(8, 14), "p", "muted small")}</div>')
    features = (f'<section class="sec"><div class="wrap"><div style="text-align:center;margin-bottom:34px;display:flex;flex-direction:column;gap:10px;align-items:center">'
                f'{c.t(c.title(), "h2")}{c.t(c.sentence(6, 10), "p", "muted")}</div>'
                f'<div class="grid" style="grid-template-columns:repeat({cols},1fr)">{feats}</div></div></section>')
    stats = '<section class="sec" style="padding-top:0"><div class="wrap"><div class="grid" style="grid-template-columns:repeat(4,1fr);text-align:center">'
    for _ in range(4):
        stats += f'<div class="stat" style="align-items:center">{c.t(c.n(), "div", "stat-n")}{c.t(c.words(1, 2), "div", "muted small")}</div>'
    stats += "</div></div></section>"
    cta = (f'<section class="sec"><div class="wrap">{c.card(f"""<div style="display:flex;align-items:center;justify-content:space-between;gap:24px"><div style="display:flex;flex-direction:column;gap:6px">{c.t(c.title(), "h2")}{c.t(c.sentence(6, 10), "p", "muted")}</div>{c.btn(c.words(1, 2), "primary")}</div>""", style="padding:34px")}</div></section>')
    parts = [hero, features, stats, cta] if c.rng.random() < 0.5 else [hero, stats, features, cta]
    return nav + "".join(parts) + c.footer()


def k_dashboard(c: Ctx) -> str:
    side_items = ""
    icons = list(ICON_PATHS)
    for i in range(7):
        act = " act" if i == 0 else ""
        side_items += f'<div class="side-a{act}" data-gt="nav-item">{icon(icons[i % len(icons)], 20)}{c.t(c.words(1, 2), "span")}</div>'
    sidebar = (f'<aside data-gt="nav" class="sidebar">{c.t(c.rng.choice(["نوا", "آسمان"]) if c.fa else "Nova", "div", "logo", "padding:8px 14px 18px")}'
               f'{side_items}</aside>')
    top = (f'<div style="display:flex;align-items:center;justify-content:space-between">'
           f'<div style="display:flex;flex-direction:column">{c.t(c.title(), "h2")}{c.t(c.sentence(4, 8), "p", "muted small")}</div>'
           f'<div style="display:flex;align-items:center;gap:14px">{icon("bell", 22, "var(--muted)")}{c.avatar(40)}{c.btn(c.words(1, 2), "primary")}</div></div>')
    stats = '<div class="grid" style="grid-template-columns:repeat(4,1fr)">'
    for i in range(4):
        stats += c.card(f'<div class="stat">{c.t(c.words(1, 2), "div", "muted small")}{c.t(c.n(), "div", "stat-n")}'
                        f'{c.badge(c.n(lo=1, hi=40) + ("٪" if c.fa else "%"), c.rng.choice(["ok", "bad", "info"]))}</div>')
    stats += "</div>"
    chart = c.card(f'{c.t(c.words(1, 3), "h3", "", "margin-bottom:14px")}{c.img(560, 220, radius=8)}', style="flex:2")
    side_list = ""
    for _ in range(4):
        side_list += (f'<div style="display:flex;align-items:center;gap:12px">{c.avatar(38)}<div style="display:flex;flex-direction:column;flex:1">'
                      f'{c.t(c.name(), "span", "", "font-weight:600;font-size:14.5px")}{c.t(c.words(1, 2), "span", "muted small")}</div>{c.badge(c.words(1, 1), c.rng.choice(["ok", "warn", "info"]))}</div>')
    people = c.card(f'{c.t(c.words(1, 2), "h3", "", "margin-bottom:14px")}<div class="list">{side_list}</div>', style="flex:1")
    row = f'<div style="display:flex;gap:22px;align-items:stretch">{chart}{people}</div>'
    tbl = _table_card(c, 5, 4)
    return f'<div class="shell">{sidebar}<main class="main">{top}{stats}{row}{tbl}</main></div>'


def _table_card(c: Ctx, rows: int, cols: int, title: bool = True) -> str:
    head = "".join(f'<th>{c.t(c.words(1, 2), "span")}</th>' for _ in range(cols))
    body = ""
    for i in range(rows):
        cells = f'<td>{c.t(c.name(), "span")}</td>'
        for j in range(cols - 2):
            cells += f'<td>{c.t(c.n(), "span")}</td>' if j % 2 == 0 else f'<td>{c.t(c.words(1, 2), "span")}</td>'
        cells += f'<td>{c.badge(c.words(1, 1), c.rng.choice(["ok", "warn", "bad"]))}</td>'
        body += f"<tr>{cells}</tr>"
    inner = (c.t(c.title(), "h3", "", "padding:18px 16px 12px") if title else "") + \
        f'<table data-gt="table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'
    return c.card(inner, style="padding:0;overflow:hidden")


def k_table(c: Ctx) -> str:
    nav = c.nav([c.words(1, 1) for _ in range(5)], "Admin" if not c.fa else "مدیریت", c.avatar(36))
    filt = (f'<div style="display:flex;gap:12px;align-items:center">{c.inp(c.words(1, 2))}{c.select(c.words(1, 1))}{c.btn(c.words(1, 1), "primary")}</div>')
    pag = '<div style="display:flex;gap:8px;justify-content:center;margin-top:6px">' + "".join(
        c.btn(c.n(k=i + 1), "primary" if i == 0 else "ghost", "padding:6px 14px") for i in range(5)) + "</div>"
    return (nav + f'<div class="wrap" style="padding-top:28px;display:flex;flex-direction:column;gap:20px">'
            f'{c.t(c.title(), "h1", "", "font-size:28px")}{filt}{_table_card(c, 9, 5, False)}{pag}</div>' + c.footer())


def k_form(c: Ctx) -> str:
    nav = c.nav([c.words(1, 1) for _ in range(3)], "Forms" if not c.fa else "فرم")
    two = (f'<div class="grid" style="grid-template-columns:1fr 1fr">{c.field(c.words(1, 2), c.words(1, 2))}{c.field(c.words(1, 2), c.words(1, 2))}</div>')
    sel = f'<div class="field">{c.t(c.words(1, 2), "label", "lbl")}{c.select(c.words(1, 2))}</div>'
    ta = (f'<div class="field">{c.t(c.words(1, 2), "label", "lbl")}'
          f'<div data-gt="input" class="input" style="height:110px;align-items:flex-start;padding-top:12px;color:var(--muted)"><span>{esc(c.words(2, 4))}</span></div></div>')
    checks = (f'<div style="display:flex;flex-direction:column;gap:12px">{c.check(c.words(1, 3))}{c.check(c.words(1, 3), False)}'
              f'<div style="display:flex;gap:24px">{c.check(c.words(1, 1), True, "radio")}{c.check(c.words(1, 1), False, "radio")}</div>{c.toggle(c.words(1, 3))}</div>')
    form = (f'<div style="display:flex;flex-direction:column;gap:20px">{c.t(c.title(), "h2")}{c.t(c.sentence(6, 10), "p", "muted")}'
            f'{two}{c.field(c.words(1, 2), "name@example.com", "email")}{sel}{ta}{checks}'
            f'<div style="display:flex;gap:12px">{c.btn(c.words(1, 2), "primary")}{c.btn(c.words(1, 1), "ghost")}</div></div>')
    return nav + f'<div class="wrap" style="max-width:720px;padding-top:34px;padding-bottom:30px">{c.card(form, style="padding:34px")}</div>' + c.footer()


def k_pricing(c: Ctx) -> str:
    nav = c.nav([c.words(1, 1) for _ in range(4)], "Plans" if not c.fa else "طرح‌ها", c.btn(c.words(1, 1), "outline"))
    cards = ""
    for i in range(3):
        feats = "".join(
            f'<div style="display:flex;gap:10px;align-items:center">{icon("check", 18, "var(--primary)")}{c.t(c.words(1, 3), "span", "small")}</div>'
            for _ in range(5))
        hot = i == 1
        inner = (f'<div style="display:flex;flex-direction:column;gap:14px">'
                 f'{c.badge(c.words(1, 1), "info") if hot else ""}{c.t(c.words(1, 2), "h3")}'
                 f'<div style="display:flex;align-items:baseline;gap:6px">{c.t(c.price(), "div", "price-n")}</div>'
                 f'{c.t(c.sentence(4, 7), "p", "muted small")}<div class="hr"></div><div class="list" style="gap:10px">{feats}</div>'
                 f'{c.btn(c.words(1, 2), "primary" if hot else "outline", "margin-top:8px;width:100%")}</div>')
        cards += c.card(inner, "", "border:2px solid var(--primary);padding:30px" if hot else "padding:30px")
    return (nav + f'<div class="wrap sec"><div style="text-align:center;margin-bottom:34px;display:flex;flex-direction:column;gap:10px;align-items:center">{c.t(c.title(), "h1")}{c.t(c.sentence(7, 11), "p", "muted")}</div>'
            f'<div class="grid" style="grid-template-columns:repeat(3,1fr);align-items:start">{cards}</div></div>' + c.footer())


def k_login(c: Ctx) -> str:
    box = (f'<div style="display:flex;flex-direction:column;gap:18px;width:400px">'
           f'<div style="text-align:center;display:flex;flex-direction:column;gap:8px;align-items:center">{c.t("NM" if not c.fa else "نوا", "div", "logo", "font-size:30px")}{c.t(c.title(), "h2")}{c.t(c.sentence(4, 7), "p", "muted small")}</div>'
           f'{c.field(c.words(1, 2), "name@example.com", "email")}{c.field(c.words(1, 2), "••••••••", "password")}'
           f'<div style="display:flex;justify-content:space-between;align-items:center">{c.check(c.words(1, 2))}{c.t(c.words(1, 2), "a", "small", "color:var(--primary)")}</div>'
           f'{c.btn(c.words(1, 2), "primary", "width:100%;height:46px")}'
           f'<div style="display:flex;align-items:center;gap:12px"><div class="hr" style="flex:1"></div>{c.t(c.words(1, 1), "span", "muted small")}<div class="hr" style="flex:1"></div></div>'
           f'{c.btn(c.words(1, 2), "outline", "width:100%;height:46px")}</div>')
    bg = f'min-height:100vh;display:flex;align-items:center;justify-content:center;padding:40px;background:var(--soft)'
    return f'<div style="{bg}">{c.card(box, style="padding:38px")}</div>'


def k_products(c: Ctx) -> str:
    nav = c.nav([c.words(1, 1) for _ in range(5)], "Shop" if not c.fa else "فروشگاه", f'<div style="width:220px">{c.inp(c.words(1, 1))}</div>')
    cols = c.rng.choice([3, 4, 4])
    cards = ""
    for i in range(cols * 2):
        inner = (f'{c.img(240, 170, radius=8)}<div style="display:flex;flex-direction:column;gap:6px;margin-top:12px">'
                 f'{c.t(c.words(1, 3), "h3", "", "font-size:16px")}'
                 f'<div style="display:flex;gap:4px;align-items:center">{"".join(icon("star", 14, "#f5a623") for _ in range(5))}</div>'
                 f'{c.t(c.price(), "div", "", "font-weight:700;font-size:17px")}'
                 f'{c.btn(c.words(1, 2), "primary", "margin-top:6px;width:100%;padding:9px")}</div>')
        cards += c.card(inner, "", "padding:14px")
    return (nav + f'<div class="wrap" style="padding-top:28px;display:flex;flex-direction:column;gap:22px">'
            f'<div style="display:flex;justify-content:space-between;align-items:center">{c.t(c.title(), "h1", "", "font-size:28px")}{c.select(c.words(1, 2))}</div>'
            f'<div class="grid" style="grid-template-columns:repeat({cols},1fr)">{cards}</div></div>' + c.footer())


def k_blog(c: Ctx) -> str:
    nav = c.nav([c.words(1, 1) for _ in range(4)], "Blog" if not c.fa else "وبلاگ")
    serif = c.p.get("serif")
    hfont = "font-family:Georgia,'DejaVu Serif',serif;" if serif else ""
    paras = "".join(c.t(c.sentence(30, 46), "p", "", "margin-bottom:16px;font-size:17px;line-height:1.95") for _ in range(4))
    art = (f'<article style="flex:1;min-width:0">{c.t(c.title(), "h1", "", hfont + "margin-bottom:10px")}'
           f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:22px">{c.avatar(38)}{c.t(c.name(), "span", "", "font-weight:600")}{c.t(c.n(lo=1, hi=28) + " " + ("آبان" if c.fa else "Nov"), "span", "muted small")}</div>'
           f'{c.img(700, 300, radius=12)}<div style="height:20px"></div>{paras}</article>')
    related = "".join(
        f'<div style="display:flex;gap:12px;align-items:center">{c.img(70, 70, radius=8)}{c.t(c.words(2, 4), "span", "", "font-weight:600;font-size:14.5px")}</div>'
        for _ in range(4))
    side = f'<aside style="width:290px;flex:none;display:flex;flex-direction:column;gap:20px">{c.card(c.t(c.words(1, 2), "h3", "", "margin-bottom:14px") + f"<div class=list>{related}</div>")}{c.card(c.t(c.words(1, 2), "h3", "", "margin-bottom:12px") + c.inp(c.words(1, 2)) + c.btn(c.words(1, 2), "primary", "margin-top:10px;width:100%"))}</aside>'
    return nav + f'<div class="wrap" style="display:flex;gap:40px;padding-top:30px;align-items:flex-start">{art}{side}</div>' + c.footer()


def k_mobile(c: Ctx) -> str:
    top = (f'<header data-gt="nav" style="display:flex;align-items:center;justify-content:space-between;padding:14px 16px;background:var(--card);border-bottom:1px solid var(--border)">'
           f'{icon("menu", 24, "var(--fg)")}{c.t("NM" if not c.fa else "نوا", "div", "logo")}{icon("search", 22, "var(--fg)")}</header>')
    hero = (f'<div style="padding:18px 16px;display:flex;flex-direction:column;gap:12px">{c.img(326, 150, radius=14)}'
            f'{c.t(c.title(), "h2", "", "font-size:22px")}{c.t(c.sentence(8, 12), "p", "muted small")}{c.btn(c.words(1, 2), "primary", "width:100%")}</div>')
    items = ""
    for _ in range(4):
        items += c.card(f'<div style="display:flex;gap:12px;align-items:center">{c.avatar(46)}<div style="display:flex;flex-direction:column;flex:1;min-width:0">'
                        f'{c.t(c.name(), "span", "", "font-weight:600")}{c.t(c.words(2, 4), "span", "muted small")}</div>{c.badge(c.n(lo=1, hi=9), "info")}</div>', "", "padding:12px")
    bottom = ('<nav data-gt="nav" style="position:sticky;bottom:0;display:flex;justify-content:space-around;padding:10px 6px;background:var(--card);border-top:1px solid var(--border);margin-top:24px">'
              + "".join(f'<div style="display:flex;flex-direction:column;align-items:center;gap:2px">{icon(ic, 22, "var(--primary)" if i == 0 else "var(--muted)")}{c.t(c.words(1, 1), "span", "small", "font-size:11.5px")}</div>'
                        for i, ic in enumerate(["home", "search", "heart", "user"])) + "</nav>")
    return top + hero + f'<div style="padding:0 16px;display:flex;flex-direction:column;gap:12px">{items}</div>' + bottom


def k_settings(c: Ctx) -> str:
    items = "".join(f'<div class="side-a{" act" if i == 1 else ""}" data-gt="nav-item">{icon(list(ICON_PATHS)[i], 20)}{c.t(c.words(1, 2), "span")}</div>' for i in range(6))
    side = f'<aside data-gt="nav" class="sidebar" style="width:260px">{items}</aside>'
    rows = ""
    for i in range(4):
        rows += (f'<div style="display:flex;justify-content:space-between;align-items:center;padding:16px 0;border-bottom:1px solid var(--border)">'
                 f'<div style="display:flex;flex-direction:column">{c.t(c.words(1, 3), "span", "", "font-weight:600")}{c.t(c.sentence(5, 9), "span", "muted small")}</div>{c.toggle("", i % 2 == 0)}</div>')
    prof = (f'<div style="display:flex;align-items:center;gap:18px">{c.avatar(72)}<div style="display:flex;flex-direction:column;gap:8px">'
            f'{c.t(c.name(), "h3")}{c.btn(c.words(1, 2), "outline", "padding:6px 16px")}</div></div>'
            f'<div class="grid" style="grid-template-columns:1fr 1fr;margin-top:22px">{c.field(c.words(1, 2), c.name())}{c.field(c.words(1, 2), "name@example.com", "email")}</div>')
    main = (f'<main class="main">{c.t(c.title(), "h2")}{c.card(prof)}{c.card(c.t(c.words(1, 3), "h3") + rows)}'
            f'<div style="display:flex;gap:12px">{c.btn(c.words(1, 2), "primary")}{c.btn(c.words(1, 1), "ghost")}{c.btn(c.words(1, 2), "danger")}</div></main>')
    return f'<div class="shell">{side}{main}</div>'


KIND_FN = {
    "landing": k_landing, "dashboard": k_dashboard, "table": k_table, "form": k_form,
    "pricing": k_pricing, "login": k_login, "products": k_products, "blog": k_blog,
    "mobile": k_mobile, "settings": k_settings,
}


def make_page(kind: str, lang: str, theme: str, seed: int) -> dict:
    rng = random.Random(f"{kind}-{lang}-{theme}-{seed}")
    width = 390 if kind == "mobile" else rng.choice([1280, 1280, 1366, 1440])
    p = dict(
        hue=rng.choice([215, 160, 265, 340, 25, 190, 140, 10]),
        radius=rng.choice([0, 4, 8, 12, 16, 22]),
        shadow=rng.random() < 0.55,
        cardborder=rng.random() < 0.6,
        whitebg=rng.random() < 0.5,
        gutter=24, cardpad=rng.choice([18, 22, 26]), gap=rng.choice([16, 20, 24, 28]),
        secpad=rng.choice([44, 56, 70]), fs=rng.choice([15, 16, 16, 17]),
        h1=rng.choice([38, 44, 48]), h2=rng.choice([26, 30, 32]),
        btnpad=rng.choice([(9, 18), (10, 22), (12, 26)]),
        hero_grad=rng.random() < 0.4,
        latin_font=rng.choice(["Inter, sans-serif"] * 4 + ["Vazirmatn, sans-serif"]),
        serif=(kind == "blog" and seed % 2 == 1 and lang == "en"),
    )
    if kind == "mobile":
        p["gutter"] = 16
    ctx = Ctx(rng=rng, fa=lang == "fa", dark=theme == "dark", width=width, p=p)
    body = KIND_FN[kind](ctx)
    return dict(id=f"{kind}-{lang}-{theme}-{seed}", kind=kind, lang=lang, theme=theme, seed=seed,
                width=width, html=_shell(ctx, body, kind))


def page_specs(full: bool = True) -> list[tuple]:
    """Deterministic list of (kind, lang, theme, seed)."""
    specs = []
    seeds = [0, 1] if full else [0]
    for seed in seeds:
        for kind in KINDS:
            for lang in LANGS:
                for theme in THEMES:
                    specs.append((kind, lang, theme, seed))
    if not full:
        # quick: a balanced subset of 16 pages
        keep = []
        for kind in KINDS:
            keep.append((kind, "fa", "light", 0))
            keep.append((kind, "en", "dark", 0))
        keep += [("dashboard", "fa", "dark", 0), ("landing", "en", "light", 0),
                 ("form", "fa", "dark", 0), ("table", "en", "light", 0)]
        specs = keep[:16] if False else keep
    return specs
