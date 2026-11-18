"""OCR backends.

The converter finds text lines itself (``textdet``) and only asks an engine to *read* the
crops, so engines are interchangeable:

* ``tesseract``  (default; Tesseract 4/5 with ``fas`` + ``eng``; ~40 MB, light on CPU/RAM)
* ``easyocr``    (optional extra ``[ocr-easyocr]``; needs torch - heavy)
* ``external``   (pass ``ocr_lines=[...]`` to ``convert`` with boxes/text from any other engine)
* ``none``       (no text: layout only)
"""
from __future__ import annotations

import os
import re
import shutil
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import cv2
import numpy as np

from .imaging import Box, crop, mode_color

# Tesseract spawns OpenMP threads per call; on small machines that makes it ~5x slower.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")


@dataclass
class OcrResult:
    text: str
    conf: float          # 0..1
    words: int = 0
    # Tesseract only: the words behind ``text`` (``(line, par, block, left, text, conf, width, top, height)``, positions relative to ``src``) and the
    # crop they were read from - the second pass re-reads single doubtful words (``PDF``, ``100%``) from it
    ws: Optional[list] = field(default=None, repr=False, compare=False)
    src: Optional[np.ndarray] = field(default=None, repr=False, compare=False)


# --------------------------------------------------------------------------- crop preparation
def prepare_crop(img: np.ndarray, box: Box, target_h: float = 34.0, max_scale: float = 4.0) -> np.ndarray:
    """Crop one text line and turn it into 'dark text on white' grey (any theme), upscaled."""
    H, W = img.shape[:2]
    pad = 3 + int(0.12 * box[3])
    c = crop(img, box, pad)
    if c.size == 0:
        return np.full((12, 12), 255, np.uint8)
    ring = np.concatenate([c[0], c[-1], c[:, 0], c[:, -1]])
    bg = np.array(mode_color(ring), np.int16)
    dist = np.abs(c.astype(np.int16) - bg).max(axis=2).astype(np.float32)
    hi = max(30.0, float(np.percentile(dist, 99.7)))
    gray = 255.0 - np.clip(dist * (255.0 / hi), 0, 255)
    gray = gray.astype(np.uint8)
    h = max(1, box[3])
    scale = float(np.clip(target_h / h, 1.0, max_scale))
    if scale > 1.05:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(gray, 8, 8, 12, 12, cv2.BORDER_CONSTANT, value=255)


# --------------------------------------------------------------------------- backends
class Backend:
    name = "base"

    def available(self) -> bool:
        return False

    def read(self, crops: Sequence[np.ndarray], lang: str = "auto") -> List[OcrResult]:
        raise NotImplementedError

    def read_boxes(self, crops: Sequence[np.ndarray], boxes: Sequence[Box], lang: str = "auto") -> List[OcrResult]:
        """Same as ``read`` but also gets the page box of every crop (backends that were fed results from elsewhere match by box)."""
        return self.read(crops, lang)


class TesseractBackend(Backend):
    name = "tesseract"
    _lock = threading.Lock()

    def __init__(self, langs: Optional[str] = None, psm: int = 6, oem: int = 1):
        self.psm, self.oem = psm, oem
        self._langs = langs

    def available(self) -> bool:
        if shutil.which("tesseract") is None:
            return False
        try:
            import pytesseract  # noqa: F401
        except Exception:
            return False
        return True

    def installed_langs(self) -> List[str]:
        import pytesseract
        try:
            return list(pytesseract.get_languages(config=""))
        except Exception:
            return []

    def _lang_arg(self, lang: str) -> str:
        have = set(self.installed_langs())
        if self._langs:
            return self._langs
        if lang == "en" or "fas" not in have:
            return "eng" if "eng" in have or not have else "+".join(sorted(have)[:1])
        if lang == "fa":
            return "fas+eng" if "eng" in have else "fas"
        return "fas+eng" if "eng" in have else "fas"

    _DIGITS = "۰۱۲۳۴۵۶۷۸۹٬٫,.٪%0123456789"

    def _read_line(self, crop: np.ndarray, la: str, psm: int = 7) -> Optional["OcrResult"]:
        """One crop = one text line, read on its own (``--psm 7``)."""
        import pytesseract
        from pytesseract import Output
        if crop.shape[0] < 8 or crop.shape[1] < 8:
            return None
        with self._lock:
            dd = pytesseract.image_to_data(crop, lang=la, config=f"--oem {self.oem} --psm {psm} -c preserve_interword_spaces=0", output_type=Output.DICT)
        ws = []
        for i in range(len(dd["text"])):
            t = (dd["text"][i] or "").strip()
            try:
                cf = float(dd["conf"][i])
            except Exception:
                cf = -1
            if t and cf >= 0:
                ws.append((dd["line_num"][i], dd["par_num"][i], dd["block_num"][i], dd["left"][i], t, cf, dd["width"][i], dd["top"][i], dd["height"][i]))
        if not ws:
            return None
        r = self._assemble(ws)
        r.src = crop
        return r

    def _read_word(self, crop: np.ndarray, la: str, psm: int = 8, whitelist: str = ""):
        """One word crop -> (text, conf 0..100)."""
        import pytesseract
        from pytesseract import Output
        cfg = f"--oem {self.oem} --psm {psm}" + (f" -c tessedit_char_whitelist={whitelist}" if whitelist else "")
        with self._lock:
            dd = pytesseract.image_to_data(crop, lang=la, config=cfg, output_type=Output.DICT)
        toks = [((dd["text"][i] or "").strip(), float(dd["conf"][i])) for i in range(len(dd["text"])) if (dd["text"][i] or "").strip() and float(dd["conf"][i]) >= 0]
        if not toks:
            return "", 0.0
        return "".join(t for t, _ in toks), sum(c for _, c in toks) / len(toks)

    def rescan(self, img, boxes, crops, results, lang: str = "auto"):
        """Second pass over the doubtful lines (see ``rescan.py``); ``results`` of the lines it is happy with are returned unchanged."""
        from . import rescan as _rs
        return _rs.rescan(self, img, boxes, crops, results, lang, prepare=lambda im, b, th: prepare_crop(im, b, target_h=th, max_scale=5.0))

    def _sheet(self, crops, la: str, cfg: str):
        """One tesseract run over all ``crops`` stacked into a tall sheet -> list[OcrResult | None]."""
        import pytesseract
        from pytesseract import Output
        GAP = 46
        W = max(c.shape[1] for c in crops) + 24 if crops else 1
        ys, y = [], 12
        for c in crops:
            ys.append((y, y + c.shape[0]))
            y += c.shape[0] + GAP
        sheet = np.full((y, W), 255, np.uint8)
        for c, (y0, y1) in zip(crops, ys):
            sheet[y0:y1, 12:12 + c.shape[1]] = c
        with self._lock:
            d = pytesseract.image_to_data(sheet, lang=la, config=cfg, output_type=Output.DICT)
        starts = np.array([a for a, _ in ys])
        per: List[List[tuple]] = [[] for _ in crops]
        for i in range(len(d["text"])):
            t = (d["text"][i] or "").strip()
            try:
                cf = float(d["conf"][i])
            except Exception:
                cf = -1
            if not t or cf < 0:
                continue
            cy = d["top"][i] + d["height"][i] / 2.0
            k = int(np.searchsorted(starts, cy, side="right") - 1)
            if 0 <= k < len(crops):
                per[k].append((d["line_num"][i], d["par_num"][i], d["block_num"][i], d["left"][i] - 12, t, cf, d["width"][i], d["top"][i] - ys[k][0], d["height"][i]))
        out = []
        for k, ws in enumerate(per):
            r = self._assemble(ws) if ws else None
            if r is not None:
                r.src = crops[k]
            out.append(r)
        return out

    def read(self, crops, lang: str = "auto") -> List[OcrResult]:
        import pytesseract
        from pytesseract import Output
        la = self._lang_arg(lang)
        base = f"--oem {self.oem} --psm {self.psm} -c preserve_interword_spaces=0"
        out: List[Optional[OcrResult]] = self._sheet(crops, la, base) if crops else []
        # --- crops the stitched run lost are read one by one as a single line (psm 7)
        for k, r in enumerate(out):
            if r is not None:
                continue
            c = crops[k]
            if c.shape[0] < 8 or c.shape[1] < 8:
                out[k] = OcrResult("", 0.0)
                continue
            with self._lock:
                dd = pytesseract.image_to_data(c, lang=la, config=f"--oem {self.oem} --psm 7", output_type=Output.DICT)
            ws = []
            for i in range(len(dd["text"])):
                t = (dd["text"][i] or "").strip()
                try:
                    cf = float(dd["conf"][i])
                except Exception:
                    cf = -1
                if t and cf >= 0:
                    ws.append((dd["line_num"][i], dd["par_num"][i], dd["block_num"][i], dd["left"][i], t, cf, dd["width"][i], dd["top"][i], dd["height"][i]))
            out[k] = self._assemble(ws) if ws else OcrResult("", 0.0)
            out[k].src = c
        # --- retry ladder for weak lines (bold Persian digits, tiny words): bigger crop, Persian-only model,
        #     digit whitelist; keep whichever reading is the most confident
        low = [k for k, r in enumerate(out)
               if r.conf < 0.55 or (lang != "en" and r.text and not self._AR.search(r.text) and not any(ch.isdigit() for ch in r.text) and r.conf < 0.82)]
        if low and "fas" in set(self.installed_langs()):
            big = []
            for k in low:
                c = crops[k]
                big.append(cv2.resize(c, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC))
            tries = [("fas", base), ("fas", base + " -c tessedit_char_whitelist=" + self._DIGITS)]
            for tl, cfg in tries:
                try:
                    rr = self._sheet(big, tl, cfg)
                except Exception:
                    continue
                for k, r in zip(low, rr):
                    if r is not None and r.conf > out[k].conf + 0.05 and r.text.strip():
                        if "whitelist" in cfg and not any(ch.isdigit() for ch in r.text):
                            continue
                        out[k] = r
        return [r or OcrResult("", 0.0) for r in out]

    _AR = re.compile("[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
    _AR_LETTER = re.compile("[ء-يٱ-ۓݐ-ݿﭐ-﷿ﹰ-﻿]")        # Arabic-script LETTERS (no digits: the Persian digits lie inside the range of _AR)
    _LAT = re.compile("[A-Za-z]")

    @classmethod
    def _drop_junk_words(cls, ws):
        """In a Persian line, Latin-looking words with a low confidence are almost always icons or fragments misread as letters (``# امکانات``,
        ``als &``): drop them.  Real Latin words (PDF, a brand) are read with a high confidence and stay."""
        ar = sum(len(cls._AR.findall(w[4])) for w in ws)
        la = sum(len(cls._LAT.findall(w[4])) for w in ws)
        if ar < 2 or ar < la:
            return ws
        arl = sum(len(cls._AR_LETTER.findall(w[4])) for w in ws)      # a line of numbers has no letters: its digit tokens are numbers, not icons
        keep = []
        order = sorted(range(len(ws)), key=lambda i: ws[i][3]) if all(len(w) >= 7 for w in ws) else list(range(len(ws)))
        edge = {order[0], order[-1]}
        for i, w in enumerate(ws):
            t, cf = w[4], w[5]
            if arl >= 2 and i in edge and len(t) == 1 and t.isdigit() and cf < 97 and len(ws) >= 2 and all(len(x) >= 7 for x in ws) and cls._set_apart(ws, i):
                continue                                  # one digit set apart from a Persian line: an icon read as a digit (``2 چغر``)
            if arl >= 2 and i in edge and len(t) <= 2 and t.isdigit() and cf < 90 and len(ws) >= 3:
                continue                                  # one or two digits at the end of a Persian line: an icon (a dot, a badge, an arrow)
            if arl >= 2 and i in edge and len(ws) >= 3 and len(t) == 1 and t in "0123456789" and cf < 96:
                continue                                  # a lone Latin digit at an end of a Persian label (``9 شروع رایگان``): the icon of the button
            if arl >= 2 and i in edge and len(ws) == 2 and len(t) <= 2 and cf < 85 and all(len(x) >= 7 for x in ws) and cls._set_apart(ws, i)                     and sum(len(cls._AR_LETTER.findall(x[4])) for x in ws if x is not w) >= 3:
                continue                                  # one or two unsure characters set apart from the only word (``هص وبلاگ``): the icon of the nav item
            if not cls._AR.search(t) and not any(ch.isdigit() for ch in t):
                letters = len(cls._LAT.findall(t))
                if letters == 0 and i not in edge and all(ch in "—–-/·•.:،" for ch in t):
                    keep.append(w)                      # a dash or slash between words is punctuation, not an icon
                    continue
                # Latin letters at an end of a Persian line are, far more often than not, an icon read as letters
                if cf < 55 or (letters <= 2 and cf < 80) or (letters == 0 and cf < 88) or (i in edge and letters >= 1 and cf < 86):
                    continue
            keep.append(w)
        return TesseractBackend._drop_edge_fragments(keep or ws)

    @classmethod
    def _set_apart(cls, ws, i) -> bool:
        """is word ``i`` separated from its neighbour by a clearly wider gap than a space (at the prepared 34 px line height a space is ~8 px)?"""
        order = sorted(range(len(ws)), key=lambda j: ws[j][3])
        p = order.index(i)
        if p == 0:
            nb = order[1]
            gap = ws[nb][3] - (ws[i][3] + ws[i][6])
        elif p == len(order) - 1:
            nb = order[-2]
            gap = ws[i][3] - (ws[nb][3] + ws[nb][6])
        else:
            return False
        return gap >= 14

    @classmethod
    def _drop_edge_fragments(cls, ws):
        """An icon next to a label is read as one or two short low-confidence 'words' at an end of the line, set apart by a wider gap than the
        spaces between real words (``عد چغر چیه؟``, ``زج ورود``): drop such an end word (twice at most)."""
        ws = list(ws)
        for _ in range(2):
            if len(ws) < 3 or any(len(w) < 7 for w in ws):
                return ws
            order = sorted(range(len(ws)), key=lambda i: ws[i][3])
            gaps = [ws[order[k + 1]][3] - (ws[order[k]][3] + ws[order[k]][6]) for k in range(len(order) - 1)]
            inner = sorted(gaps[1:-1]) or [0]
            med = inner[len(inner) // 2]
            changed = False
            for pos, gap in ((0, gaps[0]), (len(order) - 1, gaps[-1])):
                w = ws[order[pos]]
                letters = sum(1 for ch in w[4] if ch.isalpha())
                if letters <= 3 and w[5] < 72 and gap >= max(9, 1.5 * med):
                    del ws[order[pos]]
                    changed = True
                    break
            if not changed:
                break
        return ws

    @classmethod
    def _logical_order(cls, ws):
        """Words of ONE visual line in reading order.  Tesseract's own word order flips when a Persian line holds a Latin word or a number
        (``PDF خروجی`` instead of ``خروجی PDF``); the pixels do not lie: a Persian line is read right to left by position, and a run of
        Latin words/numbers inside it left to right."""
        if len(ws) < 2 or any(len(w) < 9 for w in ws):
            return ws
        hs = sorted(w[8] for w in ws)
        med_h = max(1, hs[len(hs) // 2])
        tops = [w[7] + w[8] / 2.0 for w in ws]
        if max(tops) - min(tops) > 0.8 * med_h:
            return ws                                         # several text lines in one crop: keep tesseract's order
        ar = sum(len(cls._AR_LETTER.findall(w[4])) for w in ws)
        la = sum(len(cls._LAT.findall(w[4])) for w in ws)
        if ar < 2 or ar < la:
            return sorted(ws, key=lambda w: w[3])             # a Latin line or a line of numbers: left to right
        out = sorted(ws, key=lambda w: -(w[3] + w[6] / 2.0))  # a Persian line: right to left
        i = 0
        while i < len(out):                                   # runs of Latin/digit words read left to right
            if cls._latin_word(out[i][4]):
                j = i
                while j + 1 < len(out) and cls._latin_word(out[j + 1][4]):
                    j += 1
                if j > i:
                    out[i:j + 1] = out[i:j + 1][::-1]
                i = j + 1
            else:
                i += 1
        return out

    @classmethod
    def _latin_word(cls, t: str) -> bool:
        """a word that is read left to right inside a Persian line: Latin words and numbers (Persian digits too)"""
        return not cls._AR_LETTER.search(t) and any(ch.isalnum() for ch in t)

    @staticmethod
    def _assemble(ws) -> OcrResult:
        # tesseract returns words of a line in reading order; keep it, join with spaces
        ws = sorted(ws, key=lambda w: (w[2], w[1], w[0]))  # stable by block/par/line, original order within
        ws = TesseractBackend._drop_junk_words(ws)
        ws = TesseractBackend._logical_order(ws)
        text = " ".join(w[4] for w in ws)
        tot = sum(len(w[4]) for w in ws) or 1
        conf = sum(w[5] * len(w[4]) for w in ws) / tot / 100.0
        return OcrResult(text, conf, len(ws), ws=list(ws))


class EasyOcrBackend(Backend):
    name = "easyocr"
    _reader = None

    def available(self) -> bool:
        try:
            import easyocr  # noqa: F401
            return True
        except Exception:
            return False

    def read(self, crops, lang: str = "auto") -> List[OcrResult]:
        import easyocr
        if EasyOcrBackend._reader is None:
            EasyOcrBackend._reader = easyocr.Reader(["fa", "en"], gpu=False, verbose=False)
        out = []
        for c in crops:
            res = EasyOcrBackend._reader.readtext(c, detail=1, paragraph=False)
            if not res:
                out.append(OcrResult("", 0.0))
                continue
            res = sorted(res, key=lambda r: r[0][0][0])            # left-to-right by x
            from . import persian
            if persian.script_ratio(" ".join(r[1] for r in res)) >= 0.5:
                res = res[::-1]                                       # Arabic-script line: logical order is right-to-left
            text = " ".join(r[1] for r in res)
            out.append(OcrResult(text, float(np.mean([r[2] for r in res])), len(res)))
        return out


class CallbackBackend(Backend):
    """Wrap any function ``fn(crops, lang) -> [(text, conf) | OcrResult | str]`` (a remote service, another engine ...) as a backend."""
    name = "external"

    def __init__(self, fn, name: str = "external"):
        self.fn, self.name = fn, name

    def available(self) -> bool:
        return True

    def read(self, crops, lang: str = "auto") -> List[OcrResult]:
        out = []
        for r in self.fn(list(crops), lang):
            if isinstance(r, OcrResult):
                out.append(r)
            elif isinstance(r, str):
                out.append(OcrResult(r, 0.9, len(r.split())))
            else:
                t, c = r[0], float(r[1]) if len(r) > 1 else 0.9
                out.append(OcrResult(str(t or ""), c, len(str(t or "").split())))
        if len(out) != len(crops):
            raise RuntimeError("external OCR returned %d results for %d crops" % (len(out), len(crops)))
        return out


class PrecomputedBackend(Backend):
    """Results read elsewhere (e.g. on a GPU machine) for crops that ``ocr_crops`` produced: ``results`` = ``[{"box": [x, y, w, h], "text": str, "conf": float}]``.

    Crops are matched by their box, so the page analysis may differ slightly from the one that produced the request; crops without a result
    are read by ``fallback`` (default: the local engine, ``auto``) so a partial answer never loses text.  ``stats`` tells how many came from where."""

    name = "external"

    def __init__(self, results, fallback: Optional[Backend] = None, name: str = "external"):
        self.name = name
        self.by_box = {tuple(int(round(v)) for v in r["box"]): r for r in results}
        self.fallback = fallback
        self.stats = {"external": 0, "fallback": 0, "empty": 0}

    def available(self) -> bool:
        return True

    def read(self, crops, lang: str = "auto"):
        raise RuntimeError("PrecomputedBackend needs the boxes: use read_boxes")

    def read_boxes(self, crops, boxes, lang: str = "auto") -> List[OcrResult]:
        out: List[Optional[OcrResult]] = []
        missing = []
        for i, b in enumerate(boxes):
            r = self.by_box.get(tuple(int(v) for v in b))
            if r is None:
                out.append(None)
                missing.append(i)
            else:
                t = str(r.get("text") or "")
                out.append(OcrResult(t, float(r.get("conf", 0.9)), len(t.split())))
                self.stats["external"] += 1
        if missing:
            fb = self.fallback
            if fb is None:
                try:
                    fb = get_backend("auto")
                except Exception:  # noqa: BLE001
                    fb = NullBackend()
            res = fb.read([crops[i] for i in missing], lang)
            for i, r in zip(missing, res):
                out[i] = r
                self.stats["fallback" if r.text else "empty"] += 1
        return [r or OcrResult("", 0.0) for r in out]


class NullBackend(Backend):
    name = "none"

    def available(self) -> bool:
        return True

    def read(self, crops, lang: str = "auto"):
        return [OcrResult("", 0.0) for _ in crops]


BACKENDS = {"tesseract": TesseractBackend, "easyocr": EasyOcrBackend, "none": NullBackend}


def get_backend(name: str = "auto") -> Backend:
    """``auto`` = first available of tesseract, easyocr, none."""
    if name in ("auto", None):
        for n in ("tesseract", "easyocr"):
            b = BACKENDS[n]()
            if b.available():
                return b
        return NullBackend()
    if name not in BACKENDS:
        raise ValueError(f"unknown OCR backend {name!r}; choose one of {sorted(BACKENDS)} or 'auto'")
    b = BACKENDS[name]()
    if not b.available():
        raise RuntimeError(f"OCR backend {name!r} is not available. Install it (see README: extras [ocr-tesseract] / [ocr-easyocr]).")
    return b
