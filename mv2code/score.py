"""Honest self-check of a conversion: render the HTML and compare it with the screenshot.

SSIM and mean pixel similarity (strict, over all channels) plus the *editable share*: the part of
the page area that is real HTML/CSS and not an embedded raster (photos are allowed to stay
images, so the screenshot's photo areas are excluded).  Needs the ``[browser]`` extra.
"""
from __future__ import annotations

import numpy as np

from .refine import PageRenderer, ssim_gray

RASTER_JS = r"""
() => { const out = []; const sx = scrollX, sy = scrollY;
  document.querySelectorAll('*').forEach(e => { const cs = getComputedStyle(e);
    const bg = cs.backgroundImage || ''; const mk = cs.maskImage || cs.webkitMaskImage || '';
    if (e.tagName === 'IMG' || e.tagName === 'CANVAS' || e.tagName === 'VIDEO' || bg.indexOf('url(') >= 0 || mk.indexOf('url(') >= 0) {
      const r = e.getBoundingClientRect(); if (r.width > 1 && r.height > 1) out.push([r.left + sx, r.top + sy, r.width, r.height]); } });
  const texts = []; const w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT); let n;
  while ((n = w.nextNode())) { const t = n.textContent.trim(); if (t && n.parentElement && !['SCRIPT','STYLE'].includes(n.parentElement.tagName)) texts.push(t); }
  return {rasters: out, text: texts.join(' '), W: document.documentElement.scrollWidth, H: document.documentElement.scrollHeight}; }
"""


def compare(original_rgb: np.ndarray, html: str, assets, browser, photo_boxes=()):
    """Return a dict of scores for ``html`` rendered at the screenshot's width."""
    H, W = original_rgb.shape[:2]
    pr = PageRenderer(browser, assets, W)
    try:
        img, extra = pr.render(html, js=RASTER_JS)
    finally:
        pr.close()
    ssim = ssim_gray(original_rgb, img)
    h = max(H, img.shape[0])

    def pad(x):
        o = np.empty((h, W, 3), np.uint8)
        o[:] = x[-1:, -1:, :]
        o[:x.shape[0], :min(W, x.shape[1])] = x[:, :W]
        return o
    pix = 1.0 - float(np.mean(np.abs(pad(original_rgb).astype(np.int16) - pad(img).astype(np.int16)))) / 255.0
    allowed = np.zeros((H, W), bool)
    for x, y, w, hh in photo_boxes:
        allowed[max(0, int(y) - 3):min(H, int(y + hh) + 4), max(0, int(x) - 3):min(W, int(x + w) + 4)] = True
    raster = np.zeros((H, W), bool)
    for x, y, w, hh in extra["rasters"]:
        raster[max(0, int(y)):min(H, int(y + hh + .5)), max(0, int(x)):min(W, int(x + w + .5))] = True
    bad = int((raster & ~allowed).sum())
    editable = max(0.0, 1.0 - bad / max(1, H * W - int(allowed.sum())))
    return dict(ssim=round(ssim, 4), pixel_similarity=round(pix, 4), editable_share=round(editable, 4),
                rendered_size=[int(img.shape[1]), int(img.shape[0])], real_text_chars=len(extra["text"].replace(" ", "")))
