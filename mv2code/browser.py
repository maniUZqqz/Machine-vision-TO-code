"""Headless-browser helpers (optional extra ``[browser]``): used by ``mv2code.refine`` to
measure/render its own output and by the benchmark to probe ground-truth pages and
converter output with exactly the same code.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

PROBE_JS = r"""
() => {
  const SKIP = new Set(['SCRIPT','STYLE','NOSCRIPT','TEMPLATE','TITLE','HEAD']);
  const sx = window.scrollX, sy = window.scrollY;
  const vis = (el) => { const cs = getComputedStyle(el);
    return cs.display !== 'none' && cs.visibility !== 'hidden' && parseFloat(cs.opacity) > 0.02; };
  const visChain = (el) => { for (let e = el; e && e !== document.documentElement; e = e.parentElement)
    { if (!vis(e)) return false; } return true; };
  const union = (rects) => { let x0=1e9,y0=1e9,x1=-1e9,y1=-1e9;
    for (const q of rects) { x0=Math.min(x0,q.left); y0=Math.min(y0,q.top); x1=Math.max(x1,q.right); y1=Math.max(y1,q.bottom); }
    return [x0+sx, y0+sy, x1-x0, y1-y0]; };
  const out = {W: Math.max(document.documentElement.scrollWidth, document.body ? document.body.scrollWidth : 0),
               H: Math.max(document.documentElement.scrollHeight, document.body ? document.body.scrollHeight : 0),
               iw: window.innerWidth, texts: [], rasters: [], gt: []};
  const occl = (b, par) => { const cx = b[0]+b[2]/2 - sx, cy = b[1]+b[3]/2 - sy;
    if (cx < 0 || cy < 0 || cx >= innerWidth || cy >= innerHeight) return false;
    const top = document.elementFromPoint(cx, cy);
    return !!top && !(top === par || par.contains(top) || top.contains(par)); };
  const textRects = (root) => { const w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT); let n; const rs = [];
    while ((n = w.nextNode())) { if (!n.textContent.trim()) continue;
      const pe = n.parentElement; if (!pe || SKIP.has(pe.tagName)) continue;
      const r = document.createRange(); r.selectNodeContents(n);
      for (const q of r.getClientRects()) if (q.width > 0.5 && q.height > 0.5) rs.push(q); }
    return rs; };
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let n;
  while ((n = walker.nextNode())) {
    const s = n.textContent.replace(/\s+/g, ' ').trim();
    if (!s) continue;
    const par = n.parentElement;
    if (!par || SKIP.has(par.tagName) || !visChain(par)) continue;
    const r = document.createRange(); r.selectNodeContents(n);
    const rects = [...r.getClientRects()].filter(q => q.width > 0.5 && q.height > 0.5);
    if (!rects.length) continue;
    const b = union(rects), cs = getComputedStyle(par);
    if (cs.userSelect === 'none') continue;
    // per-line split (word rects grouped by vertical centre), logical order within a line
    const words = []; const re = /\S+/g; let mm; const txt = n.textContent;
    while ((mm = re.exec(txt))) { const rr = document.createRange(); rr.setStart(n, mm.index); rr.setEnd(n, mm.index + mm[0].length);
      const q = rr.getBoundingClientRect(); if (q.width > 0.5 && q.height > 0.5) words.push({t: mm[0], q}); }
    const lns = [];
    for (const w of words) { const cy = (w.q.top + w.q.bottom) / 2; let L = null;
      for (const l of lns) if (Math.abs(l.cy - cy) < 0.5 * (w.q.bottom - w.q.top)) { L = l; break; }
      if (!L) { L = {cy, ws: []}; lns.push(L); } L.ws.push(w); }
    const lines = lns.map(l => ({text: l.ws.map(w => w.t).join(' '), box: union(l.ws.map(w => w.q))}));
    out.texts.push({text: s, box: b, fs: parseFloat(cs.fontSize), fw: parseInt(cs.fontWeight) || 400,
      color: cs.color, ff: cs.fontFamily, dir: cs.direction, ta: cs.textAlign, occluded: occl(b, par), kind: 'text', lines});
  }
  document.querySelectorAll('input[placeholder],textarea[placeholder],input[value]').forEach(el => {
    const s = (el.value || el.getAttribute('placeholder') || '').trim();
    if (!s || !visChain(el)) return;
    const q = el.getBoundingClientRect(); const cs = getComputedStyle(el);
    const b = [q.left+sx, q.top+sy, q.width, q.height];
    out.texts.push({text: s, box: b, fs: parseFloat(cs.fontSize), fw: parseInt(cs.fontWeight) || 400, color: cs.color,
      ff: cs.fontFamily, dir: cs.direction, ta: cs.textAlign, occluded: occl(b, el), kind: 'placeholder'});
  });
  document.querySelectorAll('*').forEach(el => {
    if (SKIP.has(el.tagName)) return;
    const cs = getComputedStyle(el);
    const tag = el.tagName;
    const bg = cs.backgroundImage || '';
    if (tag === 'IMG' || tag === 'CANVAS' || tag === 'VIDEO' || tag === 'PICTURE' || bg.indexOf('url(') >= 0) {
      if (!visChain(el)) return;
      const q = el.getBoundingClientRect();
      if (q.width < 1 || q.height < 1) return;
      out.rasters.push({box: [q.left+sx, q.top+sy, q.width, q.height], tag: tag, bg: bg.indexOf('url(') >= 0});
    }
  });
  document.querySelectorAll('[data-gt]').forEach((el, i) => {
    const role = el.getAttribute('data-gt');
    if (!visChain(el)) return;
    let b, text = '';
    if (role === 'text') { const rs = textRects(el); if (!rs.length) return; b = union(rs); text = el.textContent.replace(/\s+/g,' ').trim(); }
    else { const q = el.getBoundingClientRect(); b = [q.left+sx, q.top+sy, q.width, q.height]; text = (el.textContent||'').replace(/\s+/g,' ').trim(); }
    if (b[2] < 1 || b[3] < 1) return;
    const cs = getComputedStyle(el);
    const inner = role === 'text' && !!el.closest('[data-gt="button"],[data-gt="badge"],[data-gt="table"],[data-gt="input"]');
    out.gt.push({role, box: b, text, inner, fs: parseFloat(cs.fontSize), fw: parseInt(cs.fontWeight) || 400, color: cs.color,
      bg: cs.backgroundColor, radius: cs.borderTopLeftRadius, bw: cs.borderTopWidth, bc: cs.borderTopColor, dir: cs.direction});
  });
  return out;
}
"""

DEFAULT_CHROME = ("/opt/up3d-uitest/browsers/chromium_headless_shell-1243/"
                  "chrome-headless-shell-linux64/chrome-headless-shell")


def find_chrome() -> str | None:
    env = os.environ.get("MV2CODE_CHROME")
    if env and os.path.exists(env):
        return env
    if os.path.exists(DEFAULT_CHROME):
        return DEFAULT_CHROME
    return None  # let playwright find its own browser


class Browser:
    """Context manager owning one headless browser."""

    def __init__(self):
        self._pw = None
        self._browser = None

    def __enter__(self):
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        kwargs = dict(args=["--no-sandbox", "--force-device-scale-factor=1", "--font-render-hinting=none",
                            "--disable-lcd-text", "--hide-scrollbars", "--disable-gpu"])
        exe = find_chrome()
        try:
            if exe:
                self._browser = self._pw.chromium.launch(executable_path=exe, **kwargs)
            else:
                self._browser = self._pw.chromium.launch(channel="chromium", **kwargs)
        except Exception:
            self._pw.stop()
            raise
        return self

    def __exit__(self, *exc):
        try:
            if self._browser:
                self._browser.close()
        finally:
            if self._pw:
                self._pw.stop()

    def render(self, url: str, width: int, min_height: int = 800, max_height: int = 6000,
               probe: bool = True, shot: bool = True, js: str = None):
        """Open ``url`` at ``width`` px; return (rgb ndarray | None, probe dict | None)."""
        page = self._browser.new_page(viewport={"width": int(width), "height": int(min_height)},
                                      device_scale_factor=1)
        try:
            page.goto(url, wait_until="load")
            try:
                page.evaluate("document.fonts && document.fonts.ready")
            except Exception:
                pass
            h = min_height
            for _ in range(3):
                H = page.evaluate("Math.max(document.documentElement.scrollHeight, document.body?document.body.scrollHeight:0)")
                H = int(min(max(H, 1), max_height))
                if H == h:
                    break
                h = H
                page.set_viewport_size({"width": int(width), "height": int(h)})
            page.wait_for_timeout(30)
            img = None
            if shot:
                png = page.screenshot(full_page=False, type="png")
                arr = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
                img = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            data = page.evaluate(js or PROBE_JS) if (probe or js) else None
            return img, data
        finally:
            page.close()
