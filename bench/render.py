"""Benchmark-side browser helpers (the browser wrapper itself lives in the package)."""
from __future__ import annotations

import glob
import os
import shutil
import tempfile
from contextlib import contextmanager

from mv2code.browser import Browser, PROBE_JS, find_chrome  # noqa: F401


def ensure_fonts(dest_dir: str) -> None:
    """Copy the bundled OFL fonts next to the pages (``<dest>/fonts``)."""
    src = os.path.join(os.path.dirname(__file__), "fonts")
    dst = os.path.join(dest_dir, "fonts")
    os.makedirs(dst, exist_ok=True)
    for f in glob.glob(os.path.join(src, "*.woff2")):
        t = os.path.join(dst, os.path.basename(f))
        if not os.path.exists(t):
            shutil.copy2(f, t)


@contextmanager
def tempdir(prefix: str = "mv2code-"):
    d = tempfile.mkdtemp(prefix=prefix)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)
