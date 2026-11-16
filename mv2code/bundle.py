"""Write / zip the result of ``convert`` (html + fonts + tree.json + report.json)."""
from __future__ import annotations

import io
import json
import os
import shutil
import zipfile


def write_bundle(result: dict, outdir: str, name: str = "index", original=None) -> dict:
    """Write ``<name>.html``, ``fonts/``, ``tree.json`` and ``report.json`` into ``outdir``."""
    os.makedirs(outdir, exist_ok=True)
    paths = {}
    p = os.path.join(outdir, name + ".html")
    with open(p, "w", encoding="utf-8") as f:
        f.write(result["html"])
    paths["html"] = p
    for a in result.get("assets", []):
        dst = os.path.join(outdir, a["path"])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if a.get("source") and os.path.exists(a["source"]):
            shutil.copy2(a["source"], dst)
        elif a.get("bytes") is not None:
            with open(dst, "wb") as f:
                f.write(a["bytes"])
    for key, fn in (("tree_json", "tree.json"), ("report", "report.json")):
        p = os.path.join(outdir, fn)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(result[key], f, ensure_ascii=False, indent=1)
        paths[key] = p
    if original is not None:
        import cv2
        p = os.path.join(outdir, "original.png")
        cv2.imwrite(p, cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
        paths["original"] = p
    return paths


def zip_bytes(result: dict, name: str = "index", original_png: bytes | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(name + ".html", result["html"])
        z.writestr("tree.json", json.dumps(result["tree_json"], ensure_ascii=False, indent=1))
        z.writestr("report.json", json.dumps(result["report"], ensure_ascii=False, indent=1))
        for a in result.get("assets", []):
            if a.get("source") and os.path.exists(a["source"]):
                z.write(a["source"], a["path"])
        if original_png:
            z.writestr("original.png", original_png)
    return buf.getvalue()
