"""Adapters that turn "a converter" into the uniform shape the benchmark needs:

    convert(rgb ndarray) -> {"html": str, "elements": [{role, box, inner}], "meta": {...}}

``legacy`` wraps the original (v1) vision_engine + code_generator so that the BEFORE
numbers come from the very same benchmark; ``mv2code`` wraps the new package.
"""
from __future__ import annotations

import os
import sys

import numpy as np

INNER_PARENTS = {"button", "badge", "table", "input"}


class Mv2codeAdapter:
    name = "mv2code"

    def __init__(self, mode: str = "responsive", **opts):
        self.mode = mode
        self.opts = opts

    def convert(self, rgb: np.ndarray, browser=None) -> dict:
        from mv2code import convert
        res = convert(rgb, mode=self.mode, lang=self.opts.get("lang", "auto"),
                      ocr=self.opts.get("ocr", "auto"), refine=self.opts.get("refine", "auto"),
                      browser=browser)
        elems = []

        def walk(n, inner=False):
            role = n.get("role")
            box = n.get("box")
            if role and box:
                elems.append(dict(role=role, box=box, inner=inner and role == "text", text=n.get("text", "")))
            nxt = inner or role in INNER_PARENTS
            for c in n.get("children", []) or []:
                walk(c, nxt)

        walk(res["tree_json"])
        return dict(html=res["html"], elements=elems, meta=res.get("report", {}), assets=res.get("assets", []))


class LegacyAdapter:
    name = "legacy"

    def __init__(self, mode: str = "exact", **opts):
        self.mode = mode
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if root not in sys.path:
            sys.path.insert(0, root)
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
        os.environ.setdefault("SECRET_KEY", "bench-only")
        os.environ.setdefault("DEBUG", "True")
        import django
        django.setup()
        from vision_engine.pipeline import create_default_pipeline
        from code_generator.generator import CodeGenerator
        self.pipeline = create_default_pipeline()
        self.generator = CodeGenerator()

    def convert(self, rgb: np.ndarray, browser=None) -> dict:
        from bench.legacy_rescale import _rescale_to_original
        from vision_engine.models.page import PageStructure
        from vision_engine.models.elements import ElementType
        context = self.pipeline.run(rgb)
        _rescale_to_original(context)
        orphan = [t for t in context.text_elements if t.parent_id is None]
        roots = sorted(context.regions + orphan, key=lambda e: e.bbox.area, reverse=True)
        page = PageStructure(width=context.width, height=context.height, background_color=context.page_background,
                             root_elements=roots, text_direction=getattr(context, "page_direction", "ltr"))
        out = self.generator.generate(page)
        elems = []
        roles = {ElementType.BUTTON: "button", ElementType.INPUT: "input", ElementType.SELECT: "input",
                 ElementType.TEXTAREA: "input", ElementType.IMAGE: "image", ElementType.TEXT: "text",
                 ElementType.CONTAINER: "card", ElementType.CHECKBOX: "control", ElementType.RADIO: "control"}

        def walk(e, parent_role=None):
            r = roles.get(e.element_type)
            if r:
                b = e.bbox
                elems.append(dict(role=r, box=[b.x, b.y, b.width, b.height], inner=(r == "text" and parent_role in INNER_PARENTS),
                                  text=getattr(e, "text", "") or ""))
            for c in e.children:
                walk(c, r)

        for e in roots:
            walk(e)
        return dict(html=out["combined"], elements=elems, meta={}, assets=[])


def get_adapter(engine: str, mode: str, **opts):
    if engine == "legacy":
        return LegacyAdapter(mode, **opts)
    return Mv2codeAdapter(mode, **opts)
