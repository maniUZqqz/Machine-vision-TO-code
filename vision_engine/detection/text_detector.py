import re
import uuid
import cv2
import numpy as np
from typing import List

from ..models.elements import (
    TextElement, ElementType, BoundingBox, TextDirection,
)
from ..utils.geometry import find_smallest_container, find_container_by_center

# Regex for Persian/Arabic characters
PERSIAN_PATTERN = re.compile(r'[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]')


class TextDetector:
    """
    Step 3: Text detection using EasyOCR.
    Handles multi-language (Persian + English) and RTL detection.
    Uses CLAHE contrast enhancement for better detection on dark UIs.
    """

    def __init__(self, languages: List[str] = None,
                 confidence_threshold: float = 0.35):
        self.languages = languages or ['en']
        self.confidence_threshold = confidence_threshold
        self._reader = None

    @property
    def reader(self):
        """Lazy initialization of EasyOCR reader."""
        if self._reader is None:
            import os
            import sys
            import io
            import easyocr

            # Fix Windows encoding issue with EasyOCR progress bar
            if sys.platform == 'win32' and not isinstance(sys.stdout, io.TextIOWrapper):
                os.environ['PYTHONIOENCODING'] = 'utf-8'

            # Try GPU first, fall back to CPU
            try:
                self._reader = easyocr.Reader(
                    self.languages, gpu=True, verbose=False
                )
            except Exception:
                self._reader = easyocr.Reader(
                    self.languages, gpu=False, verbose=False
                )
        return self._reader

    def _enhance_for_ocr(self, img: np.ndarray) -> np.ndarray:
        """Enhance image contrast for better OCR on dark/low-contrast UIs.

        Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
        to the lightness channel in LAB color space. This dramatically
        improves text detection on dark dashboards without affecting
        light-background UIs.
        """
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]

        # Check if the image is dark (mean luminance < 120)
        mean_lum = float(l_channel.mean())
        if mean_lum < 120:
            # Apply strong CLAHE for dark images
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(l_channel)
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return img

    def process(self, image: np.ndarray, context) -> object:
        # Enhance contrast for better OCR
        enhanced = self._enhance_for_ocr(context.working_image)

        results = self.reader.readtext(enhanced)

        text_elements = []
        for (bbox_coords, text, confidence) in results:
            if confidence < self.confidence_threshold:
                continue
            if not text.strip():
                continue

            # Convert EasyOCR 4-point bbox to BoundingBox
            xs = [int(point[0]) for point in bbox_coords]
            ys = [int(point[1]) for point in bbox_coords]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            bbox = BoundingBox(x_min, y_min, x_max - x_min, y_max - y_min)

            # Post-process OCR text
            text = self._post_process_text(text.strip())
            if not text:
                continue

            direction = self._detect_direction(text)

            elem = TextElement(
                id=str(uuid.uuid4())[:8],
                element_type=ElementType.TEXT,
                bbox=bbox,
                text=text,
                direction=direction,
                ocr_confidence=confidence,
            )
            text_elements.append(elem)

        # Associate text elements with parent regions
        self._assign_to_regions(text_elements, context.regions)

        context.text_elements = text_elements
        return context

    def _post_process_text(self, text: str) -> str:
        """Fix common OCR errors in both English and Persian text."""
        if not text:
            return text

        # Skip single-character noise
        if len(text) == 1 and not text.isalnum():
            return ''

        # Common English OCR substitutions on technical UIs
        # Pipe | often confused with 1 or l
        # Semicolon ; confused with colon :
        # These are context-dependent, so we only fix obvious patterns

        # Fix common SCADA/dashboard label errors
        text = re.sub(r'\bFLDW\b', 'FLOW', text)
        text = re.sub(r'\bDOSIN\b', 'DOSING', text)
        text = re.sub(r'\bDUSING\b', 'DOSING', text)
        text = re.sub(r'\bUUSING\b', 'DOSING', text)
        text = re.sub(r'\bOUTPUL\b', 'OUTPUT', text)
        text = re.sub(r'\bTEMPERATBE\b', 'TEMPERATURE', text)
        text = re.sub(r'\bLEV Li\b', 'LEVEL:', text)
        text = re.sub(r'\bLETE\b', 'LEVEL', text)
        text = re.sub(r'\bpcsImin\b', 'pcs/min', text)
        text = re.sub(r'(\d+)\s*Lmin\b', r'\1 L/min', text)

        # Fix semicolons that should be colons in labels
        text = re.sub(r'(\w+);$', r'\1:', text)

        # Common Persian OCR errors
        # سوشمند → هوشمند (intelligent)
        text = text.replace('سوشمند', 'هوشمند')
        # برنامة → برنامه (program) — ة vs ه confusion
        text = text.replace('برنامة', 'برنامه')
        # ریری → ریزی (planning)
        text = text.replace('ریری', 'ریزی')
        # فمال → فعال (active)
        text = text.replace('فمال', 'فعال')
        # نموئههاى → نمونه‌های (samples)
        text = text.replace('نموئههاى', 'نمونه‌های')
        # خوداشبورد → داشبورد (dashboard)
        text = text.replace('خوداشبورد', 'داشبورد')
        # كاليبراسرون → کالیبراسیون (calibration)
        text = text.replace('كاليبراسرون', 'کالیبراسیون')
        # ورن هاى → وزن‌های (weights)
        text = text.replace('ورن هاى', 'وزن‌های')
        # ارانه → ارائه (provider)
        text = text.replace('ارانه', 'ارائه')
        # لبرنامة → برنامه (leading lam artifact)
        text = re.sub(r'^ل(برنامه)', r'\1', text)
        # دستى → دستی (manual) — ى vs ی
        text = text.replace('دستى', 'دستی')
        # وضعيت → وضعیت (status) — arabic ya to persian ya
        text = text.replace('وضعيت', 'وضعیت')
        text = text.replace('سيستم', 'سیستم')

        return text

    def _detect_direction(self, text: str) -> TextDirection:
        """Determine text direction based on character analysis."""
        persian_chars = len(PERSIAN_PATTERN.findall(text))
        total_alpha = sum(1 for c in text if c.isalpha())

        if total_alpha == 0:
            return TextDirection.LTR

        ratio = persian_chars / total_alpha
        if ratio > 0.5:
            return TextDirection.RTL
        elif ratio > 0.1:
            return TextDirection.MIXED
        return TextDirection.LTR

    def _assign_to_regions(self, text_elements, regions):
        """
        For each text element, find the smallest region that contains it.

        Uses strict containment first. Falls back to center-point containment
        when OCR bbox edges slightly exceed the container boundary (common with
        EasyOCR which can be 1-3px imprecise at box edges).
        """
        all_regions = []
        self._collect_all(regions, all_regions)

        for text_elem in text_elements:
            parent = find_smallest_container(text_elem.bbox, all_regions)
            if not parent:
                # Fallback: use text center instead of full bbox containment
                parent = find_container_by_center(text_elem.bbox, all_regions)
            if parent:
                text_elem.parent_id = parent.id
                parent.children.append(text_elem)

    def _collect_all(self, elements, result):
        for elem in elements:
            result.append(elem)
            self._collect_all(elem.children, result)
