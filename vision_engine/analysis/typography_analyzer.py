import numpy as np

from ..models.elements import TypographyInfo, ElementType
from ..utils.image_utils import crop_region, luminance


class TypographyAnalyzer:
    """
    Step 5: Estimate typography properties for text elements.
    - Font size from bounding box height
    - Font weight from stroke intensity
    - Text alignment from position within parent
    """

    def process(self, image: np.ndarray, context) -> object:
        for text_elem in context.text_elements:
            text_content = getattr(text_elem, 'text', '')
            font_size = self._estimate_font_size(
                text_elem.bbox.height, text_elem.bbox.width, len(text_content)
            )
            font_weight = self._estimate_weight(
                context.original_image, text_elem
            )
            text_align = self._detect_alignment(text_elem, context.regions)

            # Estimate line height from bbox height relative to font size
            line_height = round(text_elem.bbox.height / max(font_size, 1), 2) if font_size > 0 else 1.4

            text_elem.typography = TypographyInfo(
                font_size_px=font_size,
                font_weight=font_weight,
                text_align=text_align,
                line_height=line_height,
            )

        return context

    def _estimate_font_size(self, bbox_height: int, bbox_width: int = 0,
                            text_length: int = 0) -> int:
        """Approximate font size from bounding box dimensions and text length."""
        # Height-based estimate: bbox is roughly 1.5-1.7x the font size
        size_from_height = int(bbox_height * 0.6)

        # Width-based estimate: if we know text length, cap font size
        # so text fits in one line (avg char width ~0.55 * font_size)
        if text_length > 0 and bbox_width > 0:
            size_from_width = int(bbox_width / (text_length * 0.55))
            size_from_height = min(size_from_height, size_from_width)

        return max(8, min(size_from_height, 72))

    def _estimate_weight(self, image: np.ndarray, text_elem) -> str:
        """Estimate if text is bold based on stroke density."""
        crop = crop_region(image, text_elem.bbox)
        if crop.size == 0:
            return "normal"

        import cv2
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

        # Calculate the ratio of dark pixels (text) to total
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        dark_ratio = np.sum(binary > 0) / binary.size

        # Bold text typically has higher ink density
        return "bold" if dark_ratio > 0.35 else "normal"

    def _detect_alignment(self, text_elem, regions) -> str:
        """Detect text alignment within its parent container."""
        if not text_elem.parent_id:
            return "left"

        # Find parent
        parent = self._find_by_id(text_elem.parent_id, regions)
        if not parent:
            return "left"

        parent_bbox = parent.bbox
        text_bbox = text_elem.bbox

        left_gap = text_bbox.x - parent_bbox.x
        right_gap = parent_bbox.x2 - text_bbox.x2

        if parent_bbox.width < 10:
            return "left"

        left_ratio = left_gap / parent_bbox.width
        right_ratio = right_gap / parent_bbox.width

        # If gaps are roughly equal, it's centered
        if abs(left_ratio - right_ratio) < 0.1:
            return "center"
        elif right_ratio < left_ratio:
            return "right"
        return "left"

    def _find_by_id(self, element_id, elements):
        for elem in elements:
            if elem.id == element_id:
                return elem
            found = self._find_by_id(element_id, elem.children)
            if found:
                return found
        return None
