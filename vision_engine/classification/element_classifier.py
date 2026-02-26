import base64
import io

import cv2
import numpy as np

from ..models.elements import DetectedElement, ElementType, TextElement
from ..utils.image_utils import crop_region, luminance, color_variance
from .icon_matcher import IconMatcher


class ElementClassifier:
    """
    Step 9: Classify detected regions into semantic element types.

    Uses visual heuristics (geometry, color, children) to assign:
    BUTTON, INPUT, IMAGE, SEPARATOR — or keep as CONTAINER.

    Runs AFTER HierarchyBuilder so the full tree is available.
    """

    def __init__(self):
        self._icon_matcher = IconMatcher()

    def process(self, image: np.ndarray, context) -> object:
        img = context.working_image
        page_h, page_w = img.shape[:2]

        # Use the original (non-denoised) image for embedding image data URIs.
        # The feedback loop compares rendered HTML against the original image,
        # so data URIs from the denoised working_image would create pixel-level
        # differences at texture boundaries. If no resize was applied (same
        # dimensions), always prefer original for highest fidelity.
        orig = context.original_image
        orig_h, orig_w = orig.shape[:2]
        use_orig_for_embed = (orig_h == page_h and orig_w == page_w)

        for elem in self._walk(context.regions):
            if elem.element_type == ElementType.TEXT:
                continue
            elem.element_type = self._classify(elem, img, page_w, page_h)
            elem.semantic_tag = self._assign_semantic_tag(
                elem, page_w, page_h
            )
            # Extract border radius from contour
            elem.border_radius = self._detect_border_radius(
                img, elem.bbox
            )
            # Detect box shadow
            elem.box_shadow = self._detect_box_shadow(
                img, elem.bbox, page_w, page_h
            )
            # Embed image data for IMAGE elements
            if elem.element_type == ElementType.IMAGE:
                # Flag icons (small images ≤60x60)
                b = elem.bbox
                if (b.width <= 60 and b.height <= 60
                        and b.width >= 16 and b.height >= 16):
                    elem.is_icon = True
                    # Try to match icon to CDN library
                    self._icon_matcher.match_icon(img, elem)
                else:
                    elem.is_icon = False
                # Embed image data: prefer original image (no denoising artifacts)
                src = orig if use_orig_for_embed else img
                elem.image_data_uri = self._crop_to_data_uri(src, elem.bbox)

            # Background image for large containers with complex, multi-toned
            # backgrounds (e.g. SCADA panels with both dark housing and bright
            # instrument faces). A flat background-color causes large pixel
            # errors in the comparison; embedding the background as a data URI
            # reproduces the texture exactly while preserving child rendering.
            elif elem.element_type == ElementType.CONTAINER:
                src = orig if use_orig_for_embed else img
                if self._needs_bg_image(elem, src, page_w, page_h):
                    elem.bg_image_data_uri = self._crop_to_data_uri(
                        src, elem.bbox
                    )

        # Also assign semantic tags to orphan text elements
        for text_elem in context.text_elements:
            text_elem.semantic_tag = self._text_semantic_tag(
                text_elem, page_w, page_h
            )

        return context

    def _classify(self, elem, image, page_w, page_h):
        """Determine element type using visual heuristics.
        Order matters: check most specific types first."""

        if self._is_separator(elem):
            return ElementType.SEPARATOR

        # Checkbox/Radio: small squares/circles (before image check)
        form_type = self._is_checkbox_or_radio(elem, image)
        if form_type:
            return form_type

        if self._is_image_or_icon(elem, image):
            return ElementType.IMAGE

        # Select/dropdown: rectangle with small triangle indicator
        if self._is_select_dropdown(elem, image, page_w):
            return ElementType.SELECT

        # Textarea: large input-like rectangle
        if self._is_textarea(elem):
            return ElementType.TEXTAREA

        if self._is_button(elem):
            return ElementType.BUTTON

        if self._is_input(elem):
            return ElementType.INPUT

        return ElementType.CONTAINER

    # ------------------------------------------------------------------
    # Separator: very thin bar or line
    # ------------------------------------------------------------------

    def _is_separator(self, elem):
        b = elem.bbox
        # Thin in one dimension (< 5px) and long in the other (> 50px)
        is_horizontal = b.height <= 5 and b.width >= 50
        is_vertical = b.width <= 5 and b.height >= 50

        if not (is_horizontal or is_vertical):
            return False

        # Must have no children
        if elem.children:
            return False

        # Uniform color (if available)
        if elem.color and elem.color.background:
            return True

        return is_horizontal or is_vertical

    # ------------------------------------------------------------------
    # Image / Icon: high color variance, no text children
    # ------------------------------------------------------------------

    def _is_image_or_icon(self, elem, image):
        b = elem.bbox
        page_h, page_w = image.shape[:2]

        # Minimum meaningful size
        if b.area < 400:
            return False

        # Exclude page-spanning structural elements (sidebars, full-height panels)
        if b.height > page_h * 0.7 or b.width > page_w * 0.7:
            return False

        # Exclude very large containers (> 25% of page area)
        if b.area > page_w * page_h * 0.25:
            return False

        crop = crop_region(image, b)
        if crop.size == 0:
            return False

        variance = color_variance(crop)

        # Complex graphical elements (e.g. SCADA gauges, instrument panels) may
        # contain OCR-detected text labels or nested structural containers. If the
        # element has image-like variance and detected children only cover a small
        # fraction of the area, the visual content is predominantly graphical and
        # should be embedded as a pixel-accurate image crop rather than recreated
        # with CSS containers.
        if variance > 45 and b.width >= 80 and b.height >= 60:
            child_area = sum(c.bbox.area for c in elem.children)
            child_coverage = child_area / max(1, b.area)
            if child_coverage < 0.50:
                return True

        # Standard restrictions
        # Must have no text children
        if any(c.element_type == ElementType.TEXT for c in elem.children):
            return False

        # Must have no container children with their own children
        if any(c.children for c in elem.children):
            return False

        # Icons: small (< 60x60) with moderate variance
        # Must be at least 16x16 to avoid noise fragments
        if (b.width <= 60 and b.height <= 60
                and b.width >= 16 and b.height >= 16 and variance > 35):
            return True

        # Images: larger regions with high color variance
        # Must be at least 80x60 to be a meaningful image
        if b.width >= 80 and b.height >= 60 and variance > 45:
            return True

        return False

    # ------------------------------------------------------------------
    # Button: small colored rectangle with centered text
    # ------------------------------------------------------------------

    def _is_button(self, elem):
        b = elem.bbox

        # Size constraints: typical button dimensions
        if not (25 <= b.width <= 350):
            return False
        if not (18 <= b.height <= 75):
            return False

        # Aspect ratio: buttons are wider than tall but not extremely
        aspect = b.width / max(b.height, 1)
        if not (1.2 <= aspect <= 10):
            return False

        # Must have at least 1 text child
        text_children = [
            c for c in elem.children
            if c.element_type == ElementType.TEXT
        ]
        if not text_children:
            return False

        # Only text children (no nested containers)
        non_text = [
            c for c in elem.children
            if c.element_type != ElementType.TEXT
        ]
        if non_text:
            return False

        # Text should be roughly centered (padding balanced)
        text = text_children[0]
        h_gap_left = text.bbox.x - b.x
        h_gap_right = b.x2 - text.bbox.x2
        # Allow 50% asymmetry for centered text
        if h_gap_left > 0 and h_gap_right > 0:
            ratio = min(h_gap_left, h_gap_right) / max(h_gap_left, h_gap_right)
            if ratio < 0.3:
                return False

        # Background should be visible (not white/transparent)
        if elem.color and elem.color.background:
            bg_lum = luminance(elem.color.background)
            # Colored buttons: non-white background
            if bg_lum < 230:
                return True
            # White buttons with border
            if elem.color.border:
                return True

        return False

    # ------------------------------------------------------------------
    # Checkbox / Radio: small square or circle
    # ------------------------------------------------------------------

    def _is_checkbox_or_radio(self, elem, image):
        b = elem.bbox
        # Must be small (12-30px) and roughly square
        if not (12 <= b.width <= 30 and 12 <= b.height <= 30):
            return None
        aspect = b.width / max(b.height, 1)
        if not (0.7 <= aspect <= 1.4):
            return None
        # No children
        if elem.children:
            return None
        # Check shape: circular = radio, square = checkbox
        crop = crop_region(image, b)
        if crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        # Detect circles with HoughCircles
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                    param1=50, param2=15,
                                    minRadius=b.width // 4,
                                    maxRadius=b.width // 2 + 2)
        if circles is not None:
            return ElementType.RADIO
        return ElementType.CHECKBOX

    # ------------------------------------------------------------------
    # Select/Dropdown: input-like rectangle with small triangle
    # ------------------------------------------------------------------

    def _is_select_dropdown(self, elem, image, page_w):
        b = elem.bbox
        # Similar to input but with a caret/triangle on the right side
        if not (80 <= b.width <= 600):
            return False
        if not (25 <= b.height <= 55):
            return False
        aspect = b.width / max(b.height, 1)
        if aspect < 2.0:
            return False

        # Reject page-wide or section-wide elements (status bars, header bars).
        # A real dropdown is narrow; anything wider than 25% of the page is
        # a structural element (e.g. "SYSTEM NORMAL" status strip).
        if page_w > 0 and b.width > page_w * 0.25:
            return False

        # Reject highly saturated elements — colored status bars (green, red)
        # are not dropdowns.
        if elem.color and elem.color.background:
            bg = elem.color.background
            mx = max(bg)
            mn = min(bg)
            sat = int(255 * (mx - mn) / mx) if mx > 0 else 0
            if sat > 80:
                return False

        # Must have border or light background (like input)
        has_border = elem.color and elem.color.border
        has_light_bg = (elem.color and elem.color.background
                        and luminance(elem.color.background) > 200)
        if not (has_border or has_light_bg):
            return False
        # Check for a small triangle/caret on the right side
        crop = crop_region(image, b)
        if crop.size == 0:
            return False
        # Sample the rightmost 15% of the element
        right_region = crop[:, int(crop.shape[1] * 0.85):]
        if right_region.size == 0:
            return False
        gray_right = cv2.cvtColor(right_region, cv2.COLOR_RGB2GRAY)
        # Look for dark pixels forming a small triangle
        _, binary = cv2.threshold(gray_right, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        dark_ratio = np.sum(binary > 0) / max(binary.size, 1)
        # A dropdown caret has ~5-25% dark pixels in the right region
        if 0.03 < dark_ratio < 0.30:
            return True
        return False

    # ------------------------------------------------------------------
    # Textarea: larger input-like element
    # ------------------------------------------------------------------

    def _is_textarea(self, elem):
        b = elem.bbox
        # Textarea: wider input with more height (>55px)
        if not (100 <= b.width <= 800):
            return False
        if not (55 <= b.height <= 300):
            return False
        # Must have border or light bg
        has_border = elem.color and elem.color.border
        has_light_bg = (elem.color and elem.color.background
                        and luminance(elem.color.background) > 200)
        if not (has_border and has_light_bg):
            return False
        # No container children
        if any(c.element_type not in (ElementType.TEXT,) for c in elem.children):
            return False
        return True

    # ------------------------------------------------------------------
    # Input: bordered rectangle, light background, few/no children
    # ------------------------------------------------------------------

    def _is_input(self, elem):
        b = elem.bbox

        # Size: typical input field dimensions
        if not (80 <= b.width <= 600):
            return False
        if not (18 <= b.height <= 55):
            return False

        # Aspect ratio: inputs are significantly wider than tall
        aspect = b.width / max(b.height, 1)
        if aspect < 2.0:
            return False

        # No container children (only text or empty)
        container_children = [
            c for c in elem.children
            if c.element_type != ElementType.TEXT
        ]
        if container_children:
            return False

        # Max 1 text child (placeholder text)
        text_children = [
            c for c in elem.children
            if c.element_type == ElementType.TEXT
        ]
        if len(text_children) > 1:
            return False

        # Must have a border OR light background
        has_border = elem.color and elem.color.border
        has_light_bg = (
            elem.color and elem.color.background
            and luminance(elem.color.background) > 200
        )

        return has_border and has_light_bg

    # ------------------------------------------------------------------
    # Semantic tag assignment for containers
    # ------------------------------------------------------------------

    def _assign_semantic_tag(self, elem, page_w, page_h):
        """Map element type + position to HTML semantic tag."""

        if elem.element_type == ElementType.BUTTON:
            return "button"
        if elem.element_type == ElementType.INPUT:
            return "input"
        if elem.element_type == ElementType.IMAGE:
            return "img"
        if elem.element_type == ElementType.SEPARATOR:
            return "hr"
        if elem.element_type == ElementType.SELECT:
            return "select"
        if elem.element_type == ElementType.CHECKBOX:
            return "input"
        if elem.element_type == ElementType.RADIO:
            return "input"
        if elem.element_type == ElementType.TEXTAREA:
            return "textarea"

        # Container semantic tags based on position
        b = elem.bbox
        is_full_width = b.width >= page_w * 0.85

        # Header: top of page, full width, and NOT tall (< 15% page height)
        if b.y < page_h * 0.08 and is_full_width and b.height < page_h * 0.15:
            return "header"

        # Footer: bottom of page, full width, and NOT tall (< 15% page height)
        if b.y2 > page_h * 0.88 and is_full_width and b.height < page_h * 0.15:
            return "footer"

        # Sidebar: tall narrow panel at edge (only root-level elements)
        is_narrow = b.width < page_w * 0.20
        is_tall = b.height > page_h * 0.50
        at_edge = b.x < page_w * 0.05 or b.x2 > page_w * 0.95
        if is_narrow and is_tall and at_edge and elem.parent_id is None:
            return "aside"

        # Nav: inside header-like position, or horizontal list of links
        if b.y < page_h * 0.12 and is_full_width:
            has_multiple_text = sum(
                1 for c in elem.children
                if c.element_type == ElementType.TEXT
            ) >= 3
            if has_multiple_text:
                return "nav"

        # List detection: 3+ vertically stacked children with similar height/width
        if self._looks_like_list(elem):
            return "ul"

        # Table detection: grid layout with header-like first row
        if self._looks_like_table(elem):
            return "table"

        return "div"

    def _looks_like_list(self, elem):
        """Detect list pattern: 3+ children stacked vertically with similar sizes."""
        children = [c for c in elem.children if c.element_type != ElementType.SEPARATOR]
        if len(children) < 3:
            return False

        # All children should have similar heights (within 30%)
        heights = [c.bbox.height for c in children]
        avg_h = sum(heights) / len(heights)
        if avg_h == 0:
            return False
        height_variance = max(heights) - min(heights)
        if height_variance / avg_h > 0.5:
            return False

        # Children should be vertically stacked (sorted by y, no major x overlap issues)
        sorted_by_y = sorted(children, key=lambda c: c.bbox.y)
        for i in range(len(sorted_by_y) - 1):
            # Next item should start after current ends (no major overlap)
            if sorted_by_y[i + 1].bbox.y < sorted_by_y[i].bbox.y2 - 5:
                return False

        # Children should have similar x positions (left-aligned)
        x_positions = [c.bbox.x for c in children]
        x_range = max(x_positions) - min(x_positions)
        if x_range > elem.bbox.width * 0.3:
            return False

        return True

    def _looks_like_table(self, elem):
        """Detect table pattern: grid layout where first row has distinct style."""
        layout = getattr(elem, 'layout', None)
        if not layout:
            return False
        from vision_engine.models.layout import LayoutMode
        if layout.mode != LayoutMode.GRID:
            return False
        if not layout.columns or layout.columns < 2:
            return False
        if not layout.rows or layout.rows < 2:
            return False
        return True

    # ------------------------------------------------------------------
    # Semantic tag for text elements
    # ------------------------------------------------------------------

    def _text_semantic_tag(self, text_elem, page_w, page_h):
        """Assign semantic tag to text elements based on context."""
        # Check if parent is a nav
        # (parent_id is set but we don't have direct parent ref here,
        #  so we rely on html_builder to handle nav children)

        if hasattr(text_elem, 'typography') and text_elem.typography:
            fs = text_elem.typography.font_size_px
            if fs >= 28:
                return "h1"
            if fs >= 22:
                return "h2"
            if fs >= 18:
                return "h3"
            if fs >= 16:
                return "h4"
        return "p"

    # ------------------------------------------------------------------
    # Image cropping → base64 data URI
    # ------------------------------------------------------------------

    def _crop_to_data_uri(self, image, bbox):
        """Crop the region from the image and encode as a JPEG data URI."""
        crop = crop_region(image, bbox)
        if crop.size == 0:
            return None
        # Convert RGB to BGR for cv2 encoding
        bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        # Use JPEG at higher quality for pixel-accurate rendering
        ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            return None
        b64 = base64.b64encode(buf.tobytes()).decode('ascii')
        return f'data:image/jpeg;base64,{b64}'

    # ------------------------------------------------------------------
    # Border radius detection via corner analysis
    # ------------------------------------------------------------------

    def _detect_border_radius(self, image, bbox):
        """Detect border radius by analyzing corner transparency/background.

        Examines the top-left corner of the element. In a rounded rectangle,
        the corner triangle contains background pixels. We measure how many
        pixels in the corner are background-colored to estimate the radius.
        """
        crop = crop_region(image, bbox)
        if crop.size == 0 or bbox.width < 12 or bbox.height < 12:
            return None

        # Sample a corner region (max 20x20)
        corner_size = min(20, bbox.width // 3, bbox.height // 3)
        if corner_size < 4:
            return None

        corner = crop[:corner_size, :corner_size]
        gray_corner = cv2.cvtColor(corner, cv2.COLOR_RGB2GRAY)

        # Get the element's interior color (center pixel)
        ch, cw = crop.shape[:2]
        center_color = crop[ch // 2, cw // 2]
        center_gray = int(0.299 * center_color[0] + 0.587 * center_color[1]
                          + 0.114 * center_color[2])

        # Count pixels in the corner that differ significantly from interior
        diff = np.abs(gray_corner.astype(np.int16) - center_gray)
        bg_pixels = np.sum(diff > 20)
        total_pixels = corner_size * corner_size

        if total_pixels == 0:
            return None

        bg_ratio = bg_pixels / total_pixels

        # Map bg_ratio to border radius
        # ~0% bg → 0px (sharp corner)
        # ~10-20% bg → 4-8px radius
        # ~30%+ bg → 12-20px (very rounded / pill)
        if bg_ratio < 0.03:
            return 0  # Sharp corners
        elif bg_ratio < 0.08:
            return 4
        elif bg_ratio < 0.15:
            return 8
        elif bg_ratio < 0.25:
            return 12
        elif bg_ratio < 0.40:
            return 16
        else:
            return 20

    # ------------------------------------------------------------------
    # Box shadow detection
    # ------------------------------------------------------------------

    def _detect_box_shadow(self, image, bbox, page_w, page_h):
        """Detect box shadow by analyzing the gradient outside the element edges.

        A shadow appears as a smooth luminance gradient from the element's edge
        outward into the background. We sample strips outside each edge and
        measure gradual darkening.
        """
        margin = 15  # How far outside to sample
        h, w = image.shape[:2]

        # Only detect for containers/buttons (skip separators, very small)
        if bbox.area < 2000 or bbox.width < 30 or bbox.height < 30:
            return None

        # Get element interior luminance (center strip)
        cx, cy = bbox.center
        interior = image[
            max(0, cy - 2):min(h, cy + 3),
            max(0, cx - 2):min(w, cx + 3)
        ]
        if interior.size == 0:
            return None
        interior_lum = float(np.mean(interior) * 0.299 + np.mean(interior[:,:,1]) * 0.587 + np.mean(interior[:,:,2]) * 0.114) if interior.ndim == 3 else float(np.mean(interior))

        # Sample strips outside each edge
        shadow_detected = False
        shadow_blur = 0

        # Bottom edge (most common shadow direction)
        if bbox.y2 + margin < h:
            strip = image[bbox.y2:min(h, bbox.y2 + margin), max(0, bbox.x + 5):min(w, bbox.x2 - 5)]
            if strip.size > 0:
                gray_strip = np.mean(strip.reshape(-1, 3), axis=1) if strip.ndim == 3 else strip.flatten()
                # Check for gradual luminance change (shadow gradient)
                rows_lum = []
                strip_h = strip.shape[0]
                for r in range(strip_h):
                    row = strip[r, :]
                    if row.ndim == 2:
                        rows_lum.append(np.mean(row))
                    else:
                        rows_lum.append(float(row.mean()))

                if len(rows_lum) >= 3:
                    # Shadow = gradual change in first few pixels then stabilize
                    diff_start = abs(rows_lum[0] - rows_lum[min(2, len(rows_lum)-1)])
                    diff_end = abs(rows_lum[-1] - rows_lum[max(0, len(rows_lum)-3)])

                    # Shadow: noticeable difference at start (>5) but stabilizes at end (<3)
                    if diff_start > 5 and diff_end < diff_start * 0.6:
                        shadow_detected = True
                        # Estimate blur from gradient width
                        for i, lum in enumerate(rows_lum):
                            if i > 0 and abs(lum - rows_lum[-1]) < 2:
                                shadow_blur = max(4, i * 2)
                                break
                        else:
                            shadow_blur = 8

        # Right edge check
        if not shadow_detected and bbox.x2 + margin < w:
            strip = image[max(0, bbox.y + 5):min(h, bbox.y2 - 5), bbox.x2:min(w, bbox.x2 + margin)]
            if strip.size > 0 and strip.shape[1] >= 3:
                cols_lum = []
                for c in range(strip.shape[1]):
                    col = strip[:, c]
                    cols_lum.append(float(np.mean(col)))

                if len(cols_lum) >= 3:
                    diff_start = abs(cols_lum[0] - cols_lum[min(2, len(cols_lum)-1)])
                    diff_end = abs(cols_lum[-1] - cols_lum[max(0, len(cols_lum)-3)])
                    if diff_start > 5 and diff_end < diff_start * 0.6:
                        shadow_detected = True
                        shadow_blur = 8

        if shadow_detected:
            shadow_blur = max(4, min(shadow_blur, 20))
            return f'0 2px {shadow_blur}px rgba(0, 0, 0, 0.15)'

        return None

    # ------------------------------------------------------------------
    # Background-image detection for complex containers
    # ------------------------------------------------------------------

    def _needs_bg_image(self, elem, image, page_w, page_h):
        """Return True when a container's background is too complex for a
        single CSS colour.

        Criteria:
        - Large area (> 8 % of page) — small cards rarely need this
        - Has at least 2 children (pure empty containers don't need it)
        - Exposed pixels (not covered by any child bbox) span a wide
          luminance range: std > 55 — indicates mixed dark/bright regions
          (e.g. dark industrial housing + white instrument faces)
        """
        b = elem.bbox
        page_area = page_w * page_h
        if b.area < page_area * 0.08:
            return False
        if len(elem.children) < 2:
            return False

        h, w = image.shape[:2]
        x1, y1 = max(0, b.x), max(0, b.y)
        x2, y2 = min(w, b.x2), min(h, b.y2)
        if x2 - x1 < 20 or y2 - y1 < 20:
            return False

        region_h = y2 - y1
        region_w = x2 - x1
        covered = np.zeros((region_h, region_w), dtype=bool)
        for child in elem.children:
            cx1 = max(0, child.bbox.x - x1)
            cy1 = max(0, child.bbox.y - y1)
            cx2 = min(region_w, child.bbox.x2 - x1)
            cy2 = min(region_h, child.bbox.y2 - y1)
            if cx2 > cx1 and cy2 > cy1:
                covered[cy1:cy2, cx1:cx2] = True

        exposed_mask = ~covered
        n_exposed = exposed_mask.sum()

        # Check FULL region luminance std first.
        # For Gantt-chart/dashboard panels, the exposed background is uniform
        # (white) but the element overall has diverse content (colored bars as
        # children). The full-region std catches this case and correctly assigns
        # a bg-image so the bars are captured in the background JPEG.
        region_pixels = image[y1:y2, x1:x2]
        # Subsample for speed: take every 4th pixel
        sample = region_pixels[::4, ::4]
        full_lum = (sample[:, :, 0].astype(float) * 0.2126
                    + sample[:, :, 1].astype(float) * 0.7152
                    + sample[:, :, 2].astype(float) * 0.0722)
        if full_lum.std() > 40:
            return True

        if n_exposed < 500:
            return False  # Not enough exposed pixels to judge

        exposed_pixels = image[y1:y2, x1:x2][exposed_mask]
        lum = (exposed_pixels[:, 0].astype(float) * 0.2126
               + exposed_pixels[:, 1].astype(float) * 0.7152
               + exposed_pixels[:, 2].astype(float) * 0.0722)
        lum_std = lum.std()
        return lum_std > 25

    # ------------------------------------------------------------------

    def _walk(self, elements):
        for elem in elements:
            yield elem
            yield from self._walk(elem.children)
