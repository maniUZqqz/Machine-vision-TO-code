import cv2
import numpy as np
import uuid
from typing import List

from ..models.elements import BoundingBox, DetectedElement, ElementType
from ..utils.geometry import merge_overlapping_boxes


class ContourDetector:
    """
    Step 2: Detect UI regions using multiple strategies.

    1. Adaptive threshold — bordered elements, cards, panels
    2. Color segmentation — colored buttons, headers, sidebars
    3. Edge detection — subtle boundaries
    4. White card detection — white panels with shadow/border on light bg
    5. Rectangle detection — find rectangular UI components
    6. Header/footer detection — full-width bars at edges
    7. Color-block splitting — separate adjacent colored blocks
    8. Embedded image detection — high color-variance rectangular regions
    9. Sidebar detection — thin uniform-color strips along left/right edges
    10. Border-outlined rectangles — thin colored outlines (cyan, etc.)
    """

    def __init__(self, min_area: int = 200, merge_threshold: float = 0.3):
        self.min_area = min_area
        self.merge_threshold = merge_threshold

    def process(self, image: np.ndarray, context) -> object:
        img = context.working_image
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h, w = img.shape[:2]
        img_area = h * w

        # Compute page background luminance for contrast-based detection
        self._page_bg_lum = self._estimate_page_bg_lum(img)

        all_boxes = []

        # Strategy 1: Adaptive threshold
        all_boxes.extend(self._detect_from_threshold(gray))

        # Strategy 2: Color segmentation (with smaller kernels)
        all_boxes.extend(self._detect_from_color_segments(img))

        # Strategy 3: Canny edges (with smaller kernels)
        all_boxes.extend(self._detect_from_edges(gray))

        # Strategy 4: White cards on light background
        all_boxes.extend(self._detect_white_cards(gray, img))

        # Strategy 5: Rectangle detection
        all_boxes.extend(self._detect_rectangles(gray))

        # Strategy 6: Header/footer bars
        all_boxes.extend(self._detect_header_footer(img))

        # Strategy 7: Individual color blocks (separate adjacent colored regions)
        all_boxes.extend(self._detect_individual_color_blocks(img))

        # Strategy 8: Embedded image detection
        all_boxes.extend(self._detect_embedded_images(img, gray))

        # Strategy 9: Sidebar detection
        all_boxes.extend(self._detect_sidebars(img))

        # Strategy 10: Border-outlined rectangles
        all_boxes.extend(self._detect_border_outlined_rects(img))

        # Filter out too small / too large
        filtered = [
            b for b in all_boxes
            if self.min_area <= b.area <= img_area * 0.90
            and b.width >= 10 and b.height >= 10
        ]

        # Merge overlapping boxes (higher threshold = less aggressive)
        merged = merge_overlapping_boxes(filtered, iou_threshold=0.5)

        # Build hierarchy
        elements = self._build_hierarchy(merged)

        context.regions = elements
        return context

    def _estimate_page_bg_lum(self, img: np.ndarray) -> float:
        """Estimate page background luminance from edge pixels."""
        h, w = img.shape[:2]
        strip = min(15, h // 6, w // 6)
        edge_pixels = np.concatenate([
            img[:strip, :].reshape(-1, 3),
            img[-strip:, :].reshape(-1, 3),
            img[:, :strip].reshape(-1, 3),
            img[:, -strip:].reshape(-1, 3),
        ]).astype(np.float32)
        mean_c = edge_pixels.mean(axis=0)
        return 0.2126 * mean_c[0] + 0.7152 * mean_c[1] + 0.0722 * mean_c[2]

    def _detect_from_threshold(self, gray: np.ndarray) -> List[BoundingBox]:
        boxes = []
        for block_size in [11, 21, 51]:
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, blockSize=block_size, C=3
            )
            # Smaller kernel to avoid merging adjacent elements
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
            contours, _ = cv2.findContours(
                closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w * h >= self.min_area:
                    boxes.append(BoundingBox(x, y, w, h))
        return boxes

    def _detect_from_color_segments(self, img: np.ndarray) -> List[BoundingBox]:
        """Detect colored regions. Uses SMALL kernels to avoid merging
        adjacent cards/buttons of different colors."""
        boxes = []
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # Saturated colored regions (buttons, cards with color)
        mask_colored = cv2.inRange(hsv, (0, 40, 60), (180, 255, 255))
        # Mid-saturation regions (corporate UI muted buttons, light tinted cards)
        mask_mid_sat = cv2.inRange(hsv, (0, 20, 100), (180, 50, 240))
        # Very dark regions (sidebars, dark panels) — value cap 80 catches
        # dark panels like thin sidebars that have value 40-70
        mask_dark = cv2.inRange(hsv, (0, 0, 0), (180, 255, 80))

        for mask in [mask_colored, mask_mid_sat, mask_dark]:
            # Small kernel: prevents merging adjacent colored regions
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
            contours, _ = cv2.findContours(
                cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w * h >= self.min_area:
                    boxes.append(BoundingBox(x, y, w, h))
        return boxes

    def _detect_from_edges(self, gray: np.ndarray) -> List[BoundingBox]:
        """Canny edge detection with SMALL kernel to preserve region separation."""
        boxes = []
        edges = cv2.Canny(gray, 30, 100)
        # Smaller kernel + fewer iterations = less merging
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h >= self.min_area * 2:
                boxes.append(BoundingBox(x, y, w, h))
        return boxes

    def _detect_white_cards(self, gray: np.ndarray, img: np.ndarray) -> List[BoundingBox]:
        """Detect white card panels with subtle borders or box-shadows."""
        boxes = []

        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        magnitude = np.uint8(np.clip(magnitude, 0, 255))

        _, grad_thresh = cv2.threshold(magnitude, 8, 255, cv2.THRESH_BINARY)

        # Smaller kernel to avoid merging adjacent cards (8x8 instead of 12x12)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
        closed = cv2.morphologyEx(grad_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open, iterations=1)

        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area < self.min_area * 3:
                continue

            interior = gray[y + 5:y + h - 5, x + 5:x + w - 5]
            if interior.size > 0:
                mean_val = np.mean(interior)
                if mean_val > 170:
                    boxes.append(BoundingBox(x, y, w, h))

        return boxes

    def _detect_rectangles(self, gray: np.ndarray) -> List[BoundingBox]:
        """Detect rectangular shapes using contour approximation."""
        boxes = []

        for thresh_val in [200, 230, 245]:
            _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
            binary_inv = cv2.bitwise_not(binary)

            contours, _ = cv2.findContours(
                binary_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.min_area:
                    continue

                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect = w / h if h > 0 else 0
                    if 0.1 < aspect < 20:
                        boxes.append(BoundingBox(x, y, w, h))

        return boxes

    def _detect_individual_color_blocks(self, img: np.ndarray) -> List[BoundingBox]:
        """
        Strategy 7: Detect individual colored blocks by color clustering.

        This addresses the problem of adjacent colored cards (orange, green, blue)
        being merged into one big region. We segment unique colors and find
        connected components for each.
        """
        boxes = []
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h_img, w_img = img.shape[:2]

        # Define specific hue ranges for common UI colors
        hue_ranges = [
            (0, 15),    # Red/Orange
            (15, 35),   # Orange/Yellow
            (35, 75),   # Yellow/Green
            (75, 140),  # Green/Cyan
            (140, 170), # Blue
            (170, 180), # Purple/Magenta
        ]

        for h_lo, h_hi in hue_ranges:
            # Require decent saturation and brightness
            mask = cv2.inRange(hsv, (h_lo, 50, 60), (h_hi, 255, 255))

            if np.sum(mask) < 500:
                continue

            # Very small kernel: keep blocks separate
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

            contours, _ = cv2.findContours(
                cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = w * h
                if area < self.min_area:
                    continue
                # Must be rectangular enough (fill ratio > 40%)
                cnt_area = cv2.contourArea(cnt)
                if cnt_area / area > 0.4:
                    boxes.append(BoundingBox(x, y, w, h))

        return boxes

    def _detect_header_footer(self, img: np.ndarray) -> List[BoundingBox]:
        """Detect full-width header/footer bars at page edges."""
        boxes = []
        h, w = img.shape[:2]
        max_strip = min(80, h // 8)

        for strip_h in range(max_strip, 18, -4):
            top_strip = img[:strip_h, :]
            if self._strip_is_bar(top_strip):
                boxes.append(BoundingBox(0, 0, w, strip_h))
                break

        for strip_h in range(max_strip, 18, -4):
            bot_strip = img[h - strip_h:, :]
            if self._strip_is_bar(bot_strip):
                boxes.append(BoundingBox(0, h - strip_h, w, strip_h))
                break

        return boxes

    def _strip_is_bar(self, strip: np.ndarray) -> bool:
        """Detect if a strip is a header/footer bar.
        Uses contrast against page background instead of absolute luminance,
        so light headers (#f0f0f0) on white pages are still detected."""
        if strip.size == 0:
            return False
        pixels = strip.reshape(-1, 3).astype(np.float32)
        mean_c = pixels.mean(axis=0)
        lum = 0.2126 * mean_c[0] + 0.7152 * mean_c[1] + 0.0722 * mean_c[2]

        # Must have enough contrast against page background (> 15 luminance diff)
        bg_lum = getattr(self, '_page_bg_lum', 128)
        contrast = abs(lum - bg_lum)
        if contrast < 15:
            return False

        # Must be reasonably uniform (low std dev)
        std_dev = pixels.std(axis=0).mean()
        return std_dev < 55

    def _detect_embedded_images(self, img: np.ndarray,
                                gray: np.ndarray) -> List[BoundingBox]:
        """Strategy 8: Detect embedded photographic/image regions.

        Photographic images have high HUE DIVERSITY (many different colors)
        compared to flat UI elements which are mostly achromatic or single-hue.
        We create a "hue richness" map using block analysis and find connected
        regions of high hue diversity.
        """
        boxes = []
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # Create a local hue-diversity map using block analysis
        block_size = 30
        step = block_size // 2
        diversity_map = np.zeros((h, w), dtype=np.uint8)

        for by in range(0, h - block_size, step):
            for bx in range(0, w - block_size, step):
                block = hsv[by:by + block_size, bx:bx + block_size]
                sat = block[:, :, 1]
                val = block[:, :, 2]

                # Only consider pixels with enough saturation and brightness
                chromatic = (sat > 30) & (val > 40)
                n_chromatic = int(np.sum(chromatic))

                if n_chromatic < block_size * block_size * 0.15:
                    continue

                # Count distinct hues among chromatic pixels
                hues = block[:, :, 0][chromatic]
                # Bin hues into 18 bins (10 degrees each)
                hue_bins = np.bincount(hues // 10, minlength=18)
                n_distinct_hues = int(np.sum(hue_bins > 0))

                # Photographic: 3+ distinct hue bins (lowered from 4
                # to catch limited-palette photos like sunsets, monochrome subjects)
                if n_distinct_hues >= 3:
                    diversity_map[by:by + block_size,
                                  bx:bx + block_size] = 255

        if np.sum(diversity_map) == 0:
            return boxes

        # Close gaps and clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed = cv2.morphologyEx(
            diversity_map, cv2.MORPH_CLOSE, kernel, iterations=3
        )
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        cleaned = cv2.morphologyEx(
            closed, cv2.MORPH_OPEN, kernel_open, iterations=2
        )

        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            if area < self.min_area * 10:
                continue
            if area > h * w * 0.7:
                continue
            aspect = cw / ch if ch > 0 else 0
            if 0.15 < aspect < 10:
                boxes.append(BoundingBox(x, y, cw, ch))

        return boxes

    def _detect_sidebars(self, img: np.ndarray) -> List[BoundingBox]:
        """Strategy 9: Detect thin sidebars along left/right edges.

        Sidebars are tall, narrow strips with a distinct color from the
        page background. We scan for uniform-color vertical strips.
        """
        boxes = []
        h, w = img.shape[:2]
        max_sidebar_w = min(200, w // 4)
        min_sidebar_w = 15

        # Check left side
        for sw in range(max_sidebar_w, min_sidebar_w - 1, -5):
            strip = img[:, :sw]
            if self._is_sidebar_strip(img, strip, h, is_left=True, sw=sw):
                boxes.append(BoundingBox(0, 0, sw, h))
                break

        # Check right side
        for sw in range(max_sidebar_w, min_sidebar_w - 1, -5):
            strip = img[:, w - sw:]
            if self._is_sidebar_strip(img, strip, h, is_left=False, sw=sw):
                boxes.append(BoundingBox(w - sw, 0, sw, h))
                break

        return boxes

    def _is_sidebar_strip(self, full_img, strip: np.ndarray,
                          img_h: int, is_left: bool = True,
                          sw: int = 0) -> bool:
        """Check if a vertical strip is a sidebar.
        Must be: mostly uniform color, different from page background,
        AND have a visible edge boundary against the adjacent content."""
        if strip.size == 0:
            return False
        pixels = strip.reshape(-1, 3).astype(np.float32)
        mean_c = pixels.mean(axis=0)
        lum = 0.2126 * mean_c[0] + 0.7152 * mean_c[1] + 0.0722 * mean_c[2]

        # Reject very bright strips — a white/near-white region on a
        # light page is content area, not a sidebar
        if lum > 230:
            return False

        # Must contrast against page background
        bg_lum = getattr(self, '_page_bg_lum', 128)
        if abs(lum - bg_lum) < 20:
            return False

        # Must be reasonably uniform
        std_dev = pixels.std(axis=0).mean()
        if std_dev > 60:
            return False

        strip_w = strip.shape[1]
        if strip_w < 5:
            return False

        # The strip should be tall (at least 60% of image height)
        strip_h = strip.shape[0]
        if strip_h < img_h * 0.6:
            return False

        # Must have a visible edge boundary: compare last column of
        # the strip against the adjacent column in the full image
        h_img, w_img = full_img.shape[:2]
        sample_rows = np.linspace(0, strip_h - 1, min(50, strip_h),
                                  dtype=int)
        if is_left:
            # Right edge of left strip → compare col sw-1 vs col sw
            if sw < w_img:
                inner = full_img[sample_rows, sw - 1].astype(np.float32)
                outer = full_img[sample_rows, sw].astype(np.float32)
            else:
                return False
        else:
            # Left edge of right strip → compare col (w-sw) vs (w-sw-1)
            col = w_img - sw
            if col > 0:
                inner = full_img[sample_rows, col].astype(np.float32)
                outer = full_img[sample_rows, col - 1].astype(np.float32)
            else:
                return False

        edge_diff = np.mean(np.abs(inner - outer))
        if edge_diff < 15:
            return False

        return True

    def _detect_border_outlined_rects(self, img: np.ndarray) -> List[BoundingBox]:
        """Strategy 10: Detect rectangles outlined with thin colored borders.

        UI buttons and cards often have thin colored outlines (cyan, blue, etc.)
        that are just 1-3px wide. Standard contour detection misses these.
        We use Canny with tight parameters and look for closed rectangular shapes.
        """
        boxes = []
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # Focus on saturated pixels (colored borders)
        sat_mask = cv2.inRange(hsv, (0, 80, 100), (180, 255, 255))

        # Thin borders: dilate only slightly to connect border pixels
        kernel_thin = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(sat_mask, kernel_thin, iterations=1)

        # Close to connect border segments into rectangles
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close, iterations=2)

        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue

            peri = cv2.arcLength(cnt, True)
            if peri == 0:
                continue
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

            # Accept rectangles (4 corners) and near-rectangles (4-8 corners)
            if 4 <= len(approx) <= 8:
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect = bw / bh if bh > 0 else 0
                if 0.2 < aspect < 15:
                    # Verify the border: edge pixels should be more saturated
                    # than interior
                    if bw > 10 and bh > 10:
                        edge_strip = 3
                        interior = img[y + edge_strip:y + bh - edge_strip,
                                       x + edge_strip:x + bw - edge_strip]
                        if interior.size > 0:
                            int_hsv = cv2.cvtColor(interior, cv2.COLOR_RGB2HSV)
                            int_sat = float(int_hsv[:, :, 1].mean())
                            # Border should be more saturated than interior
                            # (interior is usually white/light gray)
                            edge_hsv = hsv[y:y + bh, x:x + bw]
                            edge_sat = float(edge_hsv[:, :, 1].mean())
                            if edge_sat > int_sat or edge_sat > 30:
                                boxes.append(BoundingBox(x, y, bw, bh))

        return boxes

    def _build_hierarchy(self, boxes: List[BoundingBox]) -> List[DetectedElement]:
        from ..utils.geometry import _containment_ratio

        sorted_boxes = sorted(boxes, key=lambda b: b.area)
        elements = []

        for bbox in sorted_boxes:
            elem = DetectedElement(
                id=str(uuid.uuid4())[:8],
                element_type=ElementType.CONTAINER,
                bbox=bbox
            )
            elements.append(elem)

        for i, elem in enumerate(elements):
            best_parent = None
            best_area = float('inf')
            for j, candidate in enumerate(elements):
                if i == j:
                    continue
                if candidate.bbox.area <= elem.bbox.area:
                    continue
                # Accept both strict containment and near-containment (85%+)
                if candidate.bbox.contains(elem.bbox):
                    if candidate.bbox.area < best_area:
                        best_parent = candidate
                        best_area = candidate.bbox.area
                elif _containment_ratio(candidate.bbox, elem.bbox) > 0.85:
                    if candidate.bbox.area < best_area:
                        best_parent = candidate
                        best_area = candidate.bbox.area

            if best_parent:
                elem.parent_id = best_parent.id
                best_parent.children.append(elem)

        return [e for e in elements if e.parent_id is None]
