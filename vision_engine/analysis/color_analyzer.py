import cv2
import numpy as np
from typing import List, Tuple

from ..models.elements import ColorInfo, DetectedElement, ElementType
from ..utils.image_utils import crop_region, luminance


def _rgb_saturation(rgb):
    """Return HSV saturation (0-255) for an RGB tuple."""
    r, g, b = rgb
    mx = max(r, g, b)
    mn = min(r, g, b)
    if mx == 0:
        return 0
    return int(255 * (mx - mn) / mx)


class ColorAnalyzer:
    """
    Step 4: Extract dominant colors per region using K-means clustering.
    - Larger sample size (30k pixels) for accuracy
    - Adaptive k based on region complexity
    - Otsu thresholding for text color extraction
    - Preserves off-white backgrounds instead of snapping to #fff
    """

    def __init__(self, n_clusters: int = 5, seed: int = 42):
        self.n_clusters = n_clusters
        self.rng = np.random.RandomState(seed)

    def process(self, image: np.ndarray, context) -> object:
        context.page_background = self._extract_page_background(
            context.working_image
        )

        for element in self._walk(context.regions):
            crop = crop_region(context.working_image, element.bbox)
            if crop.size == 0:
                continue

            # For large containers, prefer edge-sampled background
            # to avoid children's colors dominating k-means
            b = element.bbox
            if b.area > 50000 and element.children:
                bg = self._exposed_bg(context.working_image, b, element.children)
                colors = self._kmeans_colors(crop)
                fg = colors[1] if len(colors) > 1 else None
                element.color = ColorInfo(
                    background=bg,
                    foreground=fg,
                    border=self._detect_border_color(
                        context.working_image, b
                    ),
                )
            else:
                colors = self._kmeans_colors(crop)
                border_color = self._detect_border_color(
                    context.working_image, element.bbox
                )
                element.color = ColorInfo(
                    background=colors[0],
                    foreground=colors[1] if len(colors) > 1 else None,
                    border=border_color,
                )

            # Detect gradient background (for elements without bg-image override)
            if not getattr(element, 'bg_image_data_uri', None):
                gradient = self._detect_gradient(
                    context.working_image, element.bbox, element
                )
                if gradient and element.color:
                    element.color.gradient = gradient

        # Text element colors — use Otsu for reliable text/bg separation
        for text_elem in context.text_elements:
            text_elem.color = self._extract_text_color(
                context.working_image, text_elem.bbox
            )

        return context

    def _extract_text_color(self, image, bbox):
        """Extract text color using 3-class k-means on the full crop.

        A text crop can contain up to 3 distinct regions:
          (a) page/outer background (if bbox straddles a container edge)
          (b) container/inner background (behind the text)
          (c) actual text glyph pixels

        We use k-means with k=3 on all pixels.  The largest cluster
        is background; the smallest cluster that is sufficiently
        distant from the largest is the text color.
        """
        crop = crop_region(image, bbox, margin=2)
        if crop.size < 10:
            return ColorInfo(background=(255, 255, 255), foreground=(0, 0, 0))

        pixels = crop.reshape(-1, 3).astype(np.float32)
        n = len(pixels)

        # Subsample for speed on large crops
        if n > 5000:
            idx = self.rng.choice(n, 5000, replace=False)
            sample = pixels[idx]
        else:
            sample = pixels

        k = min(3, len(sample))
        # More iterations for small crops (< 1000 pixels) for better convergence
        n_attempts = 15 if n < 1000 else 10
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            80, 0.3
        )
        _, labels_s, centers = cv2.kmeans(
            sample, k, None, criteria, n_attempts, cv2.KMEANS_PP_CENTERS
        )
        counts = np.bincount(labels_s.flatten(), minlength=k)

        # Background = largest cluster
        bg_idx = int(np.argmax(counts))
        bg = tuple(int(c) for c in centers[bg_idx])

        # Foreground = the cluster most distant from background,
        # with at least a few pixels
        bg_arr = centers[bg_idx]
        min_count = max(3, n // 50)  # at least 2% of pixels
        best_idx = -1
        best_dist = -1.0
        for i in range(k):
            if i == bg_idx:
                continue
            if counts[i] < min_count:
                continue
            dist = float(np.sqrt(np.sum((centers[i] - bg_arr) ** 2)))
            if dist > best_dist:
                best_dist = dist
                best_idx = i

        if best_idx >= 0:
            fg = tuple(int(c) for c in centers[best_idx])
        else:
            lum_bg = luminance(bg)
            fg = (255, 255, 255) if lum_bg < 128 else (0, 0, 0)

        # If there are 3 clusters whose luminances span a wide range,
        # the crop likely straddles a container edge (page bg + panel
        # bg + text).  The text is the cluster whose luminance sits
        # between the brightest and darkest, and the local background
        # is whichever non-text cluster is closer in luminance.
        #
        # SKIP this correction if the initial bg has high saturation —
        # that means the text sits on a colored label (green "SYSTEM
        # NORMAL", red "ALARMS", etc.) and the initial 2-cluster split
        # correctly identified the colored bg + white text.
        bg_sat = _rgb_saturation(bg)
        if k == 3 and bg_sat < 60:
            lums = [
                luminance(tuple(int(c) for c in centers[i]))
                for i in range(3)
            ]
            sorted_by_lum = sorted(range(3), key=lambda i: lums[i])
            dark, mid, bright = sorted_by_lum

            lum_range = lums[bright] - lums[dark]
            mid_pos = (lums[mid] - lums[dark]) / max(lum_range, 1)

            # The middle cluster is text if:
            # 1. The luminance range is wide (> 50)
            # 2. The middle cluster sits genuinely between (15%-85%)
            # 3. The middle cluster has enough pixels
            if (lum_range > 50 and 0.15 < mid_pos < 0.85
                    and counts[mid] >= min_count):
                fg = tuple(int(c) for c in centers[mid])
                # Local bg = the non-text cluster closest to text
                d_dark = abs(lums[mid] - lums[dark])
                d_bright = abs(lums[mid] - lums[bright])
                local_bg_idx = dark if d_dark < d_bright else bright
                bg = tuple(int(c) for c in centers[local_bg_idx])

        # Safety: if bg and fg ended up essentially identical
        color_dist = sum((a - b) ** 2 for a, b in zip(bg, fg)) ** 0.5
        if color_dist < 25:
            lum_bg = luminance(bg)
            fg = (255, 255, 255) if lum_bg < 128 else (0, 0, 0)

        # Extra contrast boost: when bg is light and fg is only slightly
        # darker (gray-on-white), push fg toward a readable dark gray.
        # This handles washed-out text detections where k-means picks
        # a cluster that is still light-ish.
        lum_bg = luminance(bg)
        lum_fg = luminance(fg)
        lum_diff = abs(lum_bg - lum_fg)
        if lum_bg > 200 and lum_fg > 120 and lum_diff < 100:
            # fg is too light on a light bg — darken it
            factor = 0.55  # push toward darker
            fg = tuple(int(c * factor) for c in fg)

        return ColorInfo(background=bg, foreground=fg)

    def _kmeans_colors(self, crop: np.ndarray,
                       k: int = None) -> List[Tuple[int, int, int]]:
        """Run K-means on pixel colors, return sorted by cluster size."""
        k = k or self.n_clusters
        pixels = crop.reshape(-1, 3).astype(np.float32)

        # Larger sample for better accuracy
        max_samples = 30000
        if len(pixels) > max_samples:
            indices = self.rng.choice(len(pixels), max_samples, replace=False)
            pixels = pixels[indices]

        # Adaptive k: don't use more clusters than meaningful
        # Always use at least k=2 (background + foreground possible)
        pixel_std = pixels.std(axis=0).mean()
        if pixel_std < 10:
            k = 2  # Nearly uniform
        elif pixel_std < 25:
            k = 3
        k = max(2, k)  # Never go below 2

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            100, 0.2
        )
        k = min(k, len(pixels))
        if k < 1:
            return [(255, 255, 255)]

        _, labels, centers = cv2.kmeans(
            pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )

        counts = np.bincount(labels.flatten(), minlength=k)
        sorted_indices = np.argsort(-counts)

        return [
            tuple(int(c) for c in centers[idx])
            for idx in sorted_indices
        ]

    def _exposed_bg(self, image, bbox, children):
        """Extract background from pixels NOT covered by any child element.

        For large containers the edge pixels are the classic sampling region,
        but if the element has a white margin at the top (padding / shadow) the
        edge sample becomes near-white even though the real background is a
        distinct colour.  Sampling the exposed interior (gaps between children)
        gives us the true parent background.

        Falls back to _edge_sampled_bg when the exposed area is too small.
        """
        h, w = image.shape[:2]
        x1, y1 = max(0, bbox.x), max(0, bbox.y)
        x2, y2 = min(w, bbox.x2), min(h, bbox.y2)

        if x2 - x1 < 10 or y2 - y1 < 10:
            return self._edge_sampled_bg(image, bbox)

        # Build boolean mask of the element region; mark child pixels as covered
        region_h = y2 - y1
        region_w = x2 - x1
        covered = np.zeros((region_h, region_w), dtype=bool)
        # Exclude the top and bottom margins (10 % each) from sampling.
        # Many elements have white padding/shadow at the border; including
        # those pixels washes out the real background colour in lateral strips.
        margin = max(2, region_h // 10)
        covered[:margin, :] = True
        covered[region_h - margin:, :] = True
        for child in children:
            cx1 = max(0, child.bbox.x - x1)
            cy1 = max(0, child.bbox.y - y1)
            cx2 = min(region_w, child.bbox.x2 - x1)
            cy2 = min(region_h, child.bbox.y2 - y1)
            if cx2 > cx1 and cy2 > cy1:
                covered[cy1:cy2, cx1:cx2] = True

        exposed_mask = ~covered
        if exposed_mask.sum() < 50:
            # No useful exposed area — fall back to edge sampling
            return self._edge_sampled_bg(image, bbox)

        exposed_pixels = image[y1:y2, x1:x2][exposed_mask]
        if len(exposed_pixels) > 5000:
            idx = self.rng.choice(len(exposed_pixels), 5000, replace=False)
            exposed_pixels = exposed_pixels[idx]

        exposed_mean = exposed_pixels.mean(0)
        edge_bg = self._edge_sampled_bg(image, bbox)

        # Compare the two candidates; prefer exposed mean when it is clearly
        # more saturated than the edge-sampled result (the edge can be
        # dominated by a white top-margin that isn't the real background).
        # Also prefer exposed when it is significantly lighter than the edge
        # estimate — the edge may be dominated by very dark border pixels that
        # aren't representative of the real interior background (e.g. dark SCADA
        # panels whose instrument gaps are medium-dark, not near-black).
        def _sat(rgb):
            mx = max(rgb)
            return (mx - min(rgb)) / mx * 255 if mx > 0 else 0

        exp_sat = _sat(exposed_mean)
        edge_sat = _sat(edge_bg)
        exp_lum = 0.2126 * exposed_mean[0] + 0.7152 * exposed_mean[1] + 0.0722 * exposed_mean[2]
        edge_lum = 0.2126 * edge_bg[0] + 0.7152 * edge_bg[1] + 0.0722 * edge_bg[2]
        lum_advantage = exp_lum - edge_lum

        if exp_sat > max(edge_sat + 15, 30) and exp_lum < 230:
            return tuple(int(c) for c in exposed_mean)
        # Exposed pixels significantly lighter than edge → edge is dominated by
        # dark border pixels, exposed interior is more representative.
        if lum_advantage > 20 and exp_lum < 220:
            return tuple(int(c) for c in exposed_mean)
        return edge_bg

    def _edge_sampled_bg(self, image, bbox):
        """Extract background color from the edges of a region.
        For large containers, edge pixels better represent the actual
        background since children's colors dominate the interior."""
        h, w = image.shape[:2]
        x1, y1 = max(0, bbox.x), max(0, bbox.y)
        x2, y2 = min(w, bbox.x2), min(h, bbox.y2)

        if x2 - x1 < 10 or y2 - y1 < 10:
            return (200, 200, 200)

        strip = max(3, min(8, (x2 - x1) // 10, (y2 - y1) // 10))

        # Sample pixels from the 4 edges (inner border strip)
        edge_pixels = np.concatenate([
            image[y1:y1+strip, x1:x2].reshape(-1, 3),       # top
            image[y2-strip:y2, x1:x2].reshape(-1, 3),       # bottom
            image[y1:y2, x1:x1+strip].reshape(-1, 3),       # left
            image[y1:y2, x2-strip:x2].reshape(-1, 3),       # right
        ])

        if len(edge_pixels) == 0:
            return (200, 200, 200)

        # Use k-means with k=2 on edge pixels
        pixels_f = edge_pixels.astype(np.float32)
        if len(pixels_f) > 5000:
            idx = self.rng.choice(len(pixels_f), 5000, replace=False)
            pixels_f = pixels_f[idx]

        k = min(2, len(pixels_f))
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            50, 0.5
        )
        _, labels, centers = cv2.kmeans(
            pixels_f, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS
        )
        counts = np.bincount(labels.flatten(), minlength=k)

        # The dominant edge color is the background
        bg_idx = int(np.argmax(counts))
        return tuple(int(c) for c in centers[bg_idx])

    def _extract_page_background(self, image: np.ndarray
                                  ) -> Tuple[int, int, int]:
        """Extract page background from edge regions.

        Uses the dominant (most-common) color cluster as the background.
        Works for both light UIs (white/light-gray bg) and dark UIs
        (dark bg). _kmeans_colors returns clusters sorted by count
        descending, so colors[0] is already the dominant color.

        Previous logic used brightest = max(luminance) which failed for
        dark UIs like SCADA interfaces where the background is dark.
        """
        h, w = image.shape[:2]

        strip = min(20, h // 4, w // 4)
        edge_pixels = np.concatenate([
            image[:strip, :].reshape(-1, 3),
            image[-strip:, :].reshape(-1, 3),
            image[:, :strip].reshape(-1, 3),
            image[:, -strip:].reshape(-1, 3),
        ])

        tc_y1, tc_y2 = h // 4, h // 2
        tc_x1, tc_x2 = w // 4, 3 * w // 4
        content_pixels = image[tc_y1:tc_y2, tc_x1:tc_x2].reshape(-1, 3)

        all_pixels = np.concatenate([
            edge_pixels, edge_pixels, edge_pixels, content_pixels
        ])

        colors = self._kmeans_colors(
            all_pixels.reshape(1, -1, 3).squeeze(), k=3
        )

        # colors[0] is the most-common cluster (dominant background color).
        dominant = colors[0]

        # Snap near-white to pure white (#ffffff) to avoid off-white artifacts
        if luminance(dominant) > 250:
            r, g, b = dominant
            if r > 250 and g > 250 and b > 250:
                dominant = (255, 255, 255)

        return dominant

    def _detect_border_color(self, image, bbox):
        """Detect visible borders by comparing 2px edge strips to interior."""
        h, w = image.shape[:2]
        x1 = max(0, bbox.x)
        y1 = max(0, bbox.y)
        x2 = min(w, bbox.x2)
        y2 = min(h, bbox.y2)

        if x2 - x1 < 10 or y2 - y1 < 10:
            return None

        border_w = 3  # Sample 3px edges to catch thick borders
        strips = []
        if y1 + border_w < y2:
            strips.append(image[y1:y1 + border_w, x1:x2].reshape(-1, 3))
        if y2 - border_w > y1:
            strips.append(image[y2 - border_w:y2, x1:x2].reshape(-1, 3))
        if x1 + border_w < x2:
            strips.append(image[y1:y2, x1:x1 + border_w].reshape(-1, 3))
        if x2 - border_w > x1:
            strips.append(image[y1:y2, x2 - border_w:x2].reshape(-1, 3))

        if not strips:
            return None

        edge_pixels = np.concatenate(strips)

        interior = image[y1 + 4:y2 - 4, x1 + 4:x2 - 4]
        if interior.size == 0 or edge_pixels.size == 0:
            return None
        interior = interior.reshape(-1, 3)

        edge_mean = np.mean(edge_pixels, axis=0)
        interior_mean = np.mean(interior, axis=0)

        color_diff = np.sqrt(np.sum((edge_mean - interior_mean) ** 2))
        if color_diff > 20:
            return tuple(int(c) for c in edge_mean)
        return None

    def _detect_gradient(self, image: np.ndarray, bbox, element=None) -> 'Optional[str]':
        """Detect if a region has a gradient background.

        Divides the region into thirds (horizontal and vertical) and
        measures the average color in each third. If colors shift
        monotonically with sufficient magnitude, returns a CSS
        linear-gradient string. Only detects gradients for elements
        without container children (leaf-level containers and pure visuals).

        Only triggers on LOW-SATURATION elements (dark/gray/steel panels).
        High-saturation flat-design colors (yellow/green/blue buttons, cards)
        are solid colors and must NOT be detected as having gradients.

        Returns None if no significant gradient is found.
        """
        # Skip elements with container children — their color sampling
        # is dominated by children's colors, not the background gradient.
        # Exception: very tall/narrow elements (aspect < 0.35) like sidebars
        # where the background gradient runs top-to-bottom and children don't
        # cover the full width — the vertical gradient can still be detected.
        if element is not None and element.children:
            has_container_child = any(
                c.element_type.value not in ("text",)
                for c in element.children
            )
            if has_container_child:
                b = element.bbox
                aspect = b.width / max(b.height, 1)
                if aspect >= 0.35:
                    return None
                # Tall narrow sidebar: fall through to gradient detection below

        # Guard: skip high-saturation OR light elements.
        #
        # High-saturation BRIGHT flat-design colors (yellow/green/blue buttons)
        # are solid colors, not gradient. But saturated DARK colors (dark purple,
        # dark navy sidebars) CAN have genuine gradients.
        #
        # Light backgrounds (lum > 120): white/off-white/light-gray areas are
        # effectively flat — any detected "gradient" is from JPEG artifacts.
        if element is not None and element.color and element.color.background:
            bg = element.color.background
            mx = max(bg)
            mn = min(bg)
            sat = (mx - mn) / mx if mx > 0 else 0.0
            lum = 0.2126 * bg[0] + 0.7152 * bg[1] + 0.0722 * bg[2]
            # Skip: light (lum > 120) — always flat artifacts
            if lum > 120:
                return None
            # Skip: saturated AND bright (lum > 60) — flat colored buttons/cards
            # Allow: saturated AND dark (lum <= 60) — dark purple/navy sidebars
            if sat > 0.55 and lum > 60:
                return None

        h, w = image.shape[:2]
        x1, y1 = max(0, bbox.x), max(0, bbox.y)
        x2, y2 = min(w, bbox.x2), min(h, bbox.y2)
        reg_w, reg_h = x2 - x1, y2 - y1

        if reg_w < 40 or reg_h < 20:
            return None

        # Only apply gradients to elements with significant area.
        # Small elements (<15000 px = ~120x120) have too little area
        # for reliable gradient sampling; detected gradients on small
        # elements are mostly JPEG compression noise, not real gradients.
        if reg_w * reg_h < 15000:
            return None

        # --- Horizontal gradient check (left → right) ---
        third_w = reg_w // 3
        if third_w >= 5:
            c_left = tuple(int(v) for v in
                           np.mean(image[y1:y2, x1:x1 + third_w].reshape(-1, 3), axis=0))
            c_mid = tuple(int(v) for v in
                          np.mean(image[y1:y2, x1 + third_w:x1 + 2 * third_w].reshape(-1, 3), axis=0))
            c_right = tuple(int(v) for v in
                            np.mean(image[y1:y2, x2 - third_w:x2].reshape(-1, 3), axis=0))

            h_diff = sum(abs(a - b) for a, b in zip(c_left, c_right))
            if h_diff >= 30:
                # Check monotonicity: mid should lie between left and right
                is_mono = all(
                    min(c_left[i], c_right[i]) - 20 <= c_mid[i] <= max(c_left[i], c_right[i]) + 20
                    for i in range(3)
                )
                if is_mono:
                    left_hex = '#{:02x}{:02x}{:02x}'.format(*c_left)
                    right_hex = '#{:02x}{:02x}{:02x}'.format(*c_right)
                    return f'linear-gradient(to right, {left_hex}, {right_hex})'

        # --- Vertical gradient check (top → bottom) ---
        third_h = reg_h // 3
        if third_h >= 5:
            c_top = tuple(int(v) for v in
                          np.mean(image[y1:y1 + third_h, x1:x2].reshape(-1, 3), axis=0))
            c_vmid = tuple(int(v) for v in
                           np.mean(image[y1 + third_h:y1 + 2 * third_h, x1:x2].reshape(-1, 3), axis=0))
            c_bot = tuple(int(v) for v in
                          np.mean(image[y2 - third_h:y2, x1:x2].reshape(-1, 3), axis=0))

            v_diff = sum(abs(a - b) for a, b in zip(c_top, c_bot))
            if v_diff >= 30:
                is_mono = all(
                    min(c_top[i], c_bot[i]) - 20 <= c_vmid[i] <= max(c_top[i], c_bot[i]) + 20
                    for i in range(3)
                )
                if is_mono:
                    top_hex = '#{:02x}{:02x}{:02x}'.format(*c_top)
                    bot_hex = '#{:02x}{:02x}{:02x}'.format(*c_bot)
                    return f'linear-gradient(to bottom, {top_hex}, {bot_hex})'

        return None

    def _walk(self, elements):
        for elem in elements:
            yield elem
            yield from self._walk(elem.children)
