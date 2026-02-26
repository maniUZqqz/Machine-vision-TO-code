"""
Icon Matcher: Matches detected icon elements to CDN icon libraries
using shape analysis (edge detection, contour features, aspect ratio).

Uses Bootstrap Icons (bi) as the primary CDN library.
Falls back to embedded base64 image if no match is found.
"""

import cv2
import numpy as np

from ..utils.image_utils import crop_region


# Icon feature signatures: each entry describes visual properties
# (aspect_ratio_range, contour_count_range, fill_ratio_range, has_circle, category)
# Categories map to common Bootstrap Icons
ICON_SIGNATURES = {
    # Navigation / Menu
    'bi-list': {
        'aspect': (0.7, 1.4), 'fill': (0.15, 0.45),
        'h_lines': 3, 'description': 'hamburger menu',
    },
    'bi-chevron-down': {
        'aspect': (0.8, 2.0), 'fill': (0.05, 0.30),
        'v_shape': True, 'description': 'chevron down',
    },
    'bi-chevron-right': {
        'aspect': (0.5, 1.2), 'fill': (0.05, 0.30),
        'description': 'chevron right',
    },
    'bi-x': {
        'aspect': (0.7, 1.4), 'fill': (0.05, 0.25),
        'cross': True, 'description': 'close / X',
    },
    'bi-arrow-left': {
        'aspect': (1.0, 3.0), 'fill': (0.10, 0.40),
        'description': 'arrow left',
    },
    # Search
    'bi-search': {
        'aspect': (0.7, 1.4), 'fill': (0.10, 0.35),
        'has_circle': True, 'description': 'search / magnifier',
    },
    # User / Person
    'bi-person': {
        'aspect': (0.6, 1.2), 'fill': (0.15, 0.50),
        'has_circle': True, 'description': 'person / user',
    },
    'bi-person-circle': {
        'aspect': (0.8, 1.2), 'fill': (0.30, 0.70),
        'has_circle': True, 'description': 'person circle / avatar',
    },
    # Settings / Gear
    'bi-gear': {
        'aspect': (0.8, 1.2), 'fill': (0.20, 0.50),
        'has_circle': True, 'description': 'settings / gear',
    },
    # Bell / Notification
    'bi-bell': {
        'aspect': (0.6, 1.1), 'fill': (0.25, 0.55),
        'description': 'notification / bell',
    },
    # Home
    'bi-house': {
        'aspect': (0.8, 1.3), 'fill': (0.20, 0.50),
        'description': 'home / house',
    },
    # Mail / Envelope
    'bi-envelope': {
        'aspect': (1.1, 2.0), 'fill': (0.15, 0.50),
        'description': 'email / envelope',
    },
    # Plus / Add
    'bi-plus': {
        'aspect': (0.7, 1.4), 'fill': (0.08, 0.30),
        'cross': True, 'description': 'add / plus',
    },
    # Star
    'bi-star': {
        'aspect': (0.8, 1.3), 'fill': (0.15, 0.45),
        'description': 'star / favorite',
    },
    # Heart
    'bi-heart': {
        'aspect': (0.9, 1.4), 'fill': (0.15, 0.50),
        'description': 'heart / like',
    },
    # Trash / Delete
    'bi-trash': {
        'aspect': (0.6, 1.0), 'fill': (0.15, 0.40),
        'description': 'trash / delete',
    },
    # Edit / Pencil
    'bi-pencil': {
        'aspect': (0.7, 1.4), 'fill': (0.10, 0.35),
        'description': 'edit / pencil',
    },
    # Calendar
    'bi-calendar': {
        'aspect': (0.8, 1.2), 'fill': (0.20, 0.55),
        'description': 'calendar / date',
    },
    # Check
    'bi-check': {
        'aspect': (0.8, 2.0), 'fill': (0.05, 0.25),
        'description': 'check / done',
    },
    # Download
    'bi-download': {
        'aspect': (0.7, 1.3), 'fill': (0.10, 0.35),
        'description': 'download',
    },
    # Upload
    'bi-upload': {
        'aspect': (0.7, 1.3), 'fill': (0.10, 0.35),
        'description': 'upload',
    },
    # Eye / Visibility
    'bi-eye': {
        'aspect': (1.2, 2.5), 'fill': (0.15, 0.45),
        'has_circle': True, 'description': 'eye / visible',
    },
    # Filter
    'bi-funnel': {
        'aspect': (0.7, 1.3), 'fill': (0.15, 0.40),
        'description': 'filter / funnel',
    },
}


class IconMatcher:
    """
    Matches icon-sized image elements to Bootstrap Icons using
    shape analysis with OpenCV.
    """

    CDN_URL = "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css"
    LIBRARY_NAME = "bootstrap-icons"

    def match_icon(self, image: np.ndarray, element) -> bool:
        """
        Analyze an icon element and try to match it to a CDN icon.

        Returns True if a match was found, False otherwise.
        Sets element.icon_name and element.icon_library on success.
        """
        if not element.is_icon:
            return False

        crop = crop_region(image, element.bbox)
        if crop.size == 0:
            return False

        # Extract features from the icon image
        features = self._extract_features(crop)
        if features is None:
            return False

        # Try to match features against known icon signatures
        best_match = self._find_best_match(features)
        if best_match:
            element.icon_name = best_match
            element.icon_library = self.LIBRARY_NAME
            return True

        return False

    def _extract_features(self, crop):
        """Extract visual features from an icon crop."""
        if crop.ndim == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        else:
            gray = crop.copy()

        h, w = gray.shape[:2]
        if h < 8 or w < 8:
            return None

        # Basic features
        aspect_ratio = w / max(h, 1)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Binary threshold (adaptive for icons)
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Fill ratio: percentage of dark/foreground pixels
        fill_ratio = np.sum(binary > 0) / max(binary.size, 1)

        # Contour analysis
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        contour_count = len(contours)

        # Circle detection
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, w // 2,
                                    param1=80, param2=20,
                                    minRadius=w // 6,
                                    maxRadius=w // 2 + 2)
        has_circle = circles is not None and len(circles[0]) > 0

        # Horizontal lines detection (for hamburger menu)
        h_lines = self._count_horizontal_lines(binary, h, w)

        # Cross / X detection
        has_cross = self._detect_cross(binary, h, w)

        # V-shape detection (for chevrons)
        has_v_shape = self._detect_v_shape(edges, h, w)

        return {
            'aspect': aspect_ratio,
            'fill': fill_ratio,
            'contour_count': contour_count,
            'has_circle': has_circle,
            'h_lines': h_lines,
            'has_cross': has_cross,
            'has_v_shape': has_v_shape,
        }

    def _count_horizontal_lines(self, binary, h, w):
        """Count horizontal line-like features (for hamburger menu detection)."""
        # Project binary horizontally
        h_proj = np.sum(binary > 0, axis=1)
        threshold = w * 0.3  # Line must span at least 30% width

        # Count runs of pixels above threshold
        in_line = False
        count = 0
        for val in h_proj:
            if val >= threshold:
                if not in_line:
                    count += 1
                    in_line = True
            else:
                in_line = False
        return count

    def _detect_cross(self, binary, h, w):
        """Detect X or + shape pattern."""
        # Check if there's significant content on both diagonals
        center_y, center_x = h // 2, w // 2
        quarter_h, quarter_w = h // 4, w // 4

        # Vertical line check
        v_strip = binary[quarter_h:h - quarter_h,
                         center_x - 2:center_x + 3]
        v_ratio = np.sum(v_strip > 0) / max(v_strip.size, 1)

        # Horizontal line check
        h_strip = binary[center_y - 2:center_y + 3,
                         quarter_w:w - quarter_w]
        h_ratio = np.sum(h_strip > 0) / max(h_strip.size, 1)

        return v_ratio > 0.4 and h_ratio > 0.4

    def _detect_v_shape(self, edges, h, w):
        """Detect V/chevron shape using edge analysis."""
        # Bottom half should have converging edges
        bottom_half = edges[h // 2:, :]
        if bottom_half.size == 0:
            return False

        # Check for converging edge pixels (more spread at top, narrower at bottom)
        top_row_spread = np.sum(bottom_half[0, :] > 0)
        mid_row = bottom_half.shape[0] // 2
        if mid_row < 1:
            return False
        mid_row_spread = np.sum(bottom_half[mid_row, :] > 0)

        return top_row_spread > mid_row_spread * 1.5

    def _find_best_match(self, features):
        """Find the best matching icon name for extracted features."""
        best_score = 0
        best_name = None

        for icon_name, sig in ICON_SIGNATURES.items():
            score = self._compute_match_score(features, sig)
            if score > best_score:
                best_score = score
                best_name = icon_name

        # Require minimum confidence threshold
        if best_score >= 0.5:
            return best_name

        # Fallback: use generic icon based on dominant feature
        return self._generic_fallback(features)

    def _compute_match_score(self, features, sig):
        """Compute match score between features and an icon signature."""
        score = 0
        checks = 0

        # Aspect ratio
        a_min, a_max = sig.get('aspect', (0.5, 2.0))
        if a_min <= features['aspect'] <= a_max:
            score += 1
        checks += 1

        # Fill ratio
        if 'fill' in sig:
            f_min, f_max = sig['fill']
            if f_min <= features['fill'] <= f_max:
                score += 1
        checks += 1

        # Circle presence
        if 'has_circle' in sig:
            if features['has_circle'] == sig['has_circle']:
                score += 1.5  # Higher weight for distinctive feature
            else:
                score -= 0.5
        checks += 1

        # Horizontal lines (hamburger)
        if 'h_lines' in sig:
            if features['h_lines'] >= sig['h_lines']:
                score += 2  # Very distinctive
        checks += 1

        # Cross/X shape
        if 'cross' in sig:
            if features['has_cross'] == sig['cross']:
                score += 2
        checks += 1

        # V-shape (chevron)
        if 'v_shape' in sig:
            if features['has_v_shape'] == sig['v_shape']:
                score += 2
        checks += 1

        return score / max(checks, 1)

    def _generic_fallback(self, features):
        """Return a generic icon name based on dominant features."""
        # Hamburger menu
        if features['h_lines'] >= 3:
            return 'bi-list'

        # Cross/X
        if features['has_cross']:
            if features['fill'] < 0.15:
                return 'bi-plus'
            return 'bi-x'

        # Chevron
        if features['has_v_shape']:
            return 'bi-chevron-down'

        # Circle with something
        if features['has_circle']:
            if features['fill'] > 0.4:
                return 'bi-person-circle'
            return 'bi-search'

        # No strong match → return None (keep embedded image)
        return None
