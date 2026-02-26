"""
Feedback Loop: Renders the generated HTML, compares it pixel-by-pixel
with the original screenshot, identifies discrepancy regions, and
applies automatic corrections to improve output fidelity.

Uses Playwright for headless HTML rendering and OpenCV for comparison.
"""

import os
import tempfile

import cv2
import numpy as np


class FeedbackLoop:
    """
    Iteratively improves HTML output by comparing rendered output
    with the original screenshot.
    """

    def __init__(self, max_iterations=3, similarity_threshold=0.85):
        self.max_iterations = max_iterations
        self.similarity_threshold = similarity_threshold

    def run(self, original_image: np.ndarray, html_content: str,
            css_rules: list, page_elements: list) -> dict:
        """
        Run the feedback loop.

        Args:
            original_image: Original screenshot (RGB numpy array)
            html_content: Generated HTML string
            css_rules: CSS rules list from CSSBuilder
            page_elements: List of root DetectedElement objects

        Returns:
            dict with 'html', 'corrections', 'similarity_scores'
        """
        h, w = original_image.shape[:2]
        corrections = []
        similarity_scores = []

        current_html = html_content

        for iteration in range(self.max_iterations):
            # Render current HTML to image
            rendered = self._render_html(current_html, w, h)
            if rendered is None:
                break

            # Compare with original
            similarity, diff_map, diff_regions = self._compare_images(
                original_image, rendered
            )
            similarity_scores.append(similarity)

            # Stop if good enough
            if similarity >= self.similarity_threshold:
                break

            # Analyze diff regions and generate corrections
            iter_corrections = self._analyze_differences(
                original_image, rendered, diff_regions, page_elements
            )

            if not iter_corrections:
                break

            corrections.extend(iter_corrections)

            # Apply corrections to HTML
            current_html = self._apply_corrections(
                current_html, iter_corrections
            )

        return {
            'html': current_html,
            'corrections': corrections,
            'similarity_scores': similarity_scores,
        }

    def compare_only(self, original_image: np.ndarray,
                     html_content: str) -> dict:
        """
        Only compare (no correction). Returns similarity score and diff map.
        Useful for reporting quality metrics.
        """
        h, w = original_image.shape[:2]

        rendered = self._render_html(html_content, w, h)
        if rendered is None:
            return {
                'similarity': 0.0,
                'diff_map': None,
                'rendered': None,
                'error': 'Failed to render HTML',
            }

        similarity, diff_map, diff_regions = self._compare_images(
            original_image, rendered
        )

        return {
            'similarity': similarity,
            'diff_map': diff_map,
            'rendered': rendered,
            'diff_regions': diff_regions,
        }

    def _render_html(self, html_content: str, width: int,
                     height: int) -> np.ndarray:
        """Render HTML to a screenshot using Playwright."""
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            return None

        # Write HTML to temp file
        tmp_fd, tmp_path = tempfile.mkstemp(suffix='.html')
        try:
            with os.fdopen(tmp_fd, 'w', encoding='utf-8') as f:
                f.write(html_content)

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page(
                    viewport={'width': width, 'height': height}
                )
                page.goto(f'file:///{tmp_path.replace(os.sep, "/")}')
                # Wait for fonts and rendering
                page.wait_for_load_state('networkidle')
                page.wait_for_timeout(500)

                # Take screenshot
                screenshot_bytes = page.screenshot(
                    full_page=False,
                    type='png',
                )
                browser.close()

            # Decode screenshot
            nparr = np.frombuffer(screenshot_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return None

            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize to match original if needed
            if img.shape[:2] != (height, width):
                img = cv2.resize(img, (width, height),
                                 interpolation=cv2.INTER_AREA)

            return img

        except Exception:
            return None
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _compare_images(self, original: np.ndarray,
                        rendered: np.ndarray) -> tuple:
        """
        Compare two images and return similarity score, diff map,
        and regions of significant difference.
        """
        # Convert to grayscale
        gray_orig = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        gray_rend = cv2.cvtColor(rendered, cv2.COLOR_RGB2GRAY)

        # Compute absolute difference
        diff = cv2.absdiff(gray_orig, gray_rend)

        # Color difference
        color_diff = np.mean(
            np.abs(original.astype(np.float32) - rendered.astype(np.float32)),
            axis=2
        ).astype(np.uint8)

        # Combined diff map
        diff_map = np.maximum(diff, color_diff)

        # Similarity score: percentage of pixels within threshold
        threshold = 30  # Pixels within 30 gray levels = "similar"
        similar_pixels = np.sum(diff_map < threshold)
        total_pixels = diff_map.size
        similarity = similar_pixels / total_pixels

        # Find regions with significant differences
        _, binary_diff = cv2.threshold(diff_map, 40, 255, cv2.THRESH_BINARY)

        # Dilate to connect nearby diff pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        dilated = cv2.dilate(binary_diff, kernel, iterations=2)

        # Find contours of diff regions
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        diff_regions = []
        for cnt in contours:
            x, y, rw, rh = cv2.boundingRect(cnt)
            area = rw * rh
            if area > 500:  # Ignore tiny diffs
                diff_regions.append({
                    'x': x, 'y': y, 'width': rw, 'height': rh,
                    'area': area,
                    'severity': float(np.mean(diff_map[y:y+rh, x:x+rw])),
                })

        # Sort by severity (worst first)
        diff_regions.sort(key=lambda r: r['severity'], reverse=True)

        return similarity, diff_map, diff_regions

    def _analyze_differences(self, original, rendered, diff_regions,
                             page_elements) -> list:
        """
        Analyze diff regions and generate correction suggestions.
        """
        corrections = []

        for region in diff_regions[:10]:  # Process top 10 worst regions
            rx, ry = region['x'], region['y']
            rw, rh = region['width'], region['height']

            # Find which element(s) overlap this diff region
            matching_elements = self._find_elements_in_region(
                page_elements, rx, ry, rw, rh
            )

            if not matching_elements:
                continue

            for elem in matching_elements:
                # Compare colors in this region
                orig_crop = original[
                    max(0, ry):min(original.shape[0], ry + rh),
                    max(0, rx):min(original.shape[1], rx + rw)
                ]
                rend_crop = rendered[
                    max(0, ry):min(rendered.shape[0], ry + rh),
                    max(0, rx):min(rendered.shape[1], rx + rw)
                ]

                if orig_crop.size == 0 or rend_crop.size == 0:
                    continue

                correction = self._detect_correction(
                    elem, orig_crop, rend_crop, region
                )
                if correction:
                    corrections.append(correction)

        return corrections

    def _detect_correction(self, elem, orig_crop, rend_crop, region):
        """Detect what type of correction is needed for an element."""
        # Color correction: compare average colors
        orig_mean = np.mean(orig_crop.reshape(-1, 3), axis=0)
        rend_mean = np.mean(rend_crop.reshape(-1, 3), axis=0)
        color_diff = np.linalg.norm(orig_mean - rend_mean)

        if color_diff > 30:
            # Significant color difference
            correct_color = tuple(int(c) for c in orig_mean)
            return {
                'type': 'color',
                'element_id': elem.id,
                'property': 'background-color',
                'old_value': rend_mean.tolist(),
                'new_value': correct_color,
                'severity': float(color_diff),
                'region': region,
            }

        return None

    def _find_elements_in_region(self, elements, rx, ry, rw, rh):
        """Find elements whose bbox overlaps with the given region."""
        matches = []
        for elem in self._walk(elements):
            b = elem.bbox
            # Check overlap
            overlap_x = max(0, min(b.x + b.width, rx + rw) - max(b.x, rx))
            overlap_y = max(0, min(b.y + b.height, ry + rh) - max(b.y, ry))
            overlap_area = overlap_x * overlap_y
            elem_area = b.width * b.height

            if overlap_area > 0 and elem_area > 0:
                overlap_ratio = overlap_area / min(elem_area, rw * rh)
                if overlap_ratio > 0.3:
                    matches.append(elem)

        return matches

    def _apply_corrections(self, html_content, corrections):
        """Apply color corrections by modifying inline styles in HTML."""
        for correction in corrections:
            if correction['type'] == 'color':
                elem_id = correction['element_id']
                r, g, b = correction['new_value']
                new_color = f'#{r:02x}{g:02x}{b:02x}'

                # Find the element's CSS class and inject inline style
                css_class = f'el-{elem_id}'
                old_marker = f'class="{css_class}"'
                new_marker = (
                    f'class="{css_class}" '
                    f'style="background-color: {new_color}"'
                )
                html_content = html_content.replace(
                    old_marker, new_marker, 1
                )

        return html_content

    def _walk(self, elements):
        """Walk all elements recursively."""
        for elem in elements:
            yield elem
            yield from self._walk(elem.children)
