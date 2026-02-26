import uuid
import numpy as np

from ..models.elements import BoundingBox, DetectedElement, ElementType, TextDirection
from ..models.layout import LayoutInfo, LayoutMode


class HierarchyBuilder:
    """
    Step 8: Final assembly of the element tree.
    Cleans up the hierarchy and prepares for code generation.
    """

    def process(self, image, context) -> object:
        img_h, img_w = image.shape[:2]
        self.tolerance = max(8, int(min(img_w, img_h) * 0.015))

        # Remove empty containers with no children and no text
        context.regions = self._prune_empty(context.regions)

        # Wrap fragmented sidebars/edge panels into unified containers.
        context.regions = self._wrap_edge_groups(context.regions, img_w, img_h)

        # Nest YOLO root elements inside their containing classical panels.
        # YOLO detects buttons/icons at root level; they should be children of
        # the JPEG-background panels they visually reside in.  Without nesting,
        # root YOLO elements get higher z-index than panels in CSS stacking,
        # obscuring the JPEG content with wrong colors.
        true_page_area = img_w * img_h
        context.regions = self._nest_orphans_into_containers(
            context.regions, true_page_area=true_page_area
        )

        # Auto-group children into rows/columns for containers with many flat children.
        # This creates intermediate wrapper elements so layout analyzer can detect flex/grid.
        context.regions = self._auto_group_children(context.regions, img_w, img_h)

        # Determine page-level text direction
        rtl_count = 0
        total_count = 0
        for text_elem in context.text_elements:
            total_count += 1
            if text_elem.direction == TextDirection.RTL:
                rtl_count += 1

        if total_count > 0 and rtl_count / total_count > 0.5:
            context.page_direction = "rtl"
        else:
            context.page_direction = "ltr"

        return context

    def _nest_orphans_into_containers(self, elements, true_page_area=None):
        """Nest YOLO-detected root-level elements inside their smallest containing
        classical-detected parent.

        YOLO elements (ID starts with "yolo_") are created at root level with
        no parent_id. If a YOLO element's bbox is fully contained within a
        classical container, it should become a child of that container so that
        its position is computed relative to its visual parent (not the page).

        Only YOLO elements are moved — classical elements stay at their
        originally assigned level. This avoids disrupting well-built
        classical hierarchies.
        """
        if not elements:
            return elements

        # Use the true image area to decide which elements are "page-level"
        # containers (and should NOT be used as parents of small elements).
        # Fallback: use the max element area, but this can wrongly exclude
        # the main panel when it IS the largest root-level element.
        page_area = true_page_area or max(1, max(e.bbox.area for e in elements))

        # Candidate containers: classical (non-YOLO) elements AND YOLO elements
        # that are of type CONTAINER (card, sidebar, header_bar, navbar).
        # This handles the common case where the main dark panel was absorbed by
        # a YOLO card detection in _merge_detections (IoU>0.5) and now has a
        # "yolo_3_X" ID — it must still be able to parent smaller YOLO elements.
        classical_containers = [
            e for e in elements
            if e.bbox.area < page_area * 0.98
            and (
                not e.id.startswith("yolo_")           # all classical elements
                or e.element_type == ElementType.CONTAINER  # YOLO containers too
            )
        ]
        # Sort by area ascending so we find the SMALLEST valid container
        classical_containers.sort(key=lambda e: e.bbox.area)

        assigned_ids = set()

        for elem in elements:
            # Only process YOLO elements without an existing parent
            if not elem.id.startswith("yolo_"):
                continue
            if elem.parent_id is not None:
                continue

            best_parent = None
            best_area = float('inf')
            for container in classical_containers:
                if not container.bbox.contains(elem.bbox):
                    continue
                if container.bbox.area < elem.bbox.area * 2:
                    continue  # container too small to be a real parent
                if container.bbox.area < best_area:
                    best_parent = container
                    best_area = container.bbox.area

            if best_parent is not None:
                elem.parent_id = best_parent.id
                best_parent.children.append(elem)
                assigned_ids.add(id(elem))

        # Remove newly-assigned YOLO elements from root level
        result = [e for e in elements if id(e) not in assigned_ids]
        return result

    def _auto_group_children(self, elements, img_w, img_h):
        """Recursively auto-group flat children into row wrappers."""
        for elem in elements:
            if elem.children:
                elem.children = self._auto_group_children(elem.children, img_w, img_h)
                if len(elem.children) >= 4:
                    elem.children = self._group_into_rows(elem.children, elem)
        return elements

    def _group_into_rows(self, children, parent):
        """Group children that share the same y-band into row containers."""
        if len(children) < 4:
            return children

        sorted_by_y = sorted(children, key=lambda c: c.bbox.y)
        rows = []
        current_row = [sorted_by_y[0]]

        for child in sorted_by_y[1:]:
            # Check y-range overlap with current row
            row_y_min = min(c.bbox.y for c in current_row)
            row_y_max = max(c.bbox.y2 for c in current_row)
            child_overlap = min(row_y_max, child.bbox.y2) - max(row_y_min, child.bbox.y)
            child_h = max(child.bbox.height, 1)

            if child_overlap / child_h > 0.3:
                current_row.append(child)
            else:
                rows.append(current_row)
                current_row = [child]
        rows.append(current_row)

        # Only create wrapper if we actually have multiple rows
        if len(rows) <= 1:
            return children

        # Check: do we have rows with 2+ items? If not, no benefit
        multi_item_rows = sum(1 for r in rows if len(r) >= 2)
        if multi_item_rows == 0:
            return children

        result = []
        for row in rows:
            if len(row) == 1:
                result.append(row[0])
                continue

            # Sort row items by x position
            row = sorted(row, key=lambda c: c.bbox.x)

            # Create a wrapper container for this row
            min_x = min(c.bbox.x for c in row)
            min_y = min(c.bbox.y for c in row)
            max_x2 = max(c.bbox.x2 for c in row)
            max_y2 = max(c.bbox.y2 for c in row)

            wrapper = DetectedElement(
                id=str(uuid.uuid4())[:8],
                element_type=ElementType.CONTAINER,
                bbox=BoundingBox(min_x, min_y, max_x2 - min_x, max_y2 - min_y),
            )
            # Copy parent's color if available
            if parent.color:
                wrapper.color = parent.color

            for child in row:
                child.parent_id = wrapper.id
                wrapper.children.append(child)

            result.append(wrapper)

        return result

    def _wrap_edge_groups(self, elements, img_width, img_height):
        """
        Detect fragmented sidebar/edge-panel patterns and merge them into one container.

        A REAL sidebar has ALL of these properties:
        1. ≥5 root elements in the rightmost/leftmost 12% of the page
        2. Their x-start values cluster tightly (within 60px of each other)
        3. The cluster itself is narrow: natural width < 14% of image width
        4. The cluster physically touches the page edge (max_x2 within 30px of img edge)
        5. They span ≥50% of page height together
        6. Most are truly dark (luminance < 120), similar colors

        This conservative set of criteria prevents content areas, industrial panels,
        photo sections, or any non-sidebar elements from being incorrectly merged.
        """
        edge_frac = 0.12   # Only look at the outermost 12%
        right_thresh = img_width * (1 - edge_frac)
        left_thresh = img_width * edge_frac

        right_candidates = [
            e for e in elements
            if e.bbox.x >= right_thresh and e.color and e.color.background
        ]
        left_candidates = [
            e for e in elements
            if e.bbox.x2 <= left_thresh and e.color and e.color.background
        ]

        result = []
        grouped_elems = set()

        for candidates, is_right in [(right_candidates, True), (left_candidates, False)]:
            cluster = self._dominant_x_cluster(candidates, band=60)

            # Need at least 5 elements — real sidebars have many menu items
            if not cluster or len(cluster) < 5:
                continue

            min_y = min(e.bbox.y for e in cluster)
            max_y2 = max(e.bbox.y2 for e in cluster)
            min_x = min(e.bbox.x for e in cluster)
            natural_max_x2 = max(e.bbox.x2 for e in cluster)

            # Must span ≥50% of page height
            if (max_y2 - min_y) < img_height * 0.50:
                continue

            # Cluster natural width must be narrow (< 14% of image width)
            # This rejects wide content areas that merely happen to be at the edge
            cluster_natural_width = natural_max_x2 - min_x
            if cluster_natural_width > img_width * 0.14:
                continue

            # Cluster must physically touch the page edge (not just be "near" it)
            # Real sidebars extend to the viewport boundary
            if is_right and (img_width - natural_max_x2) > 30:
                continue
            if not is_right and min_x > 30:
                continue

            # Most elements must be TRULY dark (luminance < 120, not just medium)
            dark = [
                e for e in cluster
                if self._color_luminance(e.color.background) < 120
            ]
            if len(dark) < max(3, len(cluster) * 2 // 3):
                continue

            # Dark elements must share similar colors (all parts of same sidebar panel)
            if not self._colors_are_similar(dark, threshold=70):
                continue

            # ---- All checks passed: build the unified sidebar container ----
            max_x2 = natural_max_x2

            # Snap to image edge
            if img_width - max_x2 < 30:
                max_x2 = img_width
            if min_x < 30:
                min_x = 0

            # Extend to full page height if spans >60%
            if (max_y2 - min_y) > img_height * 0.60:
                min_y = 0
                max_y2 = img_height

            container_bbox = BoundingBox(
                min_x, min_y, max_x2 - min_x, max_y2 - min_y
            )
            container = DetectedElement(
                id=str(uuid.uuid4())[:8],
                element_type=ElementType.CONTAINER,
                bbox=container_bbox,
            )
            largest = max(cluster, key=lambda e: e.bbox.area)
            container.color = largest.color

            for elem in cluster:
                elem.parent_id = container.id
                container.children.append(elem)
                grouped_elems.add(id(elem))

            result.append(container)

        # Add all elements not absorbed into a sidebar
        for e in elements:
            if id(e) not in grouped_elems:
                result.append(e)

        return result

    def _dominant_x_cluster(self, elements, band=60):
        """
        Find the largest group of elements whose bbox.x values cluster within `band` px.
        Returns that cluster, or None if no valid cluster found.
        """
        if not elements:
            return None

        best_cluster = []
        for anchor in elements:
            ax = anchor.bbox.x
            cluster = [e for e in elements if abs(e.bbox.x - ax) <= band]
            if len(cluster) > len(best_cluster):
                best_cluster = cluster

        return best_cluster if len(best_cluster) >= 2 else None

    def _color_luminance(self, color):
        return 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]

    def _colors_are_similar(self, elements, threshold=70):
        """Return True if all elements have similar background colors."""
        colors = [e.color.background for e in elements]
        avg = tuple(
            int(sum(c[i] for c in colors) / len(colors)) for i in range(3)
        )
        for color in colors:
            dist = sum((a - b) ** 2 for a, b in zip(color, avg)) ** 0.5
            if dist > threshold:
                return False
        return True

    def _prune_empty(self, elements):
        """Remove leaf containers that have no children and no visual presence."""
        result = []
        for elem in elements:
            elem.children = self._prune_empty(elem.children)
            # Always keep: has children or is text
            if elem.children or elem.element_type.value == "text":
                result.append(elem)
            elif elem.bbox.area > 1000:
                result.append(elem)
            elif self._has_visible_color(elem):
                # Keep small elements that have a distinct visible color
                result.append(elem)
        return result

    def _has_visible_color(self, elem):
        """Check if element has a non-white/non-black background."""
        if not elem.color:
            return False
        bg = elem.color.background
        luminance = 0.2126 * bg[0] + 0.7152 * bg[1] + 0.0722 * bg[2]
        if 10 < luminance < 240:
            return True
        if elem.color.border:
            return True
        return False
