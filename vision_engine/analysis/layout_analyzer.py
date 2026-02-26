import numpy as np
from typing import List
from collections import Counter

from ..models.elements import DetectedElement, ElementType
from ..models.layout import LayoutInfo, LayoutMode


class LayoutAnalyzer:
    """
    Step 7: Determine CSS layout mode for each container.
    Analyzes child positions to detect flex-row, flex-column, grid, or block.
    Also detects align-items, justify-content variants, and flex-wrap.
    """

    def __init__(self, alignment_tolerance: int = 15):
        self.tolerance = alignment_tolerance

    def process(self, image, context) -> object:
        # Adaptive tolerance: ~1.5% of image's smaller dimension, min 8px
        img_min_dim = min(context.width, context.height) if hasattr(context, 'width') else 1000
        self.tolerance = max(8, int(img_min_dim * 0.015))

        for element in self._walk(context.regions):
            # Use ALL children for layout detection, not just containers.
            # A row of 3 cards (each a container) OR a row of 3 text labels
            # should both trigger flex-row detection.
            all_children = element.children

            if len(all_children) < 2:
                element.layout = LayoutInfo(mode=LayoutMode.BLOCK)
                continue

            element.layout = self._determine_layout(
                all_children, element
            )

        return context

    def _determine_layout(self, children: List[DetectedElement],
                          parent: DetectedElement) -> LayoutInfo:
        by_y = sorted(children, key=lambda c: c.bbox.y)
        by_x = sorted(children, key=lambda c: c.bbox.x)

        # Test: horizontal row?
        if self._is_horizontal_row(children):
            gap = self._compute_horizontal_gap(by_x)
            justify = self._detect_justify_enhanced(by_x, parent)
            align = self._detect_align_items_vertical(children, parent)
            return LayoutInfo(
                mode=LayoutMode.FLEX_ROW,
                gap=gap,
                justify=justify,
                align=align,
            )

        # Test: vertical column?
        if self._is_vertical_column(children):
            gap = self._compute_vertical_gap(by_y)
            align = self._detect_align_items_horizontal(children, parent)
            return LayoutInfo(
                mode=LayoutMode.FLEX_COLUMN,
                gap=gap,
                align=align,
            )

        # Test: wrapped flex row (multiple rows of horizontally-aligned items)?
        wrap_info = self._detect_flex_wrap(children, parent)
        if wrap_info:
            return wrap_info

        # Test: grid pattern?
        grid_info = self._detect_grid(children)
        if grid_info:
            return grid_info

        # Test: majority horizontal row (≥70% of children align horizontally,
        # with some outliers like separators at different y positions)
        majority_row = self._detect_majority_row(children, parent)
        if majority_row:
            return majority_row

        # Test: majority vertical column
        majority_col = self._detect_majority_column(children, parent)
        if majority_col:
            return majority_col

        return LayoutInfo(mode=LayoutMode.BLOCK)

    def _detect_majority_row(self, children, parent):
        """Detect flex-row when ≥70% of children form a horizontal row."""
        if len(children) < 3:
            return None

        # Too many scattered children → never flex_row via majority
        # (Dashboards with 10+ scattered widgets should stay BLOCK)
        if len(children) > 10:
            return None

        # Find the largest group of children with overlapping y-ranges
        best_group = []

        for anchor in children:
            group = []
            for c in children:
                overlap = min(anchor.bbox.y2, c.bbox.y2) - max(anchor.bbox.y, c.bbox.y)
                min_h = min(anchor.bbox.height, c.bbox.height)
                if min_h > 0 and overlap / min_h > 0.3:
                    group.append(c)
            if len(group) > len(best_group):
                best_group = group

        if len(best_group) >= max(2, len(children) * 0.7):
            # Verify the group items don't overlap horizontally too much
            by_x = sorted(best_group, key=lambda c: c.bbox.x)
            h_overlaps = 0
            for i in range(len(by_x) - 1):
                if by_x[i + 1].bbox.x < by_x[i].bbox.x2 - self.tolerance:
                    h_overlaps += 1
            if h_overlaps > len(by_x) * 0.3:
                return None

            gap = self._compute_horizontal_gap(by_x)
            justify = self._detect_justify_enhanced(by_x, parent)
            align = self._detect_align_items_vertical(best_group, parent)
            return LayoutInfo(
                mode=LayoutMode.FLEX_ROW,
                gap=gap,
                justify=justify,
                align=align,
            )
        return None

    def _detect_majority_column(self, children, parent):
        """Detect flex-column when ≥70% of children form a vertical column."""
        if len(children) < 3:
            return None

        # Too many scattered children → stay BLOCK
        if len(children) > 10:
            return None

        # Find the largest group of children with overlapping x-ranges
        best_group = []

        for anchor in children:
            group = []
            for c in children:
                overlap = min(anchor.bbox.x2, c.bbox.x2) - max(anchor.bbox.x, c.bbox.x)
                min_w = min(anchor.bbox.width, c.bbox.width)
                if min_w > 0 and overlap / min_w > 0.3:
                    group.append(c)
            if len(group) > len(best_group):
                best_group = group

        if len(best_group) >= max(2, len(children) * 0.7):
            # Verify the group items don't overlap vertically too much
            by_y = sorted(best_group, key=lambda c: c.bbox.y)
            v_overlaps = 0
            for i in range(len(by_y) - 1):
                if by_y[i + 1].bbox.y < by_y[i].bbox.y2 - self.tolerance:
                    v_overlaps += 1
            if v_overlaps > len(by_y) * 0.3:
                return None

            gap = self._compute_vertical_gap(by_y)
            align = self._detect_align_items_horizontal(best_group, parent)
            return LayoutInfo(
                mode=LayoutMode.FLEX_COLUMN,
                gap=gap,
                align=align,
            )
        return None

    def _is_horizontal_row(self, children: List[DetectedElement]) -> bool:
        """Check if children form a horizontal row.
        Uses y-range overlap instead of just center-y comparison,
        so items with different heights still count as a row."""
        if len(children) < 2:
            return False

        # Too many children for a simple row — likely scattered dashboard
        if len(children) > 8:
            return False

        # Primary check: do 70%+ of children overlap in y-range?
        y_ranges = [(c.bbox.y, c.bbox.y2) for c in children]
        overlap_count = 0
        total_pairs = 0
        for i in range(len(y_ranges)):
            for j in range(i + 1, len(y_ranges)):
                total_pairs += 1
                overlap = min(y_ranges[i][1], y_ranges[j][1]) - max(y_ranges[i][0], y_ranges[j][0])
                min_h = min(y_ranges[i][1] - y_ranges[i][0], y_ranges[j][1] - y_ranges[j][0])
                if min_h > 0 and overlap / min_h > 0.3:
                    overlap_count += 1

        if total_pairs > 0 and overlap_count / total_pairs >= 0.7:
            # Also verify they don't overlap horizontally (not stacked)
            sorted_by_x = sorted(children, key=lambda c: c.bbox.x)
            overlapping_x = 0
            for i in range(len(sorted_by_x) - 1):
                if sorted_by_x[i + 1].bbox.x < sorted_by_x[i].bbox.x2 - self.tolerance:
                    overlapping_x += 1
            if overlapping_x < len(children) * 0.3:
                return True

        # Fallback: original center-y check (works for equal-height items)
        if len(children) <= 6:
            y_centers = [c.bbox.center[1] for c in children]
            return (max(y_centers) - min(y_centers)) < self.tolerance

        return False

    def _is_vertical_column(self, children: List[DetectedElement]) -> bool:
        """Check if children form a vertical column.
        Uses x-range overlap for items with different widths."""
        if len(children) < 2:
            return False

        # Too many children for a simple column
        if len(children) > 8:
            return False

        # Primary check: do 70%+ of children overlap in x-range?
        x_ranges = [(c.bbox.x, c.bbox.x2) for c in children]
        overlap_count = 0
        total_pairs = 0
        for i in range(len(x_ranges)):
            for j in range(i + 1, len(x_ranges)):
                total_pairs += 1
                overlap = min(x_ranges[i][1], x_ranges[j][1]) - max(x_ranges[i][0], x_ranges[j][0])
                min_w = min(x_ranges[i][1] - x_ranges[i][0], x_ranges[j][1] - x_ranges[j][0])
                if min_w > 0 and overlap / min_w > 0.3:
                    overlap_count += 1

        if total_pairs > 0 and overlap_count / total_pairs >= 0.7:
            # Verify they don't overlap vertically (not side-by-side)
            sorted_by_y = sorted(children, key=lambda c: c.bbox.y)
            overlapping_y = 0
            for i in range(len(sorted_by_y) - 1):
                if sorted_by_y[i + 1].bbox.y < sorted_by_y[i].bbox.y2 - self.tolerance:
                    overlapping_y += 1
            if overlapping_y < len(children) * 0.3:
                return True

        # Fallback: original center-x check
        if len(children) <= 6:
            x_centers = [c.bbox.center[0] for c in children]
            return (max(x_centers) - min(x_centers)) < self.tolerance

        return False

    def _detect_flex_wrap(self, children, parent):
        """Detect flex-wrap: wrap — items in multiple rows but each row
        is a horizontal flex row."""
        sorted_by_y = sorted(children, key=lambda c: c.bbox.y)
        rows = []
        current_row = [sorted_by_y[0]]

        for child in sorted_by_y[1:]:
            # Use y-range overlap for row grouping
            row_y_min = min(c.bbox.y for c in current_row)
            row_y_max = max(c.bbox.y2 for c in current_row)
            child_overlap = min(row_y_max, child.bbox.y2) - max(row_y_min, child.bbox.y)
            child_h = child.bbox.height
            if child_h > 0 and child_overlap / child_h > 0.3:
                current_row.append(child)
            else:
                rows.append(current_row)
                current_row = [child]
        rows.append(current_row)

        if len(rows) < 2:
            return None

        # Each row must have at least 2 items
        multi_item_rows = sum(1 for row in rows if len(row) >= 2)
        if multi_item_rows < len(rows) * 0.5:
            return None

        # Each row should have items aligned horizontally
        for row in rows:
            if len(row) < 2:
                continue
            y_centers = [c.bbox.center[1] for c in row]
            if (max(y_centers) - min(y_centers)) >= self.tolerance:
                return None

        # Check if items have similar widths (wrapping pattern)
        widths = [c.bbox.width for c in children]
        width_std = np.std(widths) if len(widths) > 1 else 0
        avg_width = np.mean(widths) if widths else 0

        # Reasonably uniform widths suggest flex-wrap
        if avg_width > 0 and width_std / avg_width < 0.35:
            gap_h = self._compute_horizontal_gap(rows[0]) if len(rows[0]) >= 2 else 0
            first_per_row = [row[0] for row in rows]
            gap_v = self._compute_vertical_gap(first_per_row)
            return LayoutInfo(
                mode=LayoutMode.FLEX_ROW,
                gap=max(gap_h, gap_v),
                justify=self._detect_justify_enhanced(
                    sorted(rows[0], key=lambda c: c.bbox.x), parent
                ),
                wrap=True,
            )

        return None

    def _detect_grid(self, children: List[DetectedElement]):
        sorted_by_y = sorted(children, key=lambda c: c.bbox.y)
        rows = []
        current_row = [sorted_by_y[0]]

        for child in sorted_by_y[1:]:
            # Use y-range overlap for row grouping (not just center-y)
            row_y_min = min(c.bbox.y for c in current_row)
            row_y_max = max(c.bbox.y2 for c in current_row)
            child_overlap = min(row_y_max, child.bbox.y2) - max(row_y_min, child.bbox.y)
            child_h = child.bbox.height
            if child_h > 0 and child_overlap / child_h > 0.3:
                current_row.append(child)
            else:
                rows.append(sorted(current_row, key=lambda c: c.bbox.x))
                current_row = [child]
        rows.append(sorted(current_row, key=lambda c: c.bbox.x))

        if len(rows) >= 2:
            col_counts = [len(row) for row in rows]
            most_common = Counter(col_counts).most_common(1)[0]
            columns = most_common[0]
            consistency = most_common[1] / len(rows)

            # Grid requires: at least 2 columns, 70%+ consistent rows,
            # and not too many single-item rows (which are scattered)
            single_rows = sum(1 for c in col_counts if c == 1)
            if (columns >= 2
                    and consistency >= 0.7
                    and single_rows < len(rows) * 0.4):

                # Verify column x-positions are consistent across rows
                consistent_rows = [r for r in rows if len(r) == columns]
                if len(consistent_rows) >= 2:
                    # Check that column x-positions align across rows
                    col_x_positions = list(zip(*[
                        [c.bbox.x for c in row] for row in consistent_rows
                    ]))
                    x_aligned = True
                    for col_xs in col_x_positions:
                        if max(col_xs) - min(col_xs) > self.tolerance * 3:
                            x_aligned = False
                            break
                    if not x_aligned:
                        return None

                gap_h = self._compute_horizontal_gap(rows[0]) if rows[0] else 0
                first_per_row = [row[0] for row in rows]
                gap_v = self._compute_vertical_gap(first_per_row)
                return LayoutInfo(
                    mode=LayoutMode.GRID,
                    columns=columns,
                    rows=len(rows),
                    gap=max(gap_h, gap_v),
                )
        return None

    def _compute_horizontal_gap(self, sorted_children) -> int:
        if len(sorted_children) < 2:
            return 0
        sorted_children = sorted(sorted_children, key=lambda c: c.bbox.x)
        gaps = []
        for i in range(len(sorted_children) - 1):
            gap = sorted_children[i + 1].bbox.x - sorted_children[i].bbox.x2
            gaps.append(max(0, gap))
        return int(np.median(gaps)) if gaps else 0

    def _compute_vertical_gap(self, sorted_children) -> int:
        if len(sorted_children) < 2:
            return 0
        sorted_children = sorted(sorted_children, key=lambda c: c.bbox.y)
        gaps = []
        for i in range(len(sorted_children) - 1):
            gap = sorted_children[i + 1].bbox.y - sorted_children[i].bbox.y2
            gaps.append(max(0, gap))
        return int(np.median(gaps)) if gaps else 0

    def _detect_justify_enhanced(self, sorted_by_x, parent) -> str:
        """Detect justify-content including space-between and space-around."""
        if len(sorted_by_x) < 2:
            return "flex-start"

        first_gap = sorted_by_x[0].bbox.x - parent.bbox.x
        last_gap = parent.bbox.x2 - sorted_by_x[-1].bbox.x2

        # Compute inner gaps
        inner_gaps = []
        for i in range(len(sorted_by_x) - 1):
            g = sorted_by_x[i + 1].bbox.x - sorted_by_x[i].bbox.x2
            inner_gaps.append(max(0, g))

        avg_inner = np.mean(inner_gaps) if inner_gaps else 0

        # space-between: edge gaps ≈ 0, inner gaps equal
        if first_gap < self.tolerance and last_gap < self.tolerance and avg_inner > self.tolerance:
            return "space-between"

        # space-around: edge gaps ≈ half of inner gaps
        if avg_inner > 0 and inner_gaps:
            half_inner = avg_inner / 2
            if (abs(first_gap - half_inner) < self.tolerance
                    and abs(last_gap - half_inner) < self.tolerance):
                return "space-around"

        # space-evenly: all gaps equal (edges = inner)
        if avg_inner > 0 and inner_gaps:
            if (abs(first_gap - avg_inner) < self.tolerance
                    and abs(last_gap - avg_inner) < self.tolerance):
                return "space-evenly"

        # center: equal edge gaps, both significant
        if abs(first_gap - last_gap) < self.tolerance and first_gap > self.tolerance:
            return "center"

        # flex-end: items pushed to the right
        if first_gap > last_gap + self.tolerance:
            return "flex-end"

        return "flex-start"

    def _detect_align_items_vertical(self, children, parent):
        """Detect vertical alignment of children in a horizontal flex row."""
        tops = [c.bbox.y - parent.bbox.y for c in children]
        bottoms = [parent.bbox.y2 - c.bbox.y2 for c in children]
        heights = [c.bbox.height for c in children]
        parent_h = parent.bbox.height

        # stretch: all children fill parent height
        if all(abs(h - parent_h) < self.tolerance * 2 for h in heights):
            return "stretch"

        # flex-start: all tops near 0
        if max(tops) < self.tolerance:
            return "flex-start"

        # flex-end: all bottoms near 0
        if max(bottoms) < self.tolerance:
            return "flex-end"

        # center: all children vertically centered
        centers = [(t + b) for t, b in zip(tops, bottoms)]
        center_diffs = [abs(t - b) for t, b in zip(tops, bottoms)]
        if all(d < self.tolerance * 2 for d in center_diffs):
            return "center"

        return "stretch"

    def _detect_align_items_horizontal(self, children, parent):
        """Detect horizontal alignment of children in a vertical flex column."""
        lefts = [c.bbox.x - parent.bbox.x for c in children]
        rights = [parent.bbox.x2 - c.bbox.x2 for c in children]
        widths = [c.bbox.width for c in children]
        parent_w = parent.bbox.width

        # stretch: all children fill parent width
        if all(abs(w - parent_w) < self.tolerance * 2 for w in widths):
            return "stretch"

        # flex-start: all lefts near 0
        if max(lefts) < self.tolerance:
            return "flex-start"

        # flex-end: all rights near 0
        if max(rights) < self.tolerance:
            return "flex-end"

        # center
        center_diffs = [abs(l - r) for l, r in zip(lefts, rights)]
        if all(d < self.tolerance * 2 for d in center_diffs):
            return "center"

        return "stretch"

    def _walk(self, elements):
        for elem in elements:
            yield elem
            yield from self._walk(elem.children)
