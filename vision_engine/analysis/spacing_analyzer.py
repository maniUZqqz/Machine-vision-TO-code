from ..models.elements import SpacingInfo
from ..models.layout import LayoutMode

# Common CSS spacing values for snapping
_CSS_SNAP_POINTS = [0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64]


class SpacingAnalyzer:
    """
    Step 6: Calculate padding (parent-to-child) and margins (child-to-sibling).
    Uses CSS-common snap points for clean output and avoids double-counting
    with flex gap.
    """

    def process(self, image, context) -> object:
        for element in self._walk(context.regions):
            if not element.children:
                continue

            children_bboxes = [c.bbox for c in element.children]

            min_child_x = min(c.x for c in children_bboxes)
            min_child_y = min(c.y for c in children_bboxes)
            max_child_x2 = max(c.x2 for c in children_bboxes)
            max_child_y2 = max(c.y2 for c in children_bboxes)

            element.spacing = SpacingInfo(
                padding_top=self._snap(min_child_y - element.bbox.y),
                padding_right=self._snap(element.bbox.x2 - max_child_x2),
                padding_bottom=self._snap(element.bbox.y2 - max_child_y2),
                padding_left=self._snap(min_child_x - element.bbox.x),
            )

            # Only compute sibling margins for BLOCK layout.
            # Flex/grid containers use gap (computed by LayoutAnalyzer).
            layout = getattr(element, 'layout', None)
            uses_gap = (
                layout and layout.mode != LayoutMode.BLOCK
            ) if layout else False

            if uses_gap:
                for child in element.children:
                    if not child.spacing:
                        child.spacing = SpacingInfo()
            else:
                sorted_children = sorted(
                    element.children, key=lambda c: (c.bbox.y, c.bbox.x)
                )
                for i, child in enumerate(sorted_children):
                    if i == 0:
                        child.spacing = child.spacing or SpacingInfo()
                        continue
                    prev = sorted_children[i - 1]
                    margin_top = self._snap(child.bbox.y - prev.bbox.y2)
                    margin_left = self._snap(child.bbox.x - prev.bbox.x2)
                    child.spacing = SpacingInfo(
                        margin_top=max(0, margin_top),
                        margin_left=max(0, margin_left),
                    )

        return context

    def _snap(self, value: int) -> int:
        """Snap to nearest common CSS spacing value."""
        value = max(0, value)
        best = 0
        best_dist = abs(value)
        for sp in _CSS_SNAP_POINTS:
            dist = abs(value - sp)
            if dist < best_dist:
                best = sp
                best_dist = dist
            elif sp > value:
                break
        return best

    def _walk(self, elements):
        for elem in elements:
            yield elem
            yield from self._walk(elem.children)
