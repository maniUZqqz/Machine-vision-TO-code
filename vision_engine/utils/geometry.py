from typing import List
from ..models.elements import BoundingBox


def merge_overlapping_boxes(boxes: List[BoundingBox],
                            iou_threshold: float = 0.3) -> List[BoundingBox]:
    """Non-Maximum Suppression + containment dedup.

    Instead of greedy merging (which creates inflated union boxes),
    we suppress duplicates while keeping the best representative:

    1. Sort by area descending (larger = more likely to be correct).
    2. For each box, suppress smaller boxes with high IoU overlap.
    3. Remove boxes that are >80% contained within a larger kept box.
    """
    if not boxes:
        return []

    # Deduplicate near-identical boxes first (IoU > 0.85)
    boxes = _deduplicate_near_identical(boxes, threshold=0.85)

    # Sort by area descending: prefer larger, more confident detections
    boxes = sorted(boxes, key=lambda b: b.area, reverse=True)

    keep = []
    suppressed = set()

    for i, box_a in enumerate(boxes):
        if i in suppressed:
            continue
        keep.append(box_a)

        for j in range(i + 1, len(boxes)):
            if j in suppressed:
                continue
            box_b = boxes[j]

            # Suppress if high overlap (NMS)
            if box_a.overlap_ratio(box_b) > iou_threshold:
                suppressed.add(j)
                continue

            # Suppress if box_b is mostly contained within box_a.
            # BUT keep legitimate child elements: boxes that are large
            # enough to be meaningful UI components (area >= 2000px²
            # AND at least 10px in both dimensions AND area ratio < 0.5).
            # This preserves buttons inside panels while removing
            # noise fragments.
            containment = _containment_ratio(box_a, box_b)
            if containment > 0.80:
                size_ratio = box_b.area / box_a.area if box_a.area > 0 else 0
                is_meaningful_child = (
                    box_b.area >= 2000
                    and box_b.width >= 15
                    and box_b.height >= 10
                    and size_ratio < 0.5
                )
                if not is_meaningful_child:
                    suppressed.add(j)

    return keep


def _deduplicate_near_identical(boxes, threshold=0.85):
    """Remove near-identical boxes (IoU > threshold), keeping the first."""
    if len(boxes) <= 1:
        return boxes

    result = []
    used = set()
    for i, a in enumerate(boxes):
        if i in used:
            continue
        result.append(a)
        for j in range(i + 1, len(boxes)):
            if j in used:
                continue
            if a.overlap_ratio(boxes[j]) > threshold:
                used.add(j)
    return result


def _containment_ratio(outer: BoundingBox, inner: BoundingBox) -> float:
    """What fraction of inner's area is contained within outer?"""
    ix1 = max(outer.x, inner.x)
    iy1 = max(outer.y, inner.y)
    ix2 = min(outer.x2, inner.x2)
    iy2 = min(outer.y2, inner.y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    return intersection / inner.area if inner.area > 0 else 0.0


def find_smallest_container(bbox: BoundingBox,
                            candidates: List) -> object:
    """Find the smallest element whose bbox contains the given bbox."""
    best = None
    best_area = float('inf')
    for candidate in candidates:
        if (candidate.bbox.contains(bbox) and
                candidate.bbox.area < best_area and
                candidate.bbox.area > bbox.area):
            best = candidate
            best_area = candidate.bbox.area
    return best


def find_container_by_center(bbox: BoundingBox, candidates: List) -> object:
    """Find the smallest container that contains the CENTER of the given bbox."""
    cx, cy = bbox.center
    best = None
    best_area = float('inf')
    for candidate in candidates:
        b = candidate.bbox
        if (b.x <= cx <= b.x2 and b.y <= cy <= b.y2 and
                b.area < best_area and b.area > bbox.area):
            best = candidate
            best_area = b.area
    return best
