"""
YOLO-based UI Element Detector

Uses a trained YOLOv8/YOLO11 model to detect UI elements in screenshots.
Falls back to the classical ContourDetector when no model is available.

This is integrated as an optional enhancement step that runs BEFORE or
alongside the ContourDetector to provide higher-confidence detections.
"""
import os
import logging
from typing import List, Optional

import numpy as np

from ..models.elements import BoundingBox, DetectedElement, ElementType

logger = logging.getLogger(__name__)

# YOLO class → ElementType mapping
YOLO_CLASS_MAP = {
    0: ('button', ElementType.BUTTON),
    1: ('input', ElementType.INPUT),
    2: ('select', ElementType.SELECT),
    3: ('card', ElementType.CONTAINER),
    4: ('image', ElementType.IMAGE),
    5: ('icon', ElementType.IMAGE),     # icon = small image
    6: ('text_block', ElementType.TEXT),
    7: ('header_bar', ElementType.CONTAINER),
    8: ('sidebar', ElementType.CONTAINER),
    9: ('navbar', ElementType.CONTAINER),
    10: ('separator', ElementType.SEPARATOR),
}

# Semantic tag overrides based on YOLO class
YOLO_SEMANTIC_TAGS = {
    0: 'button',
    1: 'input',
    2: 'select',
    3: 'div',        # card → section or div
    4: 'img',
    5: 'img',        # icon
    6: 'p',          # text block
    7: 'header',
    8: 'aside',
    9: 'nav',
    10: 'hr',
}

# Default model path (relative to this file)
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'ml_models', 'yolo_ui',
    'best_ui_detector.pt'
)


class YOLODetector:
    """
    UI Element detector using YOLO.

    If a trained model exists, uses it for detection.
    Otherwise, returns empty results (pipeline falls back to ContourDetector).
    """

    def __init__(self, model_path: str = None, confidence: float = 0.3,
                 iou_threshold: float = 0.5):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self._model = None
        self._available = False

        self._try_load_model()

    def _try_load_model(self):
        """Try to load the YOLO model."""
        if not os.path.exists(self.model_path):
            logger.info(f"YOLO model not found at {self.model_path}. "
                        "Using classical detection only.")
            return

        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            self._available = True
            logger.info(f"YOLO UI detector loaded: {self.model_path}")
        except Exception as e:
            logger.warning(f"Failed to load YOLO model: {e}")

    @property
    def available(self) -> bool:
        return self._available

    def detect(self, image: np.ndarray) -> List[DetectedElement]:
        """
        Run YOLO detection on an image.

        Returns list of DetectedElement with bbox, element_type, confidence,
        and semantic_tag set from YOLO predictions.
        """
        if not self._available:
            return []

        try:
            results = self._model.predict(
                image,
                conf=self.confidence,
                iou=self.iou_threshold,
                verbose=False,
            )
        except Exception as e:
            logger.error(f"YOLO prediction failed: {e}")
            return []

        elements = []
        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Get element type
                if cls_id not in YOLO_CLASS_MAP:
                    continue
                class_name, elem_type = YOLO_CLASS_MAP[cls_id]

                # Create element
                bbox = BoundingBox(
                    int(x1), int(y1),
                    int(x2 - x1), int(y2 - y1)
                )

                elem = DetectedElement(
                    id=f"yolo_{cls_id}_{len(elements)}",
                    element_type=elem_type,
                    bbox=bbox,
                    confidence=conf,
                )

                # Set semantic tag
                elem.semantic_tag = YOLO_SEMANTIC_TAGS.get(cls_id, 'div')

                # Mark icons
                if cls_id == 5:
                    elem.is_icon = True

                elements.append(elem)

        logger.info(f"YOLO detected {len(elements)} elements")
        return elements

    def process(self, image: np.ndarray, context) -> object:
        """
        Pipeline step: enhance detection with YOLO.

        If YOLO model is available, merges YOLO detections with
        classical contour detections. YOLO results take priority
        for overlapping regions.
        """
        if not self._available:
            return context

        yolo_elements = self.detect(context.working_image)
        if not yolo_elements:
            return context

        # Merge YOLO detections with existing regions
        # YOLO detections with high confidence override classical ones
        merged = self._merge_detections(context.regions, yolo_elements)
        context.regions = merged

        return context

    def _merge_detections(self, classical: List[DetectedElement],
                          yolo: List[DetectedElement]) -> List[DetectedElement]:
        """
        Merge YOLO detections with classical contour detections.

        Strategy:
        1. Keep all YOLO detections with conf > 0.5
        2. Keep classical detections that don't overlap heavily with YOLO
        3. For overlapping pairs, prefer the one with higher confidence
        """
        if not yolo:
            return classical

        result = []
        used_classical = set()

        # First pass: add high-confidence YOLO detections
        for ye in yolo:
            if ye.confidence < 0.4:
                continue

            # Check if any classical detection heavily overlaps
            best_overlap = 0
            best_classical_idx = -1
            for i, ce in enumerate(classical):
                if i in used_classical:
                    continue
                iou = ye.bbox.overlap_ratio(ce.bbox)
                if iou > best_overlap:
                    best_overlap = iou
                    best_classical_idx = i

            if best_overlap > 0.5 and best_classical_idx >= 0:
                # Use YOLO detection but preserve classical children
                ce = classical[best_classical_idx]
                ye.children = ce.children
                ye.color = ce.color
                used_classical.add(best_classical_idx)

            result.append(ye)

        # Second pass: add non-overlapping classical detections
        for i, ce in enumerate(classical):
            if i in used_classical:
                continue

            # Check if this classical detection overlaps with any YOLO result
            max_overlap = 0
            for ye in result:
                iou = ce.bbox.overlap_ratio(ye.bbox)
                max_overlap = max(max_overlap, iou)

            if max_overlap < 0.3:
                result.append(ce)

        return result
