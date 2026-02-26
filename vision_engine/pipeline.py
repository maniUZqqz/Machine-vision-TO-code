from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Protocol, runtime_checkable
import numpy as np

from .models.elements import DetectedElement, TextElement


@runtime_checkable
class PipelineStep(Protocol):
    """Every step in the pipeline implements this interface."""
    def process(self, image: np.ndarray, context: PipelineContext) -> PipelineContext:
        ...


@dataclass
class PipelineContext:
    """Accumulated state passed through all pipeline steps."""
    original_image: np.ndarray
    working_image: np.ndarray = None
    width: int = 0
    height: int = 0
    original_width: int = 0   # Set by ImagePreprocessor (before any resize)
    original_height: int = 0  # Set by ImagePreprocessor (before any resize)
    regions: List[DetectedElement] = field(default_factory=list)
    text_elements: List[TextElement] = field(default_factory=list)
    color_map: dict = field(default_factory=dict)
    spacing_map: dict = field(default_factory=dict)
    layout_map: dict = field(default_factory=dict)
    page_background: tuple = (255, 255, 255)

    def __post_init__(self):
        if self.working_image is None:
            self.working_image = self.original_image.copy()
        if self.width == 0 and self.original_image is not None:
            self.height, self.width = self.original_image.shape[:2]


class VisionPipeline:
    """Runs a configurable sequence of processing steps."""

    def __init__(self):
        self._steps: List[PipelineStep] = []

    def add_step(self, step: PipelineStep) -> VisionPipeline:
        self._steps.append(step)
        return self

    def run(self, image: np.ndarray) -> PipelineContext:
        context = PipelineContext(original_image=image)
        for step in self._steps:
            context = step.process(image, context)
        return context


def create_default_pipeline() -> VisionPipeline:
    from .preprocessing.image_loader import ImagePreprocessor
    from .detection.contour_detector import ContourDetector
    from .detection.text_detector import TextDetector
    from .analysis.color_analyzer import ColorAnalyzer
    from .analysis.typography_analyzer import TypographyAnalyzer
    from .analysis.spacing_analyzer import SpacingAnalyzer
    from .analysis.layout_analyzer import LayoutAnalyzer
    from .analysis.hierarchy_builder import HierarchyBuilder
    from .classification.element_classifier import ElementClassifier

    pipeline = VisionPipeline()
    pipeline.add_step(ImagePreprocessor())       # Step 1: Preprocess image
    pipeline.add_step(ContourDetector())          # Step 2: Detect UI regions (classical CV)

    # Step 2.5: YOLO enhancement (if trained model available)
    try:
        from .detection.yolo_detector import YOLODetector
        yolo = YOLODetector()
        if yolo.available:
            pipeline.add_step(yolo)              # Merge YOLO + classical detections
    except Exception:
        pass  # YOLO not available, continue with classical only

    pipeline.add_step(TextDetector(languages=['fa', 'en']))  # Step 3: OCR
    pipeline.add_step(ColorAnalyzer())            # Step 4: Extract colors
    pipeline.add_step(TypographyAnalyzer())       # Step 5: Font analysis
    pipeline.add_step(SpacingAnalyzer())          # Step 6: Padding/margins
    pipeline.add_step(HierarchyBuilder())         # Step 7: Tree assembly + auto-grouping
    pipeline.add_step(LayoutAnalyzer())           # Step 8: Flex/grid detection (after grouping)
    pipeline.add_step(ElementClassifier())        # Step 9: Classify elements
    return pipeline
