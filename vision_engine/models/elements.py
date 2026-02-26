from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum


class ElementType(Enum):
    CONTAINER = "container"
    TEXT = "text"
    IMAGE = "image"
    BUTTON = "button"
    INPUT = "input"
    SEPARATOR = "separator"
    SELECT = "select"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    TEXTAREA = "textarea"


class TextDirection(Enum):
    LTR = "ltr"
    RTL = "rtl"
    MIXED = "mixed"


@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def area(self) -> int:
        return self.width * self.height

    def contains(self, other: 'BoundingBox') -> bool:
        return (self.x <= other.x and self.y <= other.y and
                self.x2 >= other.x2 and self.y2 >= other.y2)

    def overlap_ratio(self, other: 'BoundingBox') -> float:
        ix1 = max(self.x, other.x)
        iy1 = max(self.y, other.y)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        intersection = (ix2 - ix1) * (iy2 - iy1)
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0


@dataclass
class ColorInfo:
    background: Tuple[int, int, int]
    foreground: Optional[Tuple[int, int, int]] = None
    border: Optional[Tuple[int, int, int]] = None
    gradient: Optional[str] = None  # CSS gradient e.g. "linear-gradient(to right, #1a1a1a, #3a3a3a)"

    def bg_hex(self) -> str:
        return '#{:02x}{:02x}{:02x}'.format(*self.background)

    def fg_hex(self) -> str:
        if self.foreground:
            return '#{:02x}{:02x}{:02x}'.format(*self.foreground)
        return '#000000'

    def border_hex(self) -> str:
        if self.border:
            return '#{:02x}{:02x}{:02x}'.format(*self.border)
        return None


@dataclass
class TypographyInfo:
    font_size_px: int
    font_weight: str = "normal"
    text_align: str = "left"
    line_height: Optional[float] = None


@dataclass
class SpacingInfo:
    margin_top: int = 0
    margin_right: int = 0
    margin_bottom: int = 0
    margin_left: int = 0
    padding_top: int = 0
    padding_right: int = 0
    padding_bottom: int = 0
    padding_left: int = 0


@dataclass
class DetectedElement:
    id: str
    element_type: ElementType
    bbox: BoundingBox
    color: Optional[ColorInfo] = None
    spacing: Optional[SpacingInfo] = None
    layout: object = None
    children: List['DetectedElement'] = field(default_factory=list)
    parent_id: Optional[str] = None
    confidence: float = 1.0
    semantic_tag: str = "div"
    border_radius: Optional[int] = None
    image_data_uri: Optional[str] = None
    bg_image_data_uri: Optional[str] = None  # CSS background-image for complex containers
    box_shadow: Optional[str] = None
    is_icon: bool = False
    icon_name: Optional[str] = None
    icon_library: Optional[str] = None


@dataclass
class TextElement(DetectedElement):
    text: str = ""
    direction: TextDirection = TextDirection.LTR
    typography: Optional[TypographyInfo] = None
    ocr_confidence: float = 0.0
