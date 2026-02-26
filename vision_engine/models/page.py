from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from .elements import DetectedElement


@dataclass
class PageStructure:
    width: int
    height: int
    background_color: Tuple[int, int, int]
    root_elements: List[DetectedElement] = field(default_factory=list)
    text_direction: str = "ltr"
    title: Optional[str] = None
    # Dark top-band border detected from original image top rows.
    # Tuple of (height_px, (R, G, B)) or None.
    top_border: Optional[Tuple[int, Tuple[int, int, int]]] = None

    def walk(self):
        """Depth-first traversal of all elements."""
        stack = list(reversed(self.root_elements))
        while stack:
            element = stack.pop()
            yield element
            stack.extend(reversed(element.children))
