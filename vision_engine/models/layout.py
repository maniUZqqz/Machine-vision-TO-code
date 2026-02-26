from dataclasses import dataclass
from enum import Enum
from typing import Optional


class LayoutMode(Enum):
    BLOCK = "block"
    FLEX_ROW = "flex_row"
    FLEX_COLUMN = "flex_col"
    GRID = "grid"


@dataclass
class LayoutInfo:
    mode: LayoutMode
    gap: int = 0
    columns: Optional[int] = None
    rows: Optional[int] = None
    justify: str = "flex-start"
    align: str = "stretch"
    wrap: bool = False
