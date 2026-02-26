from vision_engine.models.page import PageStructure
from vision_engine.models.elements import TextDirection


class RTLHandler:
    """Determines text directionality at page and element levels."""

    def determine_page_direction(self, page: PageStructure) -> str:
        rtl_count = 0
        total_count = 0

        for element in page.walk():
            if hasattr(element, 'direction'):
                total_count += 1
                if element.direction == TextDirection.RTL:
                    rtl_count += 1

        if total_count == 0:
            return 'ltr'
        return 'rtl' if rtl_count / total_count > 0.5 else 'ltr'
