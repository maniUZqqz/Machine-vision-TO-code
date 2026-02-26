from collections import Counter
from vision_engine.models.page import PageStructure
from vision_engine.models.elements import ElementType
from .html_builder import HTMLBuilder
from .css_builder import CSSBuilder
from .rtl_handler import RTLHandler


class CodeGenerator:
    """
    Converts a PageStructure into HTML + CSS.
    Uses position-based layout for pixel-accurate recreation.
    """

    def __init__(self):
        self.html_builder = HTMLBuilder()
        self.css_builder = CSSBuilder()
        self.rtl_handler = RTLHandler()

    def generate(self, page: PageStructure) -> dict:
        page_dir = self.rtl_handler.determine_page_direction(page)

        css_rules = self.css_builder.build(page)

        # Extract CSS variables for repeated colors
        color_vars, css_rules = self._extract_color_variables(css_rules)

        css_string = self.css_builder.to_string(css_rules)

        html_body = self.html_builder.build(page, page_dir)

        # Check if any icons use CDN
        cdn_links = self._collect_cdn_links(page)

        combined = self._assemble_document(
            html_body, css_string, color_vars, page_dir,
            page.width, page.height, cdn_links
        )

        return {
            'html': combined,
            'css': css_string,
            'combined': combined,
        }

    def _extract_color_variables(self, css_rules):
        """Find colors used 2+ times and replace with CSS variables."""
        # Count color occurrences
        color_counts = Counter()
        color_props = ('color', 'background-color', 'border-color')

        for selector, props in css_rules:
            for prop, value in props.items():
                if prop in color_props and value.startswith('#'):
                    color_counts[value] += 1
                # Also extract colors from border shorthand
                if prop in ('border', 'border-top', 'border-left'):
                    parts = value.split()
                    for part in parts:
                        if part.startswith('#'):
                            color_counts[part] += 1

        # Create variables for colors used 2+ times
        color_vars = {}
        var_index = 1
        for color, count in color_counts.most_common():
            if count < 2:
                break
            var_name = f'--color-{var_index}'
            color_vars[color] = var_name
            var_index += 1

        if not color_vars:
            return {}, css_rules

        # Replace color values with var() references
        new_rules = []
        for selector, props in css_rules:
            new_props = {}
            for prop, value in props.items():
                if prop in color_props and value in color_vars:
                    new_props[prop] = f'var({color_vars[value]})'
                elif prop in ('border', 'border-top', 'border-left'):
                    for color, var_name in color_vars.items():
                        if color in value:
                            value = value.replace(color, f'var({var_name})')
                    new_props[prop] = value
                else:
                    new_props[prop] = value
            new_rules.append((selector, new_props))

        return color_vars, new_rules

    def _collect_cdn_links(self, page):
        """Collect CDN stylesheet links needed for icon fonts."""
        cdn_urls = set()
        for elem in self._walk_elements(page.root_elements):
            if (elem.element_type == ElementType.IMAGE
                    and getattr(elem, 'icon_name', None)
                    and getattr(elem, 'icon_library', None)):
                if elem.icon_library == 'bootstrap-icons':
                    cdn_urls.add(
                        "https://cdn.jsdelivr.net/npm/"
                        "bootstrap-icons@1.11.3/font/bootstrap-icons.min.css"
                    )
        return list(cdn_urls)

    def _walk_elements(self, elements):
        """Recursively walk all elements."""
        for elem in elements:
            yield elem
            yield from self._walk_elements(elem.children)

    def _assemble_document(self, body: str, css: str, color_vars: dict,
                           direction: str, width: int, height: int,
                           cdn_links=None) -> str:
        lang = 'fa' if direction == 'rtl' else 'en'

        # Build :root CSS variables block
        root_vars = ''
        if color_vars:
            var_lines = []
            for color, var_name in color_vars.items():
                var_lines.append(f'            {var_name}: {color};')
            root_vars = '        :root {\n' + '\n'.join(var_lines) + '\n        }\n'

        # Build CDN link tags
        cdn_tags = ''
        # Add Vazirmatn font for RTL/Persian pages
        if direction == 'rtl':
            cdn_tags += (
                '\n    <link rel="preconnect" href="https://fonts.googleapis.com">'
                '\n    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
                '\n    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
                'family=Vazirmatn:wght@100..900&display=swap">'
            )
        if cdn_links:
            for url in cdn_links:
                cdn_tags += f'\n    <link rel="stylesheet" href="{url}">'

        return f"""<!DOCTYPE html>
<html lang="{lang}" dir="{direction}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Generated Page</title>{cdn_tags}
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        html {{ overflow-x: hidden; overflow-y: auto; margin: 0; padding: 0; }}
{root_vars}        body {{
            width: {width}px;
            max-width: 100vw;
            min-height: {height}px;
            transform-origin: top left;
            position: relative;
            overflow: visible;
        }}
{css}
        /* Responsive: scale down on smaller viewports */
        @media (max-width: {width}px) {{
            body {{
                transform: scale(calc(100vw / {width}));
            }}
        }}
    </style>
</head>
<body>
{body}
    <script>
        (function() {{
            var w = {width};
            function rescale() {{
                var vw = Math.min(window.innerWidth, screen.width);
                var scale = vw / w;
                if (scale > 1) scale = 1;
                document.body.style.transform = 'scale(' + scale + ')';
                // Adjust wrapper height so page doesn't overflow
                document.documentElement.style.height = ({height} * scale) + 'px';
            }}
            rescale();
            window.addEventListener('resize', rescale);
        }})();
    </script>
</body>
</html>"""
