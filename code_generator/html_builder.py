from vision_engine.models.elements import DetectedElement, ElementType, TextDirection


class HTMLBuilder:
    """
    Builds nested HTML using semantic tags based on element type
    and position context.
    """

    def build(self, page, page_dir: str) -> str:
        lines = []
        # If the original image has a dark top band (browser chrome artifact),
        # inject an absolutely-positioned div at y=0 to reproduce it.
        # Using position:absolute avoids shifting any child elements.
        top_border = getattr(page, 'top_border', None)
        if top_border:
            band_h, band_rgb = top_border
            band_hex = '#{:02x}{:02x}{:02x}'.format(*band_rgb)
            lines.append(
                f'        <div style="position:absolute;top:0;left:0;'
                f'width:{page.width}px;height:{band_h}px;'
                f'background-color:{band_hex};z-index:9999;"></div>'
            )
        for element in page.root_elements:
            rendered = self._render_recursive(element, indent=2)
            if rendered:
                lines.append(rendered)
        return '\n'.join(lines)

    def _render_recursive(self, element: DetectedElement, indent: int) -> str:
        pad = '    ' * indent
        css_class = f"el-{element.id}"

        # --- TEXT elements ---
        if element.element_type == ElementType.TEXT:
            # Skip text elements with no content — they add clutter without value
            text = getattr(element, 'text', '') or ''
            if not text.strip():
                return ''
            return self._render_text(element, pad, css_class)

        # --- SEPARATOR ---
        if element.element_type == ElementType.SEPARATOR:
            return f'{pad}<hr class="{css_class}">'

        # --- IMAGE / ICON ---
        if element.element_type == ElementType.IMAGE:
            # CDN icon match: render as <i> with icon class
            icon_name = getattr(element, 'icon_name', None)
            if icon_name and getattr(element, 'is_icon', False):
                return (
                    f'{pad}<i class="{css_class} bi {icon_name}" '
                    f'aria-hidden="true"></i>'
                )
            # Embedded image
            data_uri = getattr(element, 'image_data_uri', None)
            if data_uri:
                return (
                    f'{pad}<img class="{css_class}" '
                    f'src="{data_uri}" alt="">'
                )
            return (
                f'{pad}<div class="{css_class}" role="img" '
                f'aria-label="image placeholder"></div>'
            )

        # --- INPUT ---
        if element.element_type == ElementType.INPUT:
            placeholder = ''
            text_children = [
                c for c in element.children
                if c.element_type == ElementType.TEXT
            ]
            if text_children:
                placeholder = getattr(text_children[0], 'text', '')
            return (
                f'{pad}<input class="{css_class}" type="text" '
                f'placeholder="{self._escape(placeholder)}">'
            )

        # --- SELECT ---
        if element.element_type == ElementType.SELECT:
            option_text = ''
            text_children = [
                c for c in element.children
                if c.element_type == ElementType.TEXT
            ]
            if text_children:
                option_text = getattr(text_children[0], 'text', '')
            return (
                f'{pad}<select class="{css_class}">\n'
                f'{pad}    <option>{self._escape(option_text)}</option>\n'
                f'{pad}</select>'
            )

        # --- CHECKBOX ---
        if element.element_type == ElementType.CHECKBOX:
            return f'{pad}<input class="{css_class}" type="checkbox">'

        # --- RADIO ---
        if element.element_type == ElementType.RADIO:
            return f'{pad}<input class="{css_class}" type="radio">'

        # --- TEXTAREA ---
        if element.element_type == ElementType.TEXTAREA:
            text_content = ''
            text_children = [
                c for c in element.children
                if c.element_type == ElementType.TEXT
            ]
            if text_children:
                text_content = getattr(text_children[0], 'text', '')
            return (
                f'{pad}<textarea class="{css_class}" '
                f'placeholder="{self._escape(text_content)}"></textarea>'
            )

        # --- BUTTON ---
        if element.element_type == ElementType.BUTTON:
            return self._render_button(element, pad, css_class)

        # --- CONTAINER (div, header, footer, nav, aside, section) ---
        return self._render_container(element, pad, css_class, indent)

    # ------------------------------------------------------------------
    # Text rendering
    # ------------------------------------------------------------------

    def _render_text(self, element, pad, css_class):
        tag = self._choose_text_tag(element)
        dir_attr = ''
        if hasattr(element, 'direction'):
            if element.direction == TextDirection.RTL:
                dir_attr = ' dir="rtl"'
            elif element.direction == TextDirection.MIXED:
                dir_attr = ' dir="auto"'
        text = getattr(element, 'text', '')
        return f'{pad}<{tag} class="{css_class}"{dir_attr}>{self._escape(text)}</{tag}>'

    def _choose_text_tag(self, element) -> str:
        tag = getattr(element, 'semantic_tag', None)
        if tag and tag in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'a'):
            return tag

        if not hasattr(element, 'typography') or not element.typography:
            return 'p'

        font_size = element.typography.font_size_px
        if font_size >= 28:
            return 'h1'
        elif font_size >= 22:
            return 'h2'
        elif font_size >= 18:
            return 'h3'
        elif font_size >= 16:
            return 'h4'
        return 'p'

    # ------------------------------------------------------------------
    # Button rendering
    # ------------------------------------------------------------------

    def _render_button(self, element, pad, css_class):
        text_parts = []
        for child in element.children:
            if child.element_type == ElementType.TEXT:
                text_parts.append(getattr(child, 'text', ''))
        button_text = ' '.join(text_parts) if text_parts else ''
        return f'{pad}<button type="button" class="{css_class}">{self._escape(button_text)}</button>'

    # ------------------------------------------------------------------
    # Container rendering with semantic tags
    # ------------------------------------------------------------------

    def _render_container(self, element, pad, css_class, indent):
        tag = getattr(element, 'semantic_tag', 'div') or 'div'

        # List rendering: <ul> with <li> wrappers
        if tag == 'ul':
            return self._render_list(element, pad, css_class, indent)

        # Table rendering: <table> with <tr>/<td>
        if tag == 'table':
            return self._render_table(element, pad, css_class, indent)

        if not element.children:
            return f'{pad}<{tag} class="{css_class}"></{tag}>'

        lines = [f'{pad}<{tag} class="{css_class}">']
        for child in element.children:
            rendered = self._render_recursive(child, indent + 1)
            if rendered:
                lines.append(rendered)
        lines.append(f'{pad}</{tag}>')
        return '\n'.join(lines)

    # ------------------------------------------------------------------
    # List rendering (<ul><li>)
    # ------------------------------------------------------------------

    def _render_list(self, element, pad, css_class, indent):
        """Render a list element with <ul> and <li> wrappers."""
        inner_pad = '    ' * (indent + 1)
        lines = [f'{pad}<ul class="{css_class}">']

        for child in element.children:
            # Skip separators between list items
            if child.element_type == ElementType.SEPARATOR:
                lines.append(self._render_recursive(child, indent + 1))
                continue

            li_class = f"el-{child.id}-li"
            # If child is TEXT, render directly inside <li>
            if child.element_type == ElementType.TEXT:
                text = getattr(child, 'text', '')
                dir_attr = ''
                if hasattr(child, 'direction'):
                    from vision_engine.models.elements import TextDirection
                    if child.direction == TextDirection.RTL:
                        dir_attr = ' dir="rtl"'
                    elif child.direction == TextDirection.MIXED:
                        dir_attr = ' dir="auto"'
                lines.append(
                    f'{inner_pad}<li class="{li_class}"{dir_attr}>'
                    f'{self._escape(text)}</li>'
                )
            else:
                # Container child: wrap in <li> and render contents inside
                lines.append(f'{inner_pad}<li class="{li_class}">')
                lines.append(self._render_recursive(child, indent + 2))
                lines.append(f'{inner_pad}</li>')

        lines.append(f'{pad}</ul>')
        return '\n'.join(lines)

    # ------------------------------------------------------------------
    # Table rendering (<table><tr><td>)
    # ------------------------------------------------------------------

    def _render_table(self, element, pad, css_class, indent):
        """Render a table element using <table>, <tr>, <td>."""
        layout = getattr(element, 'layout', None)
        columns = layout.columns if layout and layout.columns else 2

        inner_pad = '    ' * (indent + 1)
        cell_pad = '    ' * (indent + 2)

        lines = [f'{pad}<table class="{css_class}">']

        children = list(element.children)

        # Group children into rows based on column count
        for row_idx in range(0, len(children), columns):
            row_children = children[row_idx:row_idx + columns]
            is_header = (row_idx == 0)
            tag_cell = 'th' if is_header else 'td'
            row_tag = 'thead' if is_header else 'tbody'

            if is_header:
                lines.append(f'{inner_pad}<thead>')
            elif row_idx == columns:
                lines.append(f'{inner_pad}<tbody>')

            lines.append(f'{inner_pad}    <tr>')
            for cell_child in row_children:
                cell_class = f"el-{cell_child.id}"
                if cell_child.element_type == ElementType.TEXT:
                    text = getattr(cell_child, 'text', '')
                    lines.append(
                        f'{cell_pad}    <{tag_cell} class="{cell_class}">'
                        f'{self._escape(text)}</{tag_cell}>'
                    )
                else:
                    lines.append(f'{cell_pad}    <{tag_cell} class="{cell_class}">')
                    lines.append(self._render_recursive(cell_child, indent + 4))
                    lines.append(f'{cell_pad}    </{tag_cell}>')
            lines.append(f'{inner_pad}    </tr>')

            if is_header:
                lines.append(f'{inner_pad}</thead>')

        # Close tbody if we had data rows
        if len(children) > columns:
            lines.append(f'{inner_pad}</tbody>')

        lines.append(f'{pad}</table>')
        return '\n'.join(lines)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _escape(text: str) -> str:
        return (
            text.replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
        )
