from vision_engine.models.elements import DetectedElement, ElementType
from vision_engine.models.layout import LayoutMode


class CSSBuilder:
    """
    Generates CSS using flex/grid when detected, falling back to
    absolute positioning for BLOCK layout mode.
    """

    def build(self, page) -> list:
        rules = []

        bg = '#{:02x}{:02x}{:02x}'.format(*page.background_color)
        # Compute page background luminance for dark-theme detection
        r, g, b = page.background_color
        self._page_bg_lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        self._page_bg_dark = self._page_bg_lum < 80

        # Choose font-family based on text direction
        direction = getattr(page, 'text_direction', 'ltr')
        if direction == 'rtl':
            font_family = "'Vazirmatn', 'IRANSans', 'Segoe UI', Tahoma, sans-serif"
        else:
            font_family = "'Segoe UI', 'Inter', 'Roboto', -apple-system, sans-serif"

        rules.append(('body', {
            'background-color': bg,
            'font-family': font_family,
        }))

        for idx, element in enumerate(page.root_elements):
            self._build_recursive(
                element, parent_bbox=None, parent_layout=None,
                depth=0, sibling_index=idx, rules=rules,
            )

        # Add hover states for buttons
        self._add_button_hover_rules(page.root_elements, rules)

        # Add global table/list rules if any exist
        self._add_table_list_global_rules(page.root_elements, rules)

        # Clip page-spanning headers so they don't overlap sidebar elements
        self._clip_headers_at_sidebars(page, rules)

        return rules

    def _add_button_hover_rules(self, elements, rules):
        """Add :hover pseudo-class rules for buttons."""
        for elem in elements:
            if elem.element_type == ElementType.BUTTON and elem.color and elem.color.background:
                bg = elem.color.background
                # Darken by 15% for hover
                hover_bg = tuple(max(0, int(c * 0.85)) for c in bg)
                hover_hex = '#{:02x}{:02x}{:02x}'.format(*hover_bg)
                selector = f'.el-{elem.id}:hover'
                rules.append((selector, {
                    'background-color': hover_hex,
                    'transition': 'background-color 0.2s ease',
                }))
            # Also add transition to the button itself (non-hover)
            if elem.element_type == ElementType.BUTTON:
                for i, (sel, props) in enumerate(rules):
                    if sel == f'.el-{elem.id}':
                        props['transition'] = 'background-color 0.2s ease'
                        break
            self._add_button_hover_rules(elem.children, rules)

    def _add_table_list_global_rules(self, elements, rules):
        """Add global CSS rules for tables and lists if they exist in the tree."""
        has_table = False
        has_list = False
        for elem in self._walk_all(elements):
            tag = getattr(elem, 'semantic_tag', 'div')
            if tag == 'table':
                has_table = True
            elif tag == 'ul':
                has_list = True
            if has_table and has_list:
                break

        if has_table:
            # Get the table element to extract colors
            table_elem = next(
                (e for e in self._walk_all(elements)
                 if getattr(e, 'semantic_tag', '') == 'table'), None
            )
            border_color = '#e0e0e0'
            if table_elem and table_elem.color and table_elem.color.border:
                border_color = table_elem.color.border_hex() or border_color

            rules.append(('th, td', {
                'border': f'1px solid {border_color}',
                'padding': '8px 12px',
                'text-align': 'left',
            }))
            rules.append(('th', {
                'font-weight': 'bold',
                'background-color': '#f5f5f5',
            }))

        if has_list:
            rules.append(('li', {
                'padding': '8px 12px',
                'border-bottom': '1px solid #e8e8e8',
            }))
            rules.append(('li:last-child', {
                'border-bottom': 'none',
            }))

    def _walk_all(self, elements):
        """Walk all elements in the tree."""
        for elem in elements:
            yield elem
            yield from self._walk_all(elem.children)

    def _build_recursive(self, element, parent_bbox, parent_layout,
                         depth, sibling_index, rules):
        # Skip CSS for text elements with no content — they are not rendered in HTML
        if element.element_type.value == 'text':
            text = getattr(element, 'text', '') or ''
            if not text.strip():
                return

        selector = f'.el-{element.id}'
        props = {}
        bbox = element.bbox

        # ----------------------------------------------------------
        # Positioning: always absolute based on detected pixel bbox.
        # Flex/grid layout algorithms ignore detected bbox coordinates,
        # causing rendered positions to drift from the original screenshot.
        # Using position:absolute with detected left/top/width/height gives
        # pixel-accurate placement for all element types.
        # ----------------------------------------------------------
        props['position'] = 'absolute'
        if parent_bbox:
            props['left'] = f'{bbox.x - parent_bbox.x}px'
            props['top'] = f'{bbox.y - parent_bbox.y}px'
        else:
            props['left'] = f'{bbox.x}px'
            props['top'] = f'{bbox.y}px'
        props['width'] = f'{bbox.width}px'
        props['height'] = f'{bbox.height}px'

        # Z-index for absolute-positioned elements:
        # Deeper elements get higher z-index (children above parents).
        # Among siblings, smaller elements get higher z-index so they
        # appear in front of larger background containers.
        if props.get('position') == 'absolute':
            # Base: depth ensures children are above parents
            # +1 so root elements start at z-index 1, not 0
            z = (depth + 1) * 100 + sibling_index
            # Elements with a JPEG background image contain their own rich
            # visual content. They must appear ABOVE any flat-color siblings
            # that overlap their bbox (e.g. a detected footer div covering
            # the bottom portion of a JPEG panel).  Apply a +50 offset so
            # bg-image containers win over flat siblings at the same depth.
            if getattr(element, 'bg_image_data_uri', None):
                z += 50
            props['z-index'] = str(z)

        # ----------------------------------------------------------
        # Layout: if this element IS a flex/grid container
        # ----------------------------------------------------------
        own_layout = getattr(element, 'layout', None)
        if own_layout and own_layout.mode != LayoutMode.BLOCK:
            self._apply_layout_props(own_layout, props)

        # ----------------------------------------------------------
        # Colors
        # ----------------------------------------------------------
        self._apply_colors(element, props)

        # ----------------------------------------------------------
        # Borders
        # ----------------------------------------------------------
        # Skip border on elements whose background is a data-URI image:
        # (a) the border is already captured in the JPEG, and
        # (b) a CSS border shifts absolutely-positioned children by 1 px,
        #     causing systematic pixel-level misalignment with the original.
        has_bg_image = bool(getattr(element, 'bg_image_data_uri', None))
        # Skip border on wide elements that touch the top page edge.
        # A full-width header at y=0 with a border-radius draws a 1px strip
        # at y=0 that doesn't exist in the original screenshot (the original
        # shows dark content at y=0-2, not a light border line).
        # Only suppress when: element spans ≥50% page width AND starts at y≤2.
        # This is a narrow condition that targets page-top headers specifically
        # without affecting small edge-touching elements in other images.
        _is_wide_top_edge = (
            parent_bbox is None  # root-level only
            and bbox.y <= 2
            and bbox.width >= 400  # at least moderately wide
        )
        if not has_bg_image and not _is_wide_top_edge and element.color and element.color.border:
            border_color = element.color.border_hex()
            if border_color:
                props['border'] = f'1px solid {border_color}'

        # ----------------------------------------------------------
        # Box shadow
        # ----------------------------------------------------------
        box_shadow = getattr(element, 'box_shadow', None)
        if box_shadow:
            props['box-shadow'] = box_shadow

        # ----------------------------------------------------------
        # Element-type-specific styles
        # ----------------------------------------------------------
        if element.element_type == ElementType.TEXT:
            self._apply_text_styles(element, bbox, props)
        elif element.element_type == ElementType.BUTTON:
            self._apply_button_styles(element, bbox, props)
        elif element.element_type == ElementType.INPUT:
            self._apply_input_styles(element, bbox, props)
        elif element.element_type == ElementType.SELECT:
            self._apply_select_styles(element, bbox, props)
        elif element.element_type == ElementType.TEXTAREA:
            self._apply_textarea_styles(element, bbox, props)
        elif element.element_type in (ElementType.CHECKBOX, ElementType.RADIO):
            self._apply_check_radio_styles(element, bbox, props)
        elif element.element_type == ElementType.SEPARATOR:
            self._apply_separator_styles(element, bbox, props)
        elif element.element_type == ElementType.IMAGE:
            self._apply_image_styles(element, bbox, props)
        else:
            # Check semantic tag for list/table styling
            semantic = getattr(element, 'semantic_tag', 'div')
            if semantic == 'ul':
                self._apply_list_styles(element, bbox, props)
            elif semantic == 'table':
                self._apply_table_styles(element, bbox, props)
            else:
                self._apply_container_styles(element, bbox, depth, props)

        rules.append((selector, props))

        # Recurse into children
        for idx, child in enumerate(element.children):
            self._build_recursive(
                child, parent_bbox=bbox, parent_layout=own_layout,
                depth=depth + 1, sibling_index=idx, rules=rules,
            )

    # ==================================================================
    # Layout properties (flex/grid on the container itself)
    # ==================================================================

    def _apply_layout_props(self, layout, props):
        """Set overflow:visible for containers whose children are absolutely positioned.
        Since all children now use position:absolute (out of flex/grid flow),
        we only need to ensure the container doesn't clip them."""
        # All layout modes: children are absolute, so overflow must be visible.
        # We still annotate the layout intent via data attributes (not emitted here).
        props['overflow'] = 'visible'

    # ==================================================================
    # Flex/grid child sizing
    # ==================================================================

    def _apply_flex_child_sizing(self, element, parent_layout, props):
        """Size a child that lives inside a flex/grid parent."""
        bbox = element.bbox

        if parent_layout.mode == LayoutMode.GRID:
            # Grid items auto-size; only set height if not stretching
            props['min-height'] = f'{bbox.height}px'
        elif parent_layout.mode in (LayoutMode.FLEX_ROW, LayoutMode.FLEX_COLUMN):
            # Flex items: use width/height but no absolute position
            if parent_layout.mode == LayoutMode.FLEX_ROW:
                props['width'] = f'{bbox.width}px'
                # Only set height if not stretching
                if parent_layout.align != 'stretch':
                    props['height'] = f'{bbox.height}px'
            else:
                # Flex column: set height, width may stretch
                props['height'] = f'{bbox.height}px'
                if parent_layout.align != 'stretch':
                    props['width'] = f'{bbox.width}px'

    # ==================================================================
    # Color handling
    # ==================================================================

    def _apply_colors(self, element, props):
        if element.element_type == ElementType.TEXT:
            if element.color and element.color.foreground:
                props['color'] = element.color.fg_hex()
                # If the text background is saturated (colored label/badge),
                # render it with its own background-color + padding.
                # Require both saturation AND luminance so dark gray
                # with a slight color cast doesn't trigger.
                if element.color.background:
                    bg = element.color.background
                    sat = self._rgb_saturation(bg)
                    lum = 0.2126 * bg[0] + 0.7152 * bg[1] + 0.0722 * bg[2]
                    if sat > 80 and lum > 50:
                        props['background-color'] = element.color.bg_hex()
                        props['padding'] = '2px 8px'
                        props['border-radius'] = '4px'
            else:
                props['color'] = '#222222'
        else:
            # For containers with a complex multi-toned background (e.g. SCADA
            # panels), use background-image so the texture is pixel-accurate.
            # The data URI was generated by ElementClassifier.
            bg_img_uri = getattr(element, 'bg_image_data_uri', None)
            if bg_img_uri:
                props['background-image'] = 'url("%s")' % bg_img_uri
                props['background-size'] = '100% 100%'
                props['background-repeat'] = 'no-repeat'
                # Do NOT set 'color' on image containers: the JPEG already
                # captures all visual content (including text).  Inheriting
                # a detected foreground color on child elements creates
                # spurious text rendering on top of the background image.
            elif element.color:
                gradient = getattr(element.color, 'gradient', None)
                if gradient and not self._bg_is_blend_artifact(element):
                    # Use CSS gradient instead of flat background-color
                    props['background'] = gradient
                elif not self._bg_is_blend_artifact(element):
                    # On dark-themed pages, skip UNSATURATED light background
                    # colors that are likely misdetections: the element sits on
                    # the dark page bg and the k-means color detector picked up
                    # a nearby light gray area by mistake.  Before the CSS-var
                    # fix these resolved to transparent → body bg (dark) →
                    # correct; now they show wrong light gray → regression.
                    # Skip only when ALL of:
                    #   • page bg is dark (lum < 80)
                    #   • element bg is significantly lighter (lum > page_lum+80)
                    #   • element bg is UNSATURATED (sat ≤ 25) — saturated
                    #     colors (red buttons, blue panels) are intentional
                    #   • element is a non-button container with children
                    _skip_light_bg = False
                    if (getattr(self, '_page_bg_dark', False)
                            and element.color.background
                            and element.element_type not in (
                                ElementType.BUTTON, ElementType.INPUT,
                                ElementType.SELECT, ElementType.TEXTAREA,
                            )):
                        _r, _g, _b = element.color.background
                        _elem_lum = 0.2126 * _r + 0.7152 * _g + 0.0722 * _b
                        _elem_sat = self._rgb_saturation((_r, _g, _b))
                        if (_elem_lum > getattr(self, '_page_bg_lum', 0) + 80
                                and _elem_sat <= 35
                                and element.children):
                            _skip_light_bg = True
                    if not _skip_light_bg:
                        props['background-color'] = element.color.bg_hex()
                if element.color.foreground:
                    props['color'] = element.color.fg_hex()

    @staticmethod
    def _rgb_saturation(rgb):
        """HSV saturation (0-255) for an RGB tuple."""
        r, g, b = rgb
        mx = max(r, g, b)
        mn = min(r, g, b)
        if mx == 0:
            return 0
        return int(255 * (mx - mn) / mx)

    def _bg_is_blend_artifact(self, element) -> bool:
        if not element.color or not element.color.background:
            return False
        parent_bg = element.color.background
        # A saturated background is almost certainly a real color (not an
        # average/blend of the children).  Skip the blend-artifact check so
        # colored headers, badges, etc. always render their background.
        if self._rgb_saturation(parent_bg) > 40:
            return False
        container_children = [
            c for c in element.children
            if c.color and c.color.background
            and c.element_type != ElementType.TEXT
        ]
        if len(container_children) < 2:
            return False
        for child in container_children:
            child_bg = child.color.background
            dist = sum((a - b) ** 2 for a, b in zip(parent_bg, child_bg)) ** 0.5
            if dist < 50:
                return False
        return True

    # ==================================================================
    # Element-type-specific styles
    # ==================================================================

    def _apply_text_styles(self, element, bbox, props):
        if hasattr(element, 'typography') and element.typography:
            typo = element.typography
            props['font-size'] = f'{typo.font_size_px}px'
            if typo.font_weight != 'normal':
                props['font-weight'] = typo.font_weight
            if typo.line_height:
                props['line-height'] = f'{typo.line_height}'
            else:
                props['line-height'] = '1.4'
            if typo.text_align == 'center':
                props['text-align'] = 'center'
            elif typo.text_align == 'right':
                props['text-align'] = 'right'
        # Detect multi-line text: if text has line breaks or height > 2x font-size
        is_multiline = False
        if hasattr(element, 'text') and element.text:
            if '\n' in element.text or len(element.text) > 50:
                is_multiline = True
        if hasattr(element, 'typography') and element.typography:
            fs = element.typography.font_size_px or 14
            if bbox.height > fs * 2.5:
                is_multiline = True

        if is_multiline:
            props['overflow-wrap'] = 'break-word'
        else:
            props['white-space'] = 'nowrap'
            props['overflow'] = 'hidden'
            props['text-overflow'] = 'ellipsis'

    def _apply_button_styles(self, element, bbox, props):
        props['cursor'] = 'pointer'
        props['border'] = props.get('border', 'none')
        # Center content
        props['display'] = props.get('display', 'flex')
        props['align-items'] = 'center'
        props['justify-content'] = 'center'
        # Border radius: use detected value or fallback
        radius = getattr(element, 'border_radius', None)
        if radius is not None:
            props['border-radius'] = f'{radius}px'
        elif bbox.height <= 40:
            props['border-radius'] = '4px'
        else:
            props['border-radius'] = '6px'

    def _apply_input_styles(self, element, bbox, props):
        props['border'] = props.get('border', '1px solid #cccccc')
        props['border-radius'] = '4px'
        props['padding'] = '0 8px'
        if not props.get('background-color'):
            props['background-color'] = '#ffffff'

    def _apply_select_styles(self, element, bbox, props):
        props['border'] = props.get('border', '1px solid #cccccc')
        props['border-radius'] = '4px'
        props['padding'] = '0 8px'
        props['appearance'] = 'auto'
        props['cursor'] = 'pointer'
        if not props.get('background-color'):
            props['background-color'] = '#ffffff'

    def _apply_textarea_styles(self, element, bbox, props):
        props['border'] = props.get('border', '1px solid #cccccc')
        props['border-radius'] = '4px'
        props['padding'] = '8px'
        props['resize'] = 'vertical'
        if not props.get('background-color'):
            props['background-color'] = '#ffffff'

    def _apply_check_radio_styles(self, element, bbox, props):
        props['cursor'] = 'pointer'
        props['accent-color'] = props.get('color', 'auto')

    def _apply_separator_styles(self, element, bbox, props):
        # Remove width/height for <hr>, set border style
        props['border'] = 'none'
        if bbox.height <= bbox.width:
            # Horizontal separator
            props['border-top'] = '1px solid'
            if element.color and element.color.background:
                props['border-top'] = f'1px solid {element.color.bg_hex()}'
            props['width'] = f'{bbox.width}px'
            if 'height' in props:
                del props['height']
        else:
            # Vertical separator
            props['border-left'] = '1px solid'
            if element.color and element.color.background:
                props['border-left'] = f'1px solid {element.color.bg_hex()}'
            props['height'] = f'{bbox.height}px'
            if 'width' in props:
                del props['width']

    def _apply_image_styles(self, element, bbox, props):
        # CDN icon: style as inline icon with font-size
        icon_name = getattr(element, 'icon_name', None)
        if icon_name and getattr(element, 'is_icon', False):
            # Icon font rendering
            icon_size = max(bbox.width, bbox.height)
            props['font-size'] = f'{icon_size}px'
            props['display'] = 'inline-flex'
            props['align-items'] = 'center'
            props['justify-content'] = 'center'
            if element.color and element.color.foreground:
                props['color'] = element.color.fg_hex()
            elif element.color and element.color.background:
                # Use bg color as icon color if no fg
                bg = element.color.background
                lum = 0.2126 * bg[0] + 0.7152 * bg[1] + 0.0722 * bg[2]
                if lum < 128:
                    props['color'] = element.color.bg_hex()
            # Remove background for icons
            if 'background-color' in props:
                del props['background-color']
            if 'background' in props:
                del props['background']
            return

        data_uri = getattr(element, 'image_data_uri', None)
        if data_uri:
            # Real image: remove bg color/gradient, set object-fit
            if 'background-color' in props:
                del props['background-color']
            if 'background' in props:  # remove gradient too — image itself is the visual
                del props['background']
            props['object-fit'] = 'cover'
            props['display'] = 'block'
        else:
            props['background-color'] = props.get('background-color', '#e0e0e0')
        radius = getattr(element, 'border_radius', None)
        props['border-radius'] = f'{radius}px' if radius else '4px'
        props['overflow'] = 'hidden'

    def _apply_container_styles(self, element, bbox, depth, props):
        # All containers whose children are absolutely positioned need
        # overflow:visible so children beyond the bbox boundary are visible.
        # Since all children now use position:absolute, every container with
        # children needs overflow:visible.
        if element.children:
            props['overflow'] = 'visible'
        else:
            props['overflow'] = 'hidden'

        # Border radius: use detected value or fallback heuristic.
        # Skip on bg-image elements: border-radius clips the JPEG content.
        if not getattr(element, 'bg_image_data_uri', None):
            radius = getattr(element, 'border_radius', None)
            if radius is not None:
                if radius > 0:
                    props['border-radius'] = f'{radius}px'
            else:
                aspect = bbox.width / max(bbox.height, 1)
                if bbox.area < 30000 and 0.25 < aspect < 8:
                    props['border-radius'] = '6px'
                elif bbox.area < 200000 and 0.25 < aspect < 15:
                    props['border-radius'] = '8px'

        # Padding for containers with children.
        # Skip padding on bg-image elements: padding shifts the containing
        # block for absolutely-positioned children, causing pixel misalignment.
        # The JPEG already captures the visual layout including internal spacing.
        if (element.spacing and element.children
                and not getattr(element, 'bg_image_data_uri', None)):
            sp = element.spacing
            if any([sp.padding_top, sp.padding_right,
                    sp.padding_bottom, sp.padding_left]):
                props['padding'] = (
                    f'{sp.padding_top}px {sp.padding_right}px '
                    f'{sp.padding_bottom}px {sp.padding_left}px'
                )

    # ==================================================================
    # List styles
    # ==================================================================

    def _apply_list_styles(self, element, bbox, props):
        """Style for <ul> elements."""
        props['list-style'] = 'none'
        props['padding'] = '0'
        props['margin'] = '0'

        # Border radius from detection or heuristic
        radius = getattr(element, 'border_radius', None)
        if radius and radius > 0:
            props['border-radius'] = f'{radius}px'

        # Overflow hidden for clean edges
        props['overflow'] = 'hidden'

    # ==================================================================
    # Table styles
    # ==================================================================

    def _apply_table_styles(self, element, bbox, props):
        """Style for <table> elements."""
        props['border-collapse'] = 'collapse'
        props['width'] = f'{bbox.width}px'
        # Don't set height on tables, let content flow

        # Remove absolute height if set (tables should auto-size height)
        if 'height' in props:
            del props['height']

        # Border radius from detection
        radius = getattr(element, 'border_radius', None)
        if radius and radius > 0:
            props['border-radius'] = f'{radius}px'
            props['overflow'] = 'hidden'

        # Add table-specific border if element has border color
        if element.color and element.color.border:
            border_color = element.color.border_hex()
            if border_color:
                props['border'] = f'1px solid {border_color}'

    # ==================================================================

    def _clip_headers_at_sidebars(self, page, rules):
        """Clip page-spanning header/footer elements so they don't overlap sidebars.

        A sidebar detected at x=X means horizontal elements (header/footer) that
        span the full page width should stop at x=X.  Without clipping, a 1920px
        header renders over the sidebar's top region with the wrong background color.

        Only clips when:
        - A root aside element touches the left or right page edge
        - A root header/div spans ≥80% of page width AND is ≤150px tall (horizontal strip)
        """
        from vision_engine.models.elements import ElementType

        root_elements = page.root_elements
        page_w = page.width

        # Find sidebar elements (aside at root level) that touch left or right edge
        sidebars = []
        for elem in root_elements:
            tag = getattr(elem, 'semantic_tag', 'div')
            if tag != 'aside':
                continue
            b = elem.bbox
            if b.x <= 20:
                sidebars.append(('left', b))
            elif b.x2 >= page_w - 20:
                sidebars.append(('right', b))

        if not sidebars:
            return

        # Extend sidebar elements to y=0 if they don't reach the top
        # (sidebar panels visually span the full page height; detection often
        # misses the top strip because a header overlaps there)
        for side, sidebar_bbox in sidebars:
            for elem in root_elements:
                tag = getattr(elem, 'semantic_tag', 'div')
                if tag != 'aside':
                    continue
                b = elem.bbox
                if b.y <= 5:
                    continue  # already at top
                if b.y > page.height * 0.25:
                    continue  # too far from top to extend safely
                # Extend to y=0
                selector = f'.el-{elem.id}'
                for i, (sel, props) in enumerate(rules):
                    if sel == selector:
                        extra = b.y
                        top_str = props.get('top', '0px')
                        new_height_str = props.get('height', '0px')
                        try:
                            old_top = int(top_str.replace('px', ''))
                            old_h = int(new_height_str.replace('px', ''))
                            props['top'] = '0px'
                            props['height'] = f'{old_h + old_top}px'
                        except ValueError:
                            pass
                        break

        # Find horizontal strip elements (headers/footers) spanning ≥80% page width
        # and look up their CSS rules to clip them
        for elem in root_elements:
            b = elem.bbox
            if b.width < page_w * 0.80:
                continue
            if b.height > 150:
                continue
            if b.x > 20:
                continue  # must start at left edge

            selector = f'.el-{elem.id}'

            for side, sidebar_bbox in sidebars:
                if side == 'right':
                    # Sidebar on the right: clip header's right edge to sidebar's left
                    new_width = sidebar_bbox.x - b.x
                    if new_width < page_w * 0.50:
                        continue  # don't clip too aggressively
                    # Update the rule if it exists
                    for i, (sel, props) in enumerate(rules):
                        if sel == selector:
                            props['width'] = f'{new_width}px'
                            break
                elif side == 'left':
                    # Sidebar on the left: shift header start to sidebar's right
                    new_left = sidebar_bbox.x2
                    new_width = b.x2 - new_left
                    if new_width < page_w * 0.50:
                        continue
                    for i, (sel, props) in enumerate(rules):
                        if sel == selector:
                            props['left'] = f'{new_left}px'
                            props['width'] = f'{new_width}px'
                            break

    # ==================================================================

    def to_string(self, rules: list) -> str:
        lines = []
        for selector, props in rules:
            lines.append(f'        {selector} {{')
            for prop, value in props.items():
                lines.append(f'            {prop}: {value};')
            lines.append(f'        }}')
            lines.append('')
        return '\n'.join(lines)
