# `tree_json` - the component tree

`mv2code.convert(...)` returns, next to `html` and `css`, a plain JSON-serialisable component tree
(`tree_json`).  It is what other tools (for example a site builder) should consume instead of
parsing the HTML.  Coordinates are in **processing pixels** (`1 px == 1 CSS px` of the screenshot;
`source_scale` tells how the screenshot was scaled to get there).

```jsonc
{
  "id": 0, "role": "page", "box": [0, 0, 1280, 1005],
  "width": 1280, "height": 1005, "direction": "rtl", "source_scale": 1.0,
  "background": "#ffffff",
  "children": [ { ...node... } ]
}
```

Every node has `id` (document order, also written to the HTML as `data-mv="<id>"`), `role` and
`box = [x, y, w, h]` (for `text` nodes the box is the *ink* box of the text).

| `role`    | meaning | extra keys |
|-----------|---------|------------|
| `section` | full-width band (hero, footer band ...) | `background` / `gradient` |
| `nav`     | header / sidebar / footer / bottom bar (`kind` = header, sidebar, footer, bottom) | |
| `card`    | any visible container: bordered, filled, shadowed | `background`, `gradient`, `border`, `radius`, `shadow` |
| `box`     | decorative / empty rectangle | same |
| `text`    | a text block (1..n lines) | `text`, `font` {family, weight, size, color}, `direction` (`rtl`/`ltr`), `align`, `lines` |
| `button`  | button (its label is the child `text`) | `text` (label) |
| `input`   | text input / textarea / select (`kind` = text, textarea, select), placeholder in `text` | |
| `badge`   | small pill with one text | `text` |
| `control` | `kind` = checkbox, radio, toggle | |
| `image`   | a real photo / illustration kept as raster | `embedded_raster: true` |
| `icon`    | small glyph, traced to inline SVG when it is single-colour | `vector: true/false` |
| `table`   | rows x columns recovered from aligned cells (becomes a real `<table>`) | children = cells in reading order |
| `divider` | thin rule | `background` |

Colours are `#rrggbb` strings, `gradient` is `{angle, from, to}` (CSS degrees), `border` is
`{top|right|bottom|left: {width, color}}`, `shadow` is `{oy, blur, alpha}` (a black box-shadow).

Typical consumer code (site builder): walk `children` of the page, group the top-level `section` /
`card` nodes into blocks, take the largest `text` (by `font.size`) as the heading, the other `text`
nodes as paragraphs, `button` nodes as calls to action and `image` nodes as pictures; `background`
and `font.color` give the palette.
