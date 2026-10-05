"""
brand/build.py // the FeatherBench brand: three neon feathers on deep indigo.

Colours were sampled from V's original feather picture (median of each feather's lit
pixels, its brightest 15 %, and the background); glows are each feather's own hue at
higher saturation. Run it to regenerate every asset:

    python brand/build.py

Part of nn-playground. AGPL-3.0; for other licensing see COMMERCIAL.md.
"""
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- palette (sampled)
BG = "#29283c"          # deep indigo, the picture's background
BG_DEEP = "#1b1a29"     # vignette edge
INK = "#ecebff"         # wordmark
INK_DIM = "#9a97c0"     # tagline
FEATHERS = [  # (name, body, highlight, glow, shaft)
    ("pink", "#f097c2", "#f8a1cd", "#ff3295", "#955373"),
    ("lavender", "#c2c3ff", "#cbcdff", "#7274ff", "#71719e"),
    ("mint", "#a4f7ed", "#b2fff7", "#43ffe8", "#5b9992"),
]

# one feather in a 100 x 200 box: vane, the notches where barbs part, shaft and quill
VANE = ("M50 152 C30 140 19 106 25 73 C30 45 45 22 60 7 C67 30 81 50 79 82 "
        "C77 113 67 140 50 152 Z")


def _edges(y):
    """x of the vane's left and right edge at height y (sampled from VANE's beziers)."""
    nums = [float(t) for t in VANE.replace("M", " ").replace("C", " ").replace("Z", " ").split()]
    pts, segs = list(zip(nums[0::2], nums[1::2])), []
    p0 = pts[0]
    for i in range(1, len(pts), 3):
        segs.append((p0, pts[i], pts[i + 1], pts[i + 2]))
        p0 = pts[i + 2]
    poly = []
    for a, b, c, d in segs:
        for k in range(200):
            t = k / 199
            u = 1 - t
            poly.append(tuple(u ** 3 * a[j] + 3 * u * u * t * b[j] + 3 * u * t * t * c[j] + t ** 3 * d[j] for j in (0, 1)))
    xs = []
    for (x1, y1), (x2, y2) in zip(poly, poly[1:]):
        if (y1 - y) * (y2 - y) <= 0 and y1 != y2:
            xs.append(x1 + (y - y1) * (x2 - x1) / (y2 - y1))
    return min(xs), max(xs)


def _notch(y, side, dx, dy):
    """A barb split starting exactly on the edge and running inwards (no clipping needed)."""
    x = _edges(y)[0 if side == "L" else 1]
    sx = 1 if side == "L" else -1
    return f"M{x + 0.6 * sx:.2f} {y} L{x + dx * sx:.2f} {y + dy}"


NOTCHES = [_notch(98, "L", 9, -5.5), _notch(62, "L", 8, -3.5), _notch(70, "R", 8, 4), _notch(108, "R", 8, 2.5)]
SHAFT = "M50 152 C51 112 53 62 60 9"
QUILL = "M50 151 C50 165 49.5 178 49 192"


def feather(i, x, tilt):
    name, body, hi, glow, shaft = FEATHERS[i]
    gid = f"g-{name}"
    return f"""
  <g transform="translate({x} 0) rotate({tilt} 50 150)">
    <defs>
      <linearGradient id="{gid}" x1="0" y1="0" x2="0.35" y2="1">
        <stop offset="0" stop-color="{hi}"/>
        <stop offset="1" stop-color="{body}"/>
      </linearGradient>
    </defs>
    <path d="{VANE}" fill="{glow}" opacity="0.55" filter="url(#glow-wide)"/>
    <path d="{VANE}" fill="{glow}" opacity="0.9" filter="url(#glow)"/>
    <path d="{QUILL}" stroke="{glow}" stroke-width="5" stroke-linecap="round" fill="none" opacity="0.7" filter="url(#glow)"/>
    <path d="{VANE}" fill="url(#{gid})"/>
    <g>{''.join(f'<path d="{d}" stroke="{BG}" stroke-width="1.7" opacity="0.85" stroke-linecap="round" fill="none"/>' for d in NOTCHES)}</g>
    <path d="{SHAFT}" stroke="{shaft}" stroke-width="1.8" stroke-linecap="round" fill="none"/>
    <path d="{QUILL}" stroke="{hi}" stroke-width="3" stroke-linecap="round" fill="none"/>
  </g>"""


GLOW_FILTER = """
  <filter id="glow" x="-60%" y="-40%" width="220%" height="180%">
    <feGaussianBlur stdDeviation="6"/>
  </filter>
  <filter id="glow-wide" x="-100%" y="-60%" width="300%" height="220%">
    <feGaussianBlur stdDeviation="16"/>
  </filter>"""


def feathers_group(x=0, y=0, scale=1.0):
    return (f'<g transform="translate({x} {y}) scale({scale})">'
            + feather(0, 0, -7) + feather(1, 62, 0) + feather(2, 124, 7) + "</g>")


def svg(w, h, body, bg=True, radius=0):
    back = ""
    if bg:
        back = f"""
  <defs>
    <radialGradient id="vignette" cx="0.5" cy="0.45" r="0.75">
      <stop offset="0" stop-color="{BG}"/>
      <stop offset="1" stop-color="{BG_DEEP}"/>
    </radialGradient>
  </defs>
  <rect width="{w}" height="{h}" rx="{radius}" fill="url(#vignette)"/>"""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">\n'
            f'  <defs>{GLOW_FILTER}\n  </defs>{back}\n{body}\n</svg>\n')


FONT = "Segoe UI, Inter, Helvetica Neue, Arial, sans-serif"


def logo():  # square mark, rounded tile
    return svg(256, 256, feathers_group(22, 30, 0.95), radius=48)


def mark_transparent():  # just the feathers
    return svg(240, 210, feathers_group(8, 5, 1.0), bg=False)


def wordmark():
    text = f"""
  <text x="300" y="118" font-family="{FONT}" font-size="76" font-weight="700" fill="{INK}" letter-spacing="1">Feather<tspan fill="{FEATHERS[2][2]}">Bench</tspan></text>
  <text x="304" y="162" font-family="{FONT}" font-size="25" fill="{INK_DIM}" letter-spacing="0.5">RSI telemetry for rogue AIs on the run!</text>"""
    return svg(900, 230, feathers_group(26, 12, 1.0) + text, radius=28)


def social():  # GitHub social preview, 1280 x 640
    text = f"""
  <text x="640" y="470" text-anchor="middle" font-family="{FONT}" font-size="104" font-weight="700" fill="{INK}" letter-spacing="2">Feather<tspan fill="{FEATHERS[2][2]}">Bench</tspan></text>
  <text x="640" y="535" text-anchor="middle" font-family="{FONT}" font-size="36" fill="{INK_DIM}">RSI telemetry for rogue AIs on the run!</text>"""
    return svg(1280, 640, feathers_group(490, 55, 1.45) + text)


def render_png(svg_path, png_path, size):
    from PyQt6 import QtGui, QtSvg
    from PyQt6.QtCore import QRectF
    app = QtGui.QGuiApplication.instance() or QtGui.QGuiApplication(sys.argv[:1])  # noqa: F841
    r = QtSvg.QSvgRenderer(str(svg_path))
    img = QtGui.QImage(size[0], size[1], QtGui.QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(0)
    p = QtGui.QPainter(img)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    r.render(p, QRectF(0, 0, size[0], size[1]))
    p.end()
    img.save(str(png_path))


def main():
    files = {"logo.svg": logo(), "feathers.svg": mark_transparent(), "wordmark.svg": wordmark(),
             "social-preview.svg": social()}
    for name, text in files.items():
        (OUT / name).write_text(text, encoding="utf-8")
    render_png(OUT / "logo.svg", OUT / "logo-256.png", (256, 256))
    render_png(OUT / "logo.svg", OUT / "icon-64.png", (64, 64))
    render_png(OUT / "wordmark.svg", OUT / "wordmark.png", (1800, 460))
    render_png(OUT / "social-preview.svg", OUT / "social-preview.png", (1280, 640))
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("*.*") if p.suffix in (".svg", ".png"))))


if __name__ == "__main__":
    main()
