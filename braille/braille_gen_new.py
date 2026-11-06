"""Random braille art: a random source image, dithered into braille cells.

The textures that look best are photos converted to braille: they have shapes,
sharp edges, smooth tonal gradients and, in empty areas, the regular dotted
lines that braille converters draw when they pad blank cells. Independent coin
flips per dot cannot produce any of that. So this generator follows the same
pipeline as the converter, with the source image drawn at random:

    random source image  ->  tone map  ->  dither to dots  ->  pad empty cells
    (clouds / waves /        (fade ramp,    (Floyd-Steinberg,   (dots 7+8, as
     ridges + curves)         figures)       Bayer or random)    the converter)

Cells sit on an exact integer lattice (no warp, no jitter), and the default
"laze" style reproduces the pixel grid measured from real converter output:
8x17 px cells, 2px dots, 4px dot pitch, 3.67px row pitch at --font-size 17.
Double --font-size for crisp dots on high-DPI screens.

Everything is reproducible from one seed; each run draws a fresh one unless
--seed is given. --tile writes a seamless repeating tile for web backgrounds.
"""

from __future__ import annotations

import argparse
import os
import secrets
from dataclasses import dataclass, replace

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# --- Braille cell -----------------------------------------------------------

DOTS_X = 2
DOTS_Y = 4
SLOTS = DOTS_X * DOTS_Y

# Unicode braille bit order: bit i of the code point offset is the dot at
# (row, col) below. Dots 1-3 run down the left column, 4-6 down the right, then
# 7 (left) and 8 (right) sit in the bottom row.
BIT_POS = ((0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, 0), (3, 1))
BOTTOM_ROW = (6, 7)  # bits of dots 7 and 8, the padding converters use

# --- Tunables ---------------------------------------------------------------

OUTPUT_WIDTH = 2048
OUTPUT_HEIGHT = 10240

# The em square in output pixels; every font metric is a fraction of it.
# 17 reproduces the measured converter grid exactly (8x17 px cells).
FONT_SIZE = 17

# Defaults used by --fixed (and as the centre of the random ranges).
FULL_BAND = 0.4
FALLOFF_EXP = 1.6
FIGURE_COUNT = 90
DENSITY_PULL = 0.30
FIGURE_MIN_PX = 200
FIGURE_MAX_PX = 800

DIRECTIONS = ("down", "up", "left", "right", "radial", "flat")
SOURCES = ("clouds", "waves", "ridges")
DITHERS = ("fs", "bayer", "random", "threshold")
FILLS = ("bottom", "none")

OUT_DIR = "out"


@dataclass(frozen=True)
class Style:
    """Braille glyph metrics, in em."""

    advance: float  # cell width (horizontal advance)
    line: float  # cell height (line pitch)
    pitch_x: float  # distance between the two dot columns
    pitch_y: float  # distance between neighbouring dot rows
    diameter: float  # dot diameter


STYLES = {
    # Measured from converted images (4166x3060 px): columns every 4px with no
    # extra gap between cells, rows at 2.5/6.5/9.5/13.5 within a 17px line,
    # dots 2px across. Expressed against a 17px em.
    "laze": Style(advance=0.470, line=1.000, pitch_x=0.235, pitch_y=0.2157, diameter=0.118),
    # IPAexGothic's 1.0 em line height with typical braille glyph proportions.
    "ipaex": Style(advance=0.600, line=1.000, pitch_x=0.225, pitch_y=0.225, diameter=0.140),
    # Measured from DejaVu Sans U+28FF: wider cells, taller lines.
    "dejavu": Style(advance=0.732, line=1.164, pitch_x=0.293, pitch_y=0.277, diameter=0.146),
}
DEFAULT_STYLE = "laze"


def style_from_font(path: str, line_height: float | None = None) -> Style:
    """Measure a Style from the U+28FF glyph (all eight dots) of a font file."""
    try:
        from fontTools.pens.recordingPen import DecomposingRecordingPen
        from fontTools.ttLib import TTFont
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("--font-file needs fontTools: pip install fonttools") from exc

    font = TTFont(path)
    em = font["head"].unitsPerEm
    cmap = font.getBestCmap()
    if 0x28FF not in cmap:
        raise SystemExit(f"{path} has no braille glyphs (U+28FF missing)")
    name = cmap[0x28FF]
    glyphs = font.getGlyphSet()
    pen = DecomposingRecordingPen(glyphs)
    glyphs[name].draw(pen)

    contours: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    for op, args in pen.value:
        if op == "moveTo":
            current = [args[0]]
        elif op in ("lineTo", "qCurveTo", "curveTo"):
            current += [a for a in args if a]
        elif op in ("closePath", "endPath"):
            contours.append(current)
            current = []
    if len(contours) != SLOTS:
        raise SystemExit(f"U+28FF has {len(contours)} contours, expected {SLOTS} dots")

    cx, cy, size = [], [], []
    for pts in contours:
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        cx.append((min(xs) + max(xs)) / 2 / em)
        cy.append((min(ys) + max(ys)) / 2 / em)
        size.append(((max(xs) - min(xs)) + (max(ys) - min(ys))) / 2 / em)

    cx, cy = np.sort(cx), np.sort(cy)
    pitch_x = float(cx[4:].mean() - cx[:4].mean())
    rows = cy.reshape(4, 2).mean(axis=1)
    pitch_y = float((rows[3] - rows[0]) / 3)
    hhea = font["hhea"]
    line = line_height or (hhea.ascent - hhea.descent + hhea.lineGap) / em
    return Style(
        advance=font["hmtx"][name][0] / em,
        line=float(line),
        pitch_x=pitch_x,
        pitch_y=pitch_y,
        diameter=float(np.mean(size)),
    )


# --- Layout -----------------------------------------------------------------


@dataclass(frozen=True)
class Layout:
    """Integer cell lattice plus one antialiased sprite per dot slot."""

    cell_w: int
    cell_h: int
    cols: int
    rows: int
    pad_x: int  # leftover pixels are split evenly, so the grid stays centred
    pad_y: int
    # per slot: (y offset in cell, x offset in cell, coverage 0..255 sprite)
    sprites: tuple

    @property
    def dot_shape(self) -> tuple[int, int]:
        return self.rows * DOTS_Y, self.cols * DOTS_X


def dot_sprite(cx: float, cy: float, radius: float, supersample: int = 16):
    """Antialiased disc centred at (cx, cy) in cell coordinates."""
    x0 = int(np.floor(cx - radius)) - 1
    y0 = int(np.floor(cy - radius)) - 1
    size = int(np.ceil(2 * radius)) + 3
    axis = (np.arange(size * supersample) + 0.5) / supersample
    xs = x0 + axis[None, :]
    ys = y0 + axis[:, None]
    inside = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius**2
    cover = inside.reshape(size, supersample, size, supersample).mean(axis=(1, 3))
    return y0, x0, np.rint(cover * 255).astype(np.uint8)


def make_layout(style: Style, font_size: int, width: int, height: int, snap: bool = True) -> Layout:
    cell_w = max(1, int(round(style.advance * font_size)))
    cell_h = max(1, int(round(style.line * font_size)))
    cols, rows = width // cell_w, height // cell_h
    if cols < 1 or rows < 1:
        raise SystemExit("canvas is smaller than one braille cell")

    radius = style.diameter * font_size / 2
    px, py = style.pitch_x * font_size, style.pitch_y * font_size
    # Snapping puts every dot centre on a whole pixel (an even-width dot is
    # centred on a pixel edge, an odd one on a pixel centre), like hinted
    # glyphs. Without it, tiny dots land on different fractions and render with
    # visibly different brightness from row to row.
    off = 0.0 if int(round(2 * radius)) % 2 == 0 else 0.5
    sprites = []
    for row, col in BIT_POS:
        cx = cell_w / 2 + (col - 0.5) * px
        cy = cell_h / 2 + (row - 1.5) * py
        if snap:
            cx, cy = round(cx - off) + off, round(cy - off) + off
        y0, x0, cover = dot_sprite(cx, cy, radius)
        # The sprite carries a 1px antialiasing margin that may poke past the
        # cell edge while being empty; crop it, and fail only on real coverage.
        top, left = max(0, -y0), max(0, -x0)
        bottom = max(0, y0 + cover.shape[0] - cell_h)
        right = max(0, x0 + cover.shape[1] - cell_w)
        h, w = cover.shape
        inner = cover[top : h - bottom, left : w - right]
        if inner.sum() != cover.sum():
            raise SystemExit(
                "dots do not fit inside the cell: lower the dot pitch/diameter, "
                "raise --line-height or raise --font-size"
            )
        sprites.append((y0 + top, x0 + left, np.ascontiguousarray(inner)))

    return Layout(
        cell_w=cell_w,
        cell_h=cell_h,
        cols=cols,
        rows=rows,
        pad_x=(width - cols * cell_w) // 2,
        pad_y=(height - rows * cell_h) // 2,
        sprites=tuple(sprites),
    )


# --- Source image: periodic value noise -------------------------------------


def smoothstep(lo: float, hi: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def sample_noise(lattice: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Smooth value noise at float lattice coordinates; wraps at the edges.

    Because the lattice wraps, every field built from it tiles seamlessly.
    """
    ny, nx = lattice.shape
    iy, ix = np.floor(y).astype(np.int64), np.floor(x).astype(np.int64)
    fy, fx = y - iy, x - ix
    fy, fx = fy * fy * (3 - 2 * fy), fx * fx * (3 - 2 * fx)
    y0, y1 = iy % ny, (iy + 1) % ny
    x0, x1 = ix % nx, (ix + 1) % nx
    top = lattice[y0, x0] * (1 - fx) + lattice[y0, x1] * fx
    bot = lattice[y1, x0] * (1 - fx) + lattice[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def fbm(rng, v, u, ny: int, nx: int, octaves: int, gain: float = 0.55, dv=0.0, du=0.0):
    """Fractal noise on the unit canvas, normalised to roughly [0, 1].

    v, u are fractions of the canvas height/width; (dv, du) displace the sample
    point in base-lattice units, which is how domain warping is done.
    """
    total = 0.0
    amp, norm = 1.0, 0.0
    for k in range(octaves):
        f = 2**k
        lattice = rng.random((ny * f, nx * f)).astype(np.float32)
        total = total + amp * sample_noise(lattice, (v * ny + dv) * f, (u * nx + du) * f)
        norm += amp
        amp *= gain
    return total / norm


def source_tone(rng, rows: int, cols: int, p: "Params", width: int, height: int) -> np.ndarray:
    """A random grayscale 'photo' at dot resolution, values in [0, 1]."""
    v = ((np.arange(rows) + 0.5) / rows)[:, None]
    u = ((np.arange(cols) + 0.5) / cols)[None, :]
    feature = max(p.feature, 16.0)
    ny = max(1, int(round(height / feature)))
    nx = max(1, int(round(width / feature)))

    # One low-frequency field drives the warp; the rest differs per kind.
    wy = fbm(rng, v, u, ny, nx, 3) - 0.5
    wx = fbm(rng, v, u, ny, nx, 3) - 0.5
    dv, du = p.warp * 2 * wy, p.warp * 2 * wx

    if p.source == "clouds":
        b = fbm(rng, v, u, ny, nx, 6, dv=dv, du=du)
    elif p.source == "ridges":
        f = fbm(rng, v, u, ny, nx, 5, dv=dv, du=du)
        b = (1.0 - np.abs(2.0 * f - 1.0)) ** 2
    else:  # waves: stretched swell + fine chop, strongly warped so nothing is periodic
        f = fbm(rng, v, u, ny * 4, max(1, nx // 3), 6, gain=0.6, dv=dv * 0.8, du=du * 0.8)
        chop = fbm(rng, v, u, ny * 10, max(1, nx), 3, dv=dv, du=du)
        cycles = max(3, int(round(height / (feature / 4.0))))
        ripple = 0.5 + 0.5 * np.sin(2 * np.pi * (v * cycles + 1.5 * p.warp * wy + 2.0 * (f - 0.5)))
        b = 0.25 * ripple + 0.50 * f + 0.25 * chop

    lo, hi = np.percentile(b, (2.0, 98.0))
    b = np.clip((b - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

    # Edge shaping: blend the smooth tone with a hard threshold of it, so some
    # areas keep gradients and others get the crisp silhouettes of a photo.
    centre = rng.uniform(0.35, 0.65)
    half = 0.02 + 0.25 * (1.0 - p.edge)
    shaped = (1.0 - p.edge) * b + p.edge * smoothstep(centre - half, centre + half, b)
    return shaped**p.gamma


# --- Tone: fade ramp + figure strokes ---------------------------------------


def fade_ramp(rows: int, cols: int, direction: str, full_band: float, exp: float, flat: float):
    """Fill probability: 1.0 across `full_band`, then fading to 0."""
    y = ((np.arange(rows) + 0.5) / rows)[:, None] * np.ones((1, cols))
    x = np.ones((rows, 1)) * ((np.arange(cols) + 0.5) / cols)[None, :]
    if direction == "flat":
        return np.full((rows, cols), flat)
    if direction == "down":
        t = y
    elif direction == "up":
        t = 1.0 - y
    elif direction == "right":
        t = x
    elif direction == "left":
        t = 1.0 - x
    elif direction == "radial":
        t = np.clip(np.hypot(2 * (x - 0.5), 2 * (y - 0.5)) / np.sqrt(2.0), 0.0, 1.0)
    else:
        raise ValueError(f"unknown direction {direction!r}")
    fade = np.clip((1.0 - t) / (1.0 - full_band), 0.0, 1.0)
    return np.where(t <= full_band, 1.0, fade**exp)


def figure_curve(rng: np.random.Generator, width: int, height: int, lo=FIGURE_MIN_PX, hi=FIGURE_MAX_PX):
    """One random curve: returns (points in px, ridge width in px, sign).

    Three kinds: a wide arc, a tight coil, or an elongated S-curve, each a
    cubic sampled densely enough to read as a smooth stroke.
    """
    size = float(rng.uniform(lo, hi))
    centre = np.array([rng.uniform(0.0, width), rng.uniform(0.0, height)])
    angle = rng.uniform(0.0, 2.0 * np.pi)
    along = np.array([np.cos(angle), np.sin(angle)])
    across = np.array([-np.sin(angle), np.cos(angle)])

    kind = rng.integers(0, 3)
    if kind == 0:  # arc
        bend = size * rng.uniform(0.4, 0.9) * rng.choice((-1.0, 1.0))
        p0, p1 = centre - along * size, centre + along * size
        c0 = p0 + along * size * 0.5 + across * bend
        c1 = p1 - along * size * 0.5 + across * bend
    elif kind == 1:  # coil
        p0 = centre
        p1 = p0 + along * size
        c0 = p0 + across * size * 0.6 + along * size * 0.2
        c1 = p1 - across * size * 0.6 - along * size * 0.2
    else:  # S-curve
        p0, p1 = centre - along * size, centre + along * size
        swing = size * rng.uniform(0.5, 1.0) * rng.choice((-1.0, 1.0))
        c0 = p0 + along * size * 0.3 + across * swing
        c1 = p1 - along * size * 0.3 - across * swing

    t = np.linspace(0.0, 1.0, 96)[:, None]
    inv = 1.0 - t
    pts = inv**3 * p0 + 3 * inv**2 * t * c0 + 3 * inv * t**2 * c1 + t**3 * p1
    ridge = size * rng.uniform(0.04, 0.10)
    return pts, ridge, float(rng.choice((-1.0, 1.0)))


def figure_field(rng, layout: Layout, count: int, pull: float, width: int, height: int,
                 wrap: bool = False, size_range=(FIGURE_MIN_PX, FIGURE_MAX_PX)) -> np.ndarray:
    """Soft signed ridges along random curves, at dot resolution, in +-`pull`.

    Each curve is drawn as a thin line and blurred, which gives a Gaussian
    ridge for a fraction of the cost of a per-dot distance field. With `wrap`
    the curve is also drawn shifted by one canvas in every direction, so a
    ridge leaving one edge re-enters on the opposite one (seamless tiles).
    """
    rows, cols = layout.dot_shape
    sx, sy = layout.cell_w / DOTS_X, layout.cell_h / DOTS_Y  # px per dot
    shifts = [(dx, dy) for dx in (-width, 0, width) for dy in (-height, 0, height)] if wrap else [(0, 0)]

    field = np.zeros((rows, cols), dtype=np.float32)
    for _ in range(count):
        pts, ridge, sign = figure_curve(rng, width, height, *size_range)
        canvas = Image.new("L", (cols, rows), 0)
        draw = ImageDraw.Draw(canvas)
        for dx, dy in shifts:
            xs = (pts[:, 0] + dx - layout.pad_x) / sx
            ys = (pts[:, 1] + dy - layout.pad_y) / sy
            draw.line(list(zip(xs.tolist(), ys.tolist())), fill=255, width=1)
        sigma = max(1.0, ridge / ((sx + sy) / 2))
        arr = np.asarray(canvas.filter(ImageFilter.GaussianBlur(sigma)), dtype=np.float32)
        peak = arr.max()
        if peak > 0:
            field += sign * arr / peak
    return np.clip(field, -pull, pull)


# --- Dithering --------------------------------------------------------------


def bayer_matrix(n: int) -> np.ndarray:
    m = np.array([[0]])
    while m.shape[0] < n:
        m = np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
    return m


def floyd_steinberg(tone: np.ndarray) -> np.ndarray:
    """Serpentine Floyd-Steinberg error diffusion; True where a dot is lit."""
    h, w = tone.shape
    buf = tone.astype(np.float64).tolist()
    out = np.zeros((h, w), dtype=bool)
    for y in range(h):
        row = buf[y]
        nxt = buf[y + 1] if y + 1 < h else None
        d = 1 if y % 2 == 0 else -1
        xs = range(w) if d == 1 else range(w - 1, -1, -1)
        lit = []
        for x in xs:
            old = row[x]
            new = 1.0 if old > 0.5 else 0.0
            if new:
                lit.append(x)
            err = old - new
            xn = x + d
            if 0 <= xn < w:
                row[xn] += err * 0.4375
            if nxt is not None:
                if 0 <= x - d < w:
                    nxt[x - d] += err * 0.1875
                nxt[x] += err * 0.3125
                if 0 <= xn < w:
                    nxt[xn] += err * 0.0625
        out[y, lit] = True
    return out


def dither(rng, tone: np.ndarray, method: str) -> np.ndarray:
    if method == "fs":
        return floyd_steinberg(tone)
    if method == "bayer":
        n = 8
        thr = (bayer_matrix(n) + 0.5) / (n * n)
        reps = (-(-tone.shape[0] // n), -(-tone.shape[1] // n))
        return tone > np.tile(thr, reps)[: tone.shape[0], : tone.shape[1]]
    if method == "random":
        return rng.random(tone.shape) < tone
    return tone > 0.5


def cells_from_dots(dots: np.ndarray) -> np.ndarray:
    """(rows*4, cols*2) dot mask -> (rows, cols, 8) with slot i = braille bit i."""
    return np.stack([dots[r::DOTS_Y, c::DOTS_X] for r, c in BIT_POS], axis=-1)


# --- Random parameters ------------------------------------------------------


@dataclass(frozen=True)
class Params:
    direction: str
    full_band: float
    falloff: float
    flat: float
    figures: int
    pull: float
    source: str
    feature: float  # source feature size, px
    warp: float  # domain-warp strength, in lattice cells
    edge: float  # 0 = smooth gradients, 1 = hard silhouettes
    floor: float  # tone floor so dark areas keep a trace of dots
    gamma: float  # >1 darkens the source: sparse bright detail on dark, like a photo


def random_params(rng: np.random.Generator, args, width: int) -> Params:
    """Draw every look parameter from the seed unless the user pinned it."""
    if args.fixed:
        drawn = Params("down", FULL_BAND, FALLOFF_EXP, 0.5, FIGURE_COUNT, DENSITY_PULL,
                       "waves", 0.3 * width, 1.4, 0.6, 0.05, 1.8)
    else:
        drawn = Params(
            direction=str(rng.choice(DIRECTIONS, p=[0.30, 0.15, 0.10, 0.10, 0.25, 0.10])),
            full_band=float(rng.uniform(0.15, 0.55)),
            falloff=float(rng.uniform(1.0, 2.4)),
            flat=float(rng.uniform(0.55, 0.90)),
            figures=int(rng.integers(20, 71)),
            pull=float(rng.uniform(0.15, 0.35)),
            source=str(rng.choice(SOURCES)),
            feature=float(rng.uniform(0.12, 0.45) * width),
            warp=float(rng.uniform(0.6, 2.2)),
            edge=float(rng.uniform(0.3, 0.9)),
            floor=float(rng.uniform(0.0, 0.12)),
            gamma=float(rng.uniform(1.0, 2.8)),
        )
    if args.tile:  # a repeating tile cannot carry a gradient: CSS does the fade
        drawn = replace(drawn, direction="flat", figures=int(rng.integers(6, 15)))
        if args.fixed:
            drawn = replace(drawn, figures=10)
    return replace(drawn, direction=args.direction or drawn.direction, source=args.source or drawn.source)


# --- Rendering --------------------------------------------------------------


def render_alpha(pattern: np.ndarray, layout: Layout, width: int, height: int) -> np.ndarray:
    """Stamp each dot slot's sprite into every cell that has it; uint8 (h, w)."""
    cells = np.zeros((layout.rows, layout.cell_h, layout.cols, layout.cell_w), dtype=np.uint8)
    for bit, (y0, x0, cover) in enumerate(layout.sprites):
        h, w = cover.shape
        mask = pattern[:, :, bit]
        region = cells[:, y0 : y0 + h, :, x0 : x0 + w]  # view: (rows, h, cols, w)
        stamp = mask[:, None, :, None] * cover[None, :, None, :]
        np.maximum(region, stamp, out=region)  # AA edges of neighbours may touch

    grid = cells.reshape(layout.rows * layout.cell_h, layout.cols * layout.cell_w)
    canvas = np.zeros((height, width), dtype=np.uint8)
    canvas[
        layout.pad_y : layout.pad_y + grid.shape[0],
        layout.pad_x : layout.pad_x + grid.shape[1],
    ] = grid
    return canvas


def to_text(pattern: np.ndarray) -> str:
    """The same pattern as real Unicode braille characters, one line per row."""
    codes = (pattern.astype(np.uint16) << np.arange(SLOTS)).sum(axis=-1)
    return "\n".join("".join(chr(0x2800 + int(c)) for c in row) for row in codes)


def write_image(alpha: np.ndarray, ink, bg, path: str, mask: bool = False) -> None:
    """Save as PNG/WebP (by extension). `mask` stores coverage only, one channel."""
    lossless = {"lossless": True, "method": 6} if path.endswith(".webp") else {"optimize": True}
    if mask:  # use as CSS mask-image: colour and background come from the page
        Image.fromarray(alpha, "L").save(path, **lossless)
        return
    layer = Image.new("RGBA", alpha.shape[::-1], ink + (255,))
    layer.putalpha(Image.fromarray(alpha, "L"))
    if bg is not None:
        base = Image.new("RGBA", alpha.shape[::-1], bg + (255,))
        layer = Image.alpha_composite(base, layer)
    layer.save(path, **lossless)


# --- Verification -----------------------------------------------------------


def check_alignment(layout: Layout, style: Style, font_size: int) -> None:
    """Assert the lattice is exact: every cell renders pixel-identically."""
    full = np.ones((layout.rows, layout.cols, SLOTS), dtype=bool)
    width = layout.pad_x * 2 + layout.cols * layout.cell_w
    height = layout.pad_y * 2 + layout.rows * layout.cell_h
    alpha = render_alpha(full, layout, width, height)
    grid = alpha[
        layout.pad_y : layout.pad_y + layout.rows * layout.cell_h,
        layout.pad_x : layout.pad_x + layout.cols * layout.cell_w,
    ]
    tiles = grid.reshape(layout.rows, layout.cell_h, layout.cols, layout.cell_w)
    assert (tiles == tiles[0:1, :, 0:1, :]).all(), "cells are not pixel-identical"

    # Coverage of a full cell must equal eight discs of the requested diameter.
    expected = SLOTS * np.pi * (style.diameter * font_size / 2) ** 2
    measured = float(tiles[0, :, 0, :].sum()) / 255
    assert abs(measured - expected) < 0.06 * expected, (
        f"dot area {measured:.1f}px vs {expected:.1f}px expected"
    )


def check_dither(dots: np.ndarray, tone: np.ndarray, method: str) -> None:
    """Assert dithering preserved the average tone (error diffusion must)."""
    if method == "threshold":
        return
    gap = abs(float(dots.mean()) - float(tone.mean()))
    assert gap < 0.03, f"{method} dithering drifted: dots {dots.mean():.3f} vs tone {tone.mean():.3f}"


# --- Entry point ------------------------------------------------------------


def parse_color(text: str) -> tuple[int, int, int]:
    text = text.lstrip("#")
    if len(text) != 6:
        raise argparse.ArgumentTypeError("colour must be 6 hex digits, e.g. 101820")
    return tuple(int(text[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=1, help="how many images to generate")
    parser.add_argument("--seed", help="hex seed for one image; omit for a fresh random one")
    parser.add_argument("--width", type=int, help=f"default {OUTPUT_WIDTH}, or ~512 with --tile")
    parser.add_argument("--height", type=int, help=f"default {OUTPUT_HEIGHT}, or ~512 with --tile")
    parser.add_argument("--font-size", type=int, default=FONT_SIZE, help="em size in px (17 = measured grid)")
    parser.add_argument("--style", choices=sorted(STYLES), default=DEFAULT_STYLE)
    parser.add_argument("--font-file", help="measure glyph metrics from this TTF/OTF instead")
    parser.add_argument("--line-height", type=float, help="override line height, in em")
    parser.add_argument("--no-snap", action="store_true", help="keep sub-pixel dot centres (default: snap to whole pixels)")
    parser.add_argument("--source", choices=SOURCES, help="pin the kind of random source image")
    parser.add_argument("--dither", choices=DITHERS, default="fs", help="how the source becomes dots")
    parser.add_argument("--fill", choices=FILLS, default="bottom",
                        help="pad empty cells with dots 7+8 like braille converters do, or leave blank")
    parser.add_argument("--direction", choices=DIRECTIONS, help="pin the fade direction")
    parser.add_argument("--fixed", action="store_true", help="use the constants, not random looks")
    parser.add_argument("--ink", choices=("white", "black", "both"), default="both")
    parser.add_argument("--bg", type=parse_color, help="opaque background hex, e.g. 101820")
    parser.add_argument("--tile", action="store_true",
                        help="seamless repeating tile (size snapped to whole cells, no gradient)")
    parser.add_argument("--webp", action="store_true", help="write lossless WebP instead of PNG (much smaller)")
    parser.add_argument("--mask", action="store_true",
                        help="one-channel coverage mask (use with CSS mask-mode: luminance); RGBA is simpler")
    parser.add_argument("--txt", action="store_true", help="also write Unicode braille text")
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    if args.font_file:
        style = style_from_font(args.font_file, args.line_height)
    else:
        style = STYLES[args.style]
        if args.line_height:
            style = replace(style, line=args.line_height)

    args.width = args.width or (512 if args.tile else OUTPUT_WIDTH)
    args.height = args.height or (512 if args.tile else OUTPUT_HEIGHT)
    layout = make_layout(style, args.font_size, args.width, args.height, not args.no_snap)
    if args.tile:
        # Whole cells only, so the lattice lines up across the wrap: no seam.
        args.width, args.height = layout.cols * layout.cell_w, layout.rows * layout.cell_h
        layout = make_layout(style, args.font_size, args.width, args.height, not args.no_snap)
        assert layout.pad_x == 0 and layout.pad_y == 0
    check_alignment(layout, style, args.font_size)
    print(
        f"grid {layout.cols}x{layout.rows} cells of {layout.cell_w}x{layout.cell_h}px "
        f"(em {args.font_size}px, dot {style.diameter * args.font_size:.1f}px)"
    )

    inks = {"white": (255, 255, 255), "black": (0, 0, 0)}
    chosen = ("white", "black") if args.ink == "both" else (args.ink,)
    os.makedirs(args.out_dir, exist_ok=True)
    rows_d, cols_d = layout.dot_shape

    for n in range(args.count):
        seed = args.seed if (args.seed and args.count == 1) else secrets.token_bytes(32).hex()
        rng = np.random.default_rng(np.random.SeedSequence(int.from_bytes(bytes.fromhex(seed), "big")))

        p = random_params(rng, args, args.width)
        tone = source_tone(rng, rows_d, cols_d, p, args.width, args.height)
        ramp = fade_ramp(rows_d, cols_d, p.direction, p.full_band, p.falloff, p.flat)
        short = min(args.width, args.height)
        size_range = (0.12 * short, 0.50 * short) if args.tile else (FIGURE_MIN_PX, FIGURE_MAX_PX)
        field = figure_field(rng, layout, p.figures, p.pull, args.width, args.height,
                             wrap=args.tile, size_range=size_range)

        # Figures fade out with the ramp so they cannot refill emptied areas.
        weight = np.clip(ramp / max(p.pull, 1e-6), 0.0, 1.0)
        target = np.clip(ramp * (p.floor + (1.0 - p.floor) * tone) + field * weight, 0.0, 1.0)

        dots = dither(rng, target, args.dither)
        check_dither(dots, target, args.dither)
        pattern = cells_from_dots(dots)
        if args.fill == "bottom":
            empty = ~pattern.any(axis=-1)
            for bit in BOTTOM_ROW:
                pattern[:, :, bit] |= empty
        alpha = render_alpha(pattern, layout, args.width, args.height)

        stem = f"braille-{seed[:8]}"
        print(
            f"[{n + 1}/{args.count}] seed: {seed}\n"
            f"    {p.source}, fade {p.direction}, band {p.full_band:.2f}, edge {p.edge:.2f}, "
            f"warp {p.warp:.1f}, {p.figures} figures, dither {args.dither}"
        )
        ext = "webp" if args.webp else "png"
        # A mask has no colour of its own, so one file serves every colourway.
        for name in (("mask",) if args.mask else chosen):
            path = os.path.join(args.out_dir, f"{stem}{'-tile' if args.tile else ''}-{name}.{ext}")
            write_image(alpha, inks.get(name), args.bg, path, mask=args.mask)
            kb = os.path.getsize(path) / 1024
            print(f"    wrote {path} {args.width}x{args.height} ({kb:.0f} KB)")
        if args.txt:
            path = os.path.join(args.out_dir, f"{stem}.txt")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(to_text(pattern) + "\n")
            print(f"    wrote {path}")


if __name__ == "__main__":
    main()
