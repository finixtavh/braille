"""Random braille-font textures as RGBA PNGs, on a perfectly aligned grid.

This renders the *look* of braille text set in a font (the output of tools such
as lazesoftware.com/en/tool/brailleaagen with the Japanese / IPAexGothic
preset): small round dots, two columns and four rows per cell, with visibly
larger gaps between cells and between lines than between the dots of a cell.

What is different from the old generator (kept as braille_gen_legacy.py):

  * Cells sit on an exact integer lattice. There is no warp and no per-dot
    jitter, so every cell is pixel-identical in geometry; nothing drifts, and
    the misalignment that mixed fonts cause in the browser cannot happen.
  * Cell geometry comes from font metrics (advance, line height, dot pitch and
    dot diameter, all in em), not from an even dot pitch.
  * Everything is random and reproducible from one seed: the fade direction,
    the dense band, the falloff, the number and shape of the figure curves and
    every dot. Each run draws a fresh seed unless --seed is given.
  * Optionally writes the same pattern as real Unicode braille text (--txt).

About the font: IPAexGothic itself has no U+2800..U+28FF glyphs (0 of 256), so
a browser draws braille from a fallback font and only the line height of
IPAexGothic (exactly 1.0 em) is kept. The default "ipaex" style combines that
1.0 em line height with the proportions of a typical braille glyph. Use
--style, --font-file and --line-height to match a specific result.
"""

from __future__ import annotations

import argparse
import os
import secrets
from dataclasses import dataclass, replace

import numpy as np
from PIL import Image

# --- Braille cell -----------------------------------------------------------

DOTS_X = 2
DOTS_Y = 4
SLOTS = DOTS_X * DOTS_Y

# Unicode braille bit order: bit i of the code point offset is the dot at
# (row, col) below. Dots 1-3 run down the left column, 4-6 down the right, then
# 7 (left) and 8 (right) sit in the bottom row.
BIT_POS = ((0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, 0), (3, 1))

# --- Tunables ---------------------------------------------------------------

# Output image size in pixels.
OUTPUT_WIDTH = 2048
OUTPUT_HEIGHT = 10240

# The em square, in output pixels. Every font metric below is a fraction of it,
# so this one number scales the whole texture: bigger = bolder and chunkier.
FONT_SIZE = 40

# Defaults used by --fixed (and as the centre of the random ranges).
FULL_BAND = 0.4  # fraction of the canvas that stays fully dense before the fade
FALLOFF_EXP = 1.6  # >1 stays dense longer then drops, <1 thins out early
FIGURE_COUNT = 90
DENSITY_PULL = 0.30  # peak density change a figure can apply
FIGURE_MIN_PX = 200
FIGURE_MAX_PX = 800

DIRECTIONS = ("down", "up", "left", "right", "radial", "flat")

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
    # 1.0 em line height is IPAexGothic's own (ascent 1802 + descent 246 over
    # 2048 units); glyph proportions follow a typical braille font (FreeMono).
    "ipaex": Style(advance=0.600, line=1.000, pitch_x=0.225, pitch_y=0.225, diameter=0.140),
    # Measured from DejaVu Sans U+28FF: wider cells, taller lines.
    "dejavu": Style(advance=0.732, line=1.164, pitch_x=0.293, pitch_y=0.277, diameter=0.146),
}
DEFAULT_STYLE = "ipaex"


def style_from_font(path: str, line_height: float | None = None) -> Style:
    """Measure a Style from the U+28FF glyph (all eight dots) of a font file.

    The dot centres and diameters come straight from the glyph outline, so the
    result matches what that font draws. The line height comes from the font's
    hhea table unless `line_height` overrides it.
    """
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


def dot_sprite(cx: float, cy: float, radius: float, supersample: int = 16):
    """Analytic-ish antialiased disc centred at (cx, cy) in cell coordinates."""
    x0 = int(np.floor(cx - radius)) - 1
    y0 = int(np.floor(cy - radius)) - 1
    size = int(np.ceil(2 * radius)) + 3
    axis = (np.arange(size * supersample) + 0.5) / supersample
    xs = x0 + axis[None, :]
    ys = y0 + axis[:, None]
    inside = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius**2
    cover = inside.reshape(size, supersample, size, supersample).mean(axis=(1, 3))
    return y0, x0, np.rint(cover * 255).astype(np.uint8)


def make_layout(style: Style, font_size: int, width: int, height: int) -> Layout:
    cell_w = max(1, int(round(style.advance * font_size)))
    cell_h = max(1, int(round(style.line * font_size)))
    cols, rows = width // cell_w, height // cell_h
    if cols < 1 or rows < 1:
        raise SystemExit("canvas is smaller than one braille cell")

    radius = style.diameter * font_size / 2
    px, py = style.pitch_x * font_size, style.pitch_y * font_size
    sprites = []
    for row, col in BIT_POS:
        cx = cell_w / 2 + (col - 0.5) * px
        cy = cell_h / 2 + (row - 1.5) * py
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


# --- Density: ramp + figures ------------------------------------------------


def fade_ramp(layout: Layout, direction: str, full_band: float, exp: float, flat: float):
    """Per-cell fill probability: 1.0 across `full_band`, then fading to 0."""
    rows, cols = layout.rows, layout.cols
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


def figure_curve(rng: np.random.Generator, width: int, height: int):
    """One random distortion curve: returns (points, ridge width, sign).

    Three kinds: a wide arc, a tight coil, or an elongated S-curve, each a
    cubic sampled densely enough that the distance field sees a smooth ridge.
    """
    size = float(rng.uniform(FIGURE_MIN_PX, FIGURE_MAX_PX))
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


def figure_field(rng, layout: Layout, count: int, pull: float, width: int, height: int):
    """Sum a soft Gaussian ridge per curve, clamped to +-`pull`, per cell."""
    rows, cols = layout.rows, layout.cols
    ys, xs = np.mgrid[0:rows, 0:cols]
    px = (layout.pad_x + (xs + 0.5) * layout.cell_w).astype(np.float32)[..., None]
    py = (layout.pad_y + (ys + 0.5) * layout.cell_h).astype(np.float32)[..., None]

    field = np.zeros((rows, cols), dtype=np.float32)
    for _ in range(count):
        pts, ridge, sign = figure_curve(rng, width, height)
        qx, qy = pts[:, 0].astype(np.float32), pts[:, 1].astype(np.float32)
        d2 = ((px - qx) ** 2 + (py - qy) ** 2).min(axis=2)
        field += sign * np.exp(-d2 / (2.0 * ridge**2))
    return np.clip(field, -pull, pull)


def dot_pattern(rng, ramp: np.ndarray, field: np.ndarray, pull: float) -> np.ndarray:
    """Boolean (rows, cols, 8) mask; slot i is Unicode braille bit i.

    Figures fade out together with the ramp, so a positive figure cannot refill
    cells the gradient already emptied.
    """
    weight = np.clip(ramp / max(pull, 1e-6), 0.0, 1.0)
    density = np.clip(ramp + field * weight, 0.0, 1.0)
    return rng.random((*density.shape, SLOTS)) < density[..., None]


# --- Random parameters ------------------------------------------------------


@dataclass(frozen=True)
class Params:
    direction: str
    full_band: float
    falloff: float
    flat: float
    figures: int
    pull: float


def random_params(rng: np.random.Generator, args) -> Params:
    """Draw every look parameter from the seed unless the user pinned it."""
    if args.fixed:
        drawn = Params("down", FULL_BAND, FALLOFF_EXP, 0.5, FIGURE_COUNT, DENSITY_PULL)
    else:
        drawn = Params(
            direction=str(rng.choice(DIRECTIONS, p=[0.30, 0.15, 0.10, 0.10, 0.25, 0.10])),
            full_band=float(rng.uniform(0.15, 0.55)),
            falloff=float(rng.uniform(1.0, 2.4)),
            flat=float(rng.uniform(0.25, 0.70)),
            figures=int(rng.integers(40, 141)),
            pull=float(rng.uniform(0.20, 0.40)),
        )
    return replace(drawn, direction=args.direction or drawn.direction)


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


def write_png(alpha: np.ndarray, ink, bg, path: str) -> None:
    layer = Image.new("RGBA", alpha.shape[::-1], ink + (255,))
    layer.putalpha(Image.fromarray(alpha, "L"))
    if bg is not None:
        base = Image.new("RGBA", alpha.shape[::-1], bg + (255,))
        layer = Image.alpha_composite(base, layer)
    layer.save(path, "PNG")


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


def check_density(pattern: np.ndarray, ramp: np.ndarray, pull: float) -> None:
    """Assert the figures stayed a perturbation of the base ramp."""
    got = pattern.mean(axis=2)
    assert abs(float(got.mean() - ramp.mean())) < pull, "overall density drifted"
    per_row = abs(got.mean(axis=1) - ramp.mean(axis=1)).max()
    assert per_row < pull + 0.12, f"a row deviates from the ramp by {per_row:.3f}"


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
    parser.add_argument("--width", type=int, default=OUTPUT_WIDTH)
    parser.add_argument("--height", type=int, default=OUTPUT_HEIGHT)
    parser.add_argument("--font-size", type=int, default=FONT_SIZE, help="em size in px")
    parser.add_argument("--style", choices=sorted(STYLES), default=DEFAULT_STYLE)
    parser.add_argument("--font-file", help="measure glyph metrics from this TTF/OTF instead")
    parser.add_argument("--line-height", type=float, help="override line height, in em")
    parser.add_argument("--direction", choices=DIRECTIONS, help="pin the fade direction")
    parser.add_argument("--fixed", action="store_true", help="use the constants, not random looks")
    parser.add_argument("--ink", choices=("white", "black", "both"), default="both")
    parser.add_argument("--bg", type=parse_color, help="opaque background hex, e.g. 101820")
    parser.add_argument("--txt", action="store_true", help="also write Unicode braille text")
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    if args.font_file:
        style = style_from_font(args.font_file, args.line_height)
    else:
        style = STYLES[args.style]
        if args.line_height:
            style = replace(style, line=args.line_height)

    layout = make_layout(style, args.font_size, args.width, args.height)
    check_alignment(layout, style, args.font_size)
    print(
        f"grid {layout.cols}x{layout.rows} cells of {layout.cell_w}x{layout.cell_h}px "
        f"(em {args.font_size}px, dot {style.diameter * args.font_size:.1f}px)"
    )

    inks = {"white": (255, 255, 255), "black": (0, 0, 0)}
    chosen = ("white", "black") if args.ink == "both" else (args.ink,)
    os.makedirs(args.out_dir, exist_ok=True)

    for n in range(args.count):
        seed = args.seed if (args.seed and args.count == 1) else secrets.token_bytes(32).hex()
        rng = np.random.default_rng(np.random.SeedSequence(int.from_bytes(bytes.fromhex(seed), "big")))

        params = random_params(rng, args)
        ramp = fade_ramp(layout, params.direction, params.full_band, params.falloff, params.flat)
        field = figure_field(rng, layout, params.figures, params.pull, args.width, args.height)
        pattern = dot_pattern(rng, ramp, field, params.pull)
        check_density(pattern, ramp, params.pull)
        alpha = render_alpha(pattern, layout, args.width, args.height)

        stem = f"braille-{seed[:8]}"
        print(
            f"[{n + 1}/{args.count}] seed: {seed}\n"
            f"    {params.direction}, band {params.full_band:.2f}, falloff {params.falloff:.2f}, "
            f"{params.figures} figures, pull {params.pull:.2f}"
        )
        for name in chosen:
            path = os.path.join(args.out_dir, f"{stem}-{name}.png")
            write_png(alpha, inks[name], args.bg, path)
            print(f"    wrote {path} {args.width}x{args.height}")
        if args.txt:
            path = os.path.join(args.out_dir, f"{stem}.txt")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(to_text(pattern) + "\n")
            print(f"    wrote {path}")


if __name__ == "__main__":
    main()
