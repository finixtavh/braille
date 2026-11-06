"""Render braille textures using real Unicode braille characters.

This is a fork of ../braille_gen.py. The difference is what a "dot" is:

  * braille_gen.py draws every dot as a disc computed from a distance formula.
    It only *resembles* braille, and a cell's eight dots are independent coin
    flips, so most cells match no real character.

  * This script maps each cell to one of the 256 real braille characters in
    U+2800..U+28FF and renders it with a font, so every cell is a valid
    braille pattern.

The density gradient and the figure curves still work: rather than rejecting
dots, a cell picks the character whose dot count lands closest to the local
density, breaking ties randomly. A dense area therefore shows heavy characters
and a sparse one shows light characters.

FALLBACK ORDER
    BRAILLE_FONTS is tried in order and the first font that actually contains
    all 256 braille glyphs wins. IPAexGothic is listed for reference but is
    skipped: it has no U+28xx coverage and renders braille as empty boxes.
"""

from __future__ import annotations

import argparse
import os
import secrets
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import braille_gen_legacy as bg  # noqa: E402  (shared geometry, figures and checks)

# --- Tunables ---------------------------------------------------------------

# Canvas and gradient. Same meaning as in braille_gen.py.
OUTPUT_WIDTH = 1024
OUTPUT_HEIGHT = 6144
FULL_BAND = 0.4
FALLOFF_EXP = 1.6

# Size of one braille cell in output pixels, as the *width* of the glyph box.
# The font's own braille design decides how the 2x4 dots sit inside it, so this
# is the size of the cell, not of an individual dot.
CELL_SIZE = 20

# Font size in points. PIL renders at this size and the cell is cropped to
# CELL_SIZE afterwards, so the two are independent: the size controls the
# antialiasing quality of the glyph, the crop controls the final geometry.
FONT_SIZE = 32

# Candidate fonts, in order of preference. The first that renders all 256
# braille patterns as distinct, non-.notdef glyphs is used.
#
# IPAexGothic (installed as /usr/share/fonts/OTF/ipaexg.ttf) is deliberately
# first in this list's documentation but is NOT usable: it is a Japanese font
# with no U+28xx coverage, so every braille character renders as .notdef, an
# empty crossed box. Leave it out or the renderer draws rectangles.
BRAILLE_FONTS = (
    "/usr/share/fonts/gnu-free/FreeMono.otf",
    "/usr/share/fonts/TTF/JetBrainsMonoNerdFontMono-Regular.ttf",
    "/usr/share/fonts/noto/NotoSansSymbols2-Regular.ttf",
)

# Ink colour per output file: (filename, rgb).
COLORWAYS = (
    ("braille-white.png", (255, 255, 255)),
    ("braille-black.png", (0, 0, 0)),
)

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out-unicode"
)

# Figure and warp settings are inherited from braille_gen so both scripts share
# one set of distortion controls. See FIGURE_COUNT, DENSITY_PULL, WARP_PX there.


# --- Font selection ---------------------------------------------------------


def font_renders_braille(path: str) -> bool:
    """True when the font draws the braille patterns as real, distinct glyphs.

    Comparing bounding boxes is not enough: a font without braille maps every
    code point to .notdef, which has a non-empty box, so a bbox test alone
    passes a font that draws empty crossed rectangles. IPAexGothic is exactly
    that case, and it is the trap this check exists to catch.

    The test renders every pattern and counts distinct bitmaps. A font with real
    braille coverage yields 256: 255 distinct patterns plus the blank cell.
    Partial coverage yields far fewer, because many code points collapse onto
    the same fallback glyph.
    """
    try:
        font = ImageFont.truetype(path, FONT_SIZE)
    except OSError:
        return False

    # Render into a tile with room to spare, so a glyph is never clipped into
    # looking like a duplicate of another.
    side = FONT_SIZE * 2
    seen = set()
    for code in range(256):
        tile = Image.new("L", (side, side), 0)
        ImageDraw.Draw(tile).text((side // 4, side // 4), chr(0x2800 + code), font=font, fill=255)
        seen.add(tile.tobytes())

    # 256 distinct bitmaps means full coverage. Anything less means the font
    # falls back to .notdef for the patterns it lacks.
    return len(seen) == 256


def load_font() -> ImageFont.FreeTypeFont:
    """Return the first usable braille font, or explain why none is."""
    for path in BRAILLE_FONTS:
        if os.path.exists(path) and font_renders_braille(path):
            return ImageFont.truetype(path, FONT_SIZE)

    raise SystemExit(
        "no usable braille font found. Checked:\n  "
        + "\n  ".join(BRAILLE_FONTS)
        + "\n\nA font qualifies only if it renders all 256 U+28xx patterns as\n"
        "distinct glyphs. IPAexGothic does not: it has no braille coverage."
    )


# --- Braille character selection -------------------------------------------

# How many raised dots each of the 256 characters has. Computed from the
# Unicode definition: the character is U+2800 + sum of dot numbers, where dot
# n contributes bit (n - 1).
POPCOUNT = np.array(
    [bin(code).count("1") for code in range(256)], dtype=np.int8
)

# Characters grouped by dot count, so a density can be matched in one lookup.
# Index is the dot count 0..8; each entry is the array of matching code points.
BY_POPCOUNT = [np.flatnonzero(POPCOUNT == k) for k in range(9)]


def cell_characters(density: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Choose one real braille character per cell for the given density.

    Each cell takes the group of characters whose dot count is nearest the
    density-scaled expectation, then one is drawn at random within that group.
    This is what keeps the texture a gradient while every cell stays a valid
    character: density 0 leans on the empty and single-dot patterns, density 1
    on the full and seven-dot ones.

    Returns a uint16 array of code points, shape (rows, cols).
    """
    rows, cols = density.shape
    # Expected dot count out of eight, then the nearest count that exists.
    wanted = np.clip(np.rint(density * 8.0), 0, 8).astype(np.int8)

    out = np.empty(density.size, dtype=np.uint16)
    for count in range(9):
        # Flatten the mask to match the flat output buffer.
        mask = (wanted == count).reshape(-1)
        n = int(mask.sum())
        if n == 0:
            continue
        members = BY_POPCOUNT[count]
        out[mask] = members[rng.integers(0, members.size, n)]

    return out.reshape(rows, cols)


# --- Rendering --------------------------------------------------------------


def render_characters(chars: np.ndarray, font: ImageFont.FreeTypeFont) -> Image.Image:
    """Draw a grid of braille characters into an alpha buffer.

    Each cell is rendered at FONT_SIZE into an offscreen tile, then that tile is
    resized to a CELL_W x CELL_H box and composited. Rendering per cell rather
    than as one long string means each cell is positioned by the grid, not by
    the font's advance width, so a font with odd metrics still produces a
    regular grid.
    """
    rows, cols = chars.shape
    canvas = Image.new("L", (OUTPUT_WIDTH, OUTPUT_HEIGHT), 0)

    # A braille cell is twice as tall as it is wide (4 dot rows, 2 dot columns),
    # so the tile is CELL_W x CELL_H, not square. Spacing rows by CELL_W instead
    # of CELL_H would overlap every cell and clip the bottom of the canvas.
    cell_w = bg.CELL_W
    cell_h = bg.CELL_H

    # Render the 256 glyphs once each, then reuse the tiles. At 256 tiles this
    # is far cheaper than drawing a glyph per cell.
    #
    # Each glyph is cropped to its own ink box before being fitted to the cell.
    # A font's braille bbox is tall and narrow (38x54 at FONT_SIZE here), so
    # resizing the whole square tile to a 2:1 cell would squash it. Cropping to
    # the ink and fitting that to the cell keeps every cell the same size, which
    # is what makes the grid read as a grid. The blank cell has no ink, so it
    # gets an empty tile.
    tiles = {}
    for code in range(256):
        # Name this `scratch`, not `canvas`: reusing the output canvas name here
        # would rebind it and the caller would get a single glyph's size back.
        scratch = Image.new("L", (FONT_SIZE * 2, FONT_SIZE * 2), 0)
        draw = ImageDraw.Draw(scratch)
        draw.text((FONT_SIZE // 2, FONT_SIZE // 2), chr(0x2800 + code), font=font, fill=255)
        box = scratch.getbbox()

        if box is None:
            tiles[code] = Image.new("L", (cell_w, cell_h), 0)
            continue

        glyph = scratch.crop(box)
        # Fit the glyph inside the cell, preserving its aspect ratio, so the
        # dots keep their shape instead of being stretched to fill.
        scale = min(cell_w / glyph.width, cell_h / glyph.height)
        new_size = (max(1, round(glyph.width * scale)), max(1, round(glyph.height * scale)))
        glyph = glyph.resize(new_size, Image.LANCZOS)

        tile = Image.new("L", (cell_w, cell_h), 0)
        tile.paste(glyph, ((cell_w - glyph.width) // 2, (cell_h - glyph.height) // 2))
        tiles[code] = tile

    cells_x, cells_y = bg.cell_grid()
    for cell_row in range(cells_y):
        for cell_col in range(cells_x):
            code = int(chars[cell_row, cell_col])
            # U+2800 is the blank cell; nothing to draw for it.
            if code == 0:
                continue
            canvas.paste(
                tiles[code],
                (cell_col * cell_w, cell_row * cell_h),
                tiles[code],
            )

    return canvas


def write_colorway(alpha: Image.Image, rgb: tuple[int, int, int], path: str) -> None:
    """Composite the character mask over a solid colour and save it."""
    image = Image.new("RGBA", (OUTPUT_WIDTH, OUTPUT_HEIGHT), rgb + (255,))
    image.putalpha(alpha)
    image.save(path, "PNG")


# --- Verification -----------------------------------------------------------


def check_unicode(chars: np.ndarray, font: ImageFont.FreeTypeFont) -> None:
    """Assert every cell is a real character and the gradient still reads.

    The essential check here is that no cell uses a dot layout the font lacks:
    all codes must fall in U+2800..U+28FF, and the font must render each one.
    """
    assert chars.min() >= 0 and chars.max() <= 255, (
        f"cell code out of the braille range: {chars.min()}..{chars.max()}"
    )

    # The blank cell (U+2800) is valid but must not dominate the dense band.
    used = np.unique(chars)
    assert used.size > 8, f"only {used.size} distinct characters used"

    # Every character actually drawn must render to something.
    for code in used:
        if code == 0:
            continue
        assert font.getmask(chr(0x2800 + int(code))).size[0] > 0, (
            f"font renders nothing for U+{0x2800 + int(code):04X}"
        )

    # Density gradient: the top band must use heavier characters than the
    # bottom, which is the whole point of mapping density to dot count.
    popcount = POPCOUNT[chars]
    rows = bg.cell_grid()[1]
    top = popcount[: max(1, rows // 20)].mean()
    bottom = popcount[-max(1, rows // 20) :].mean()
    assert top > bottom, f"gradient inverted: top {top:.2f} vs bottom {bottom:.2f}"
    assert top >= 7.0, f"top band is not dense: {top:.2f} dots per cell of 8"
    assert bottom <= 2.0, f"bottom band is not sparse: {bottom:.2f} dots per cell"


# --- Entry point ------------------------------------------------------------


def cell_density(field: np.ndarray | None) -> np.ndarray:
    """Per-cell density from the vertical ramp plus the figure field.

    Mirrors braille_gen.dot_pattern: the ramp is constant down the top band and
    fades below it, and the figure field is weighted by the ramp so figures fade
    out with the gradient instead of refilling emptied rows.
    """
    ramp = bg.row_density()[:, None]
    cols, rows = bg.cell_grid()
    if field is None:
        # The ramp is (rows, 1); broadcast it to a full cell grid.
        return np.repeat(ramp, cols, axis=1)

    per_cell = field.reshape(rows, bg.CELL_DOTS_Y, cols, bg.CELL_DOTS_X).mean(axis=(1, 3))
    weight = np.clip(ramp / max(bg.DENSITY_PULL, 1e-6), 0.0, 1.0)
    # ramp is (rows, 1) and per_cell is (rows, cols), so the sum already spans
    # the grid. Do not repeat it again: that would tile each column cols times.
    return np.clip(ramp + per_cell * weight, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", help="hex token; omit to draw a fresh one")
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--font", default=None, help="override the font path")
    args = parser.parse_args()

    if args.font:
        global BRAILLE_FONTS
        BRAILLE_FONTS = (args.font,)

    # The parent module derives its cell geometry from BRAILLE_SIZE, so point it
    # at CELL_SIZE here to keep one definition of the grid.
    bg.BRAILLE_SIZE = CELL_SIZE
    bg.CELL_W = CELL_SIZE
    bg.CELL_H = CELL_SIZE * bg.CELL_DOTS_Y // bg.CELL_DOTS_X
    bg.DOT_PITCH = bg.CELL_W / bg.CELL_DOTS_X

    token = secrets.token_bytes(32)
    seed = args.seed or token.hex()
    if args.seed is None:
        print(f"seed: {seed}")

    rng = np.random.default_rng(
        np.random.SeedSequence(int.from_bytes(bytes.fromhex(seed), "big"))
    )

    font = load_font()
    print(f"font: {font.path}")

    bg.check_geometry()
    bg.check_dots()

    field = bg.figure_field(rng, bg.grid_shape())
    bg.check_figures(field)

    density = cell_density(field)
    chars = cell_characters(density, rng)
    check_unicode(chars, font)

    alpha = render_characters(chars, font)

    os.makedirs(args.out_dir, exist_ok=True)
    for filename, rgb in COLORWAYS:
        path = os.path.join(args.out_dir, filename)
        write_colorway(alpha, rgb, path)
        print(f"wrote {path}  {OUTPUT_WIDTH}x{OUTPUT_HEIGHT}")

    counts = POPCOUNT[chars]
    print(
        f"cells {chars.shape[1]}x{chars.shape[0]}, "
        f"{np.unique(chars).size} distinct characters, "
        f"mean {counts.mean():.2f} dots per cell"
    )


if __name__ == "__main__":
    main()
