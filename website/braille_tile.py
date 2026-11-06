"""Generate a seamless braille background tile for web use.

The full-size render in ../braille_gen.py is a 4096x2160 one-off image: roughly
550 KB, because braille dots are high-frequency noise and PNG cannot compress
them. The same texture as a small repeating tile is ~9 KB, because the encoder
exploits the repetition. Browsers tile it for free, so one small file replaces a
stack of full-size backgrounds.

Seamlessness has two requirements, both met here:

  * TILE_SIZE must be an exact multiple of CELL_W and CELL_H, so the braille
    grid itself lines up across the wrap.
  * The figure field and warp noise are evaluated with toroidal distance, so a
    curve or noise feature leaving the right edge re-enters on the left.

Because the tile repeats, it carries no top-to-bottom gradient: a gradient would
band at every repeat. TONE_PULL and WARP_PX set how much texture it has, and the
CSS snippet printed at the end applies the gradient and colour in the browser
instead, where it costs no bytes.
"""

from __future__ import annotations

import argparse
import os
import secrets
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import braille_gen_legacy as bg  # noqa: E402

# --- Tunables ---------------------------------------------------------------

# Tile edge in output pixels. Must be a whole multiple of both CELL_W and
# CELL_H or a visible seam appears. bg.CELL_W is 12 at the default
# BRAILLE_SIZE=11, so 480 (40x20 cells) divides cleanly and 512 does not.
TILE_SIZE = 480

# Overall dot density, held near constant because a repeating tile cannot carry
# a gradient. Lower is sparser and compresses better. Sparse enough that the
# individual braille cells stay legible rather than merging into static.
DENSITY = 0.34

# Peak density change from the figure curves. Expressed as an absolute value,
# not a fraction of DENSITY, so the figures stay subtle at any base density: a
# large pull on a sparse tile drives whole cells to empty and reads as a bald
# patch once the tile repeats.
TONE_PULL = 0.14

# Number of random curves, and their length range in output pixels. Tiles are
# small, so figures are scaled down with the tile to keep the same look.
FIGURE_COUNT = 14
FIGURE_MIN_PX = 40
FIGURE_MAX_PX = 170

# Maximum dot displacement from the grid warp, in output pixels.
WARP_PX = 1.5

# Scale of the smooth noise driving the warp, in output pixels.
WARP_SCALE_PX = 150.0

# Where the tile is written: the sibling site repo, which lives beside this one
# in ~/git. Override with --out-dir. Nothing is written outside this path
# without it being named explicitly.
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "finixtavh-sitio",
    "img",
)

OUT_NAME = "braille-tile.png"


# --- Geometry ---------------------------------------------------------------


def tile_cells() -> tuple[int, int]:
    """Return the (columns, rows) of cells in one tile, asserting a clean wrap."""
    cells_x = TILE_SIZE / bg.CELL_W
    cells_y = TILE_SIZE / bg.CELL_H
    if cells_x != int(cells_x) or cells_y != int(cells_y):
        raise ValueError(
            f"TILE_SIZE={TILE_SIZE} is not a whole multiple of the cell "
            f"({bg.CELL_W}x{bg.CELL_H}). Choose a multiple, e.g. 480."
        )
    return int(cells_x), int(cells_y)


def tile_shape() -> tuple[int, int]:
    """Return the tile's dot-grid size as (rows, cols)."""
    cols, rows = tile_cells()
    return rows * bg.CELL_DOTS_Y, cols * bg.CELL_DOTS_X


# --- Wrapped figure field ---------------------------------------------------


def wrapped_figure_field(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    """Sum soft ridges from curves that wrap across the tile edges.

    Distance is measured on a torus: the shortest path from a dot to a curve
    point may cross any combination of the tile's edges, so a curve leaving the
    right side continues on the left. Without this the tile has a hard seam.
    """
    rows, cols = shape
    ys, xs = np.mgrid[0:rows, 0:cols].astype(np.float32)
    px = xs * bg.DOT_PITCH + bg.DOT_PITCH / 2
    py = ys * bg.DOT_PITCH + bg.DOT_PITCH / 2

    span_x = cols * bg.DOT_PITCH
    span_y = rows * bg.DOT_PITCH
    half_x, half_y = span_x / 2.0, span_y / 2.0

    field = np.zeros(shape, dtype=np.float32)
    for _ in range(FIGURE_COUNT):
        pts, width, sign = tile_curve(rng, span_x, span_y)

        d2 = np.full(shape, np.inf, dtype=np.float32)
        band = max(1, cols // 4)
        for start in range(0, cols, band):
            stop = min(start + band, cols)
            dx = px[:, start:stop, None] - pts[None, None, :, 0]
            dy = py[:, start:stop, None] - pts[None, None, :, 1]
            # Wrap into [-half, half] so opposite edges become neighbours.
            dx -= span_x * np.round(dx / span_x)
            dy -= span_y * np.round(dy / span_y)
            d2[:, start:stop] = np.min(dx * dx + dy * dy, axis=2)

        field += sign * np.exp(-d2 / (2.0 * width**2)).astype(np.float32)

    np.clip(field, -TONE_PULL, TONE_PULL, out=field)
    return field


def tile_curve(
    rng: np.random.Generator, span_x: float, span_y: float
) -> tuple[np.ndarray, float, float]:
    """Return (points, width, sign) for one curve sized for a repeating tile."""
    size = float(rng.uniform(FIGURE_MIN_PX, FIGURE_MAX_PX))
    cx, cy = rng.uniform(0.0, span_x), rng.uniform(0.0, span_y)
    angle = rng.uniform(0.0, 2.0 * np.pi)
    along = np.array([np.cos(angle), np.sin(angle)])
    across = np.array([-np.sin(angle), np.sin(angle)])

    kind = rng.integers(0, 3)
    if kind == 0:
        bend = size * rng.uniform(0.4, 0.9) * rng.choice((-1.0, 1.0))
        p0 = np.array([cx, cy]) - along * size
        p1 = np.array([cx, cy]) + along * size
        c0 = p0 + along * size * 0.5 + across * bend
        c1 = p1 - along * size * 0.5 + across * bend
    elif kind == 1:
        p0 = np.array([cx, cy])
        p1 = p0 + along * size
        c0 = p0 + across * size * 0.6 + along * size * 0.2
        c1 = p1 - across * size * 0.6 - along * size * 0.2
    else:
        swing = size * rng.uniform(0.5, 1.0) * rng.choice((-1.0, 1.0))
        p0 = np.array([cx, cy]) - along * size
        p1 = np.array([cx, cy]) + along * size
        c0 = p0 + along * size * 0.3 + across * swing
        c1 = p1 - along * size * 0.3 - across * swing

    t = np.linspace(0.0, 1.0, 64)[:, None]
    inv = 1.0 - t
    pts = inv**3 * p0 + 3 * inv**2 * t * c0 + 3 * inv * t**2 * c1 + t**3 * p1

    width = size * rng.uniform(0.10, 0.22)
    return pts.astype(np.float32), width, float(rng.choice((-1.0, 1.0)))


def wrapped_warp_offsets(
    rng: np.random.Generator, shape: tuple[int, int]
) -> np.ndarray:
    """Per-dot (dy, dx) displacement that also wraps across the tile edges.

    The noise is generated on a torus by tiling a single period, so the field is
    continuous across the wrap and the grid bends without a seam.
    """
    rows, cols = shape
    scale = WARP_SCALE_PX
    px_per = bg.DOT_PITCH

    # Period of the lattice in cells, chosen so it divides the tile exactly.
    per_c = max(2, int(round(scale / px_per)))
    while cols % per_c:
        per_c += 1
    per_r = max(2, int(round(scale / px_per)))
    while rows % per_r:
        per_r += 1

    field_y = tile_noise(rng, rows, cols, per_r, per_c)
    field_x = tile_noise(rng, rows, cols, per_r, per_c)

    limit = WARP_PX / px_per
    mag = np.hypot(field_y, field_x)
    scale_down = np.where(mag > limit, limit / np.maximum(mag, 1e-6), 1.0)
    return np.stack([field_y * scale_down, field_x * scale_down])


def tile_noise(
    rng: np.random.Generator, rows: int, cols: int, per_r: int, per_c: int
) -> np.ndarray:
    """Smooth noise in [-1, 1] built from one period of a random lattice."""
    lattice = rng.uniform(-1.0, 1.0, (per_r + 1, per_c + 1)).astype(np.float32)
    # Make the extra row/column a copy of the first, so the period closes.
    lattice[-1, :] = lattice[0, :]
    lattice[:, -1] = lattice[:, 0]

    img = Image.fromarray(lattice, "F").resize((cols, rows), Image.BICUBIC)
    return np.clip(np.asarray(img, dtype=np.float32), -1.0, 1.0)


# --- Pattern and render -----------------------------------------------------


def tile_pattern(rng: np.random.Generator, field: np.ndarray) -> np.ndarray:
    """Boolean dot mask for one tile, with figure-driven density variation."""
    rows, cols = shape_of(field)
    per_cell = field.reshape(rows // bg.CELL_DOTS_Y, bg.CELL_DOTS_Y, cols // bg.CELL_DOTS_X, bg.CELL_DOTS_X).mean(axis=(1, 3))
    density = np.clip(DENSITY + per_cell, 0.0, 1.0)

    cell_rows, cell_cols = rows // bg.CELL_DOTS_Y, cols // bg.CELL_DOTS_X
    samples = rng.random((cell_rows, cell_cols, bg.CELL_DOTS_X * bg.CELL_DOTS_Y))
    filled = samples < density[..., None]

    slots = np.arange(bg.CELL_DOTS_X * bg.CELL_DOTS_Y)
    slot_row = slots % bg.CELL_DOTS_Y
    slot_col = slots // bg.CELL_DOTS_Y

    dot_row = np.arange(cell_rows)[:, None, None] * bg.CELL_DOTS_Y + slot_row
    dot_col = np.arange(cell_cols)[None, :, None] * bg.CELL_DOTS_X + slot_col

    grid = np.zeros((rows, cols), dtype=bool)
    grid[dot_row, dot_col] = filled
    return grid


def shape_of(field: np.ndarray) -> tuple[int, int]:
    return field.shape


def render_tile(pattern: np.ndarray, offsets: np.ndarray) -> Image.Image:
    """Stamp dots into a coverage buffer and downsample to the tile size.

    Dots are stamped at their warped position rather than resampling the finished
    image, so warping smears the grid without blurring or doubling dots.
    """
    scale = bg.render_scale()
    size = int(round(bg.DOT_PITCH * scale))
    dot_rows, dot_cols = pattern.shape
    kernel = bg.dot_kernel()

    shift_y = np.rint(offsets[0] * bg.DOT_PITCH * size).astype(int)
    shift_x = np.rint(offsets[1] * bg.DOT_PITCH * size).astype(int)

    margin = int(max(abs(shift_y).max(), abs(shift_x).max())) + 1
    strips = np.zeros((dot_rows * size + 2 * margin, dot_cols * size + 2 * margin), np.uint8)

    base_y = np.repeat(np.arange(dot_rows) * size, dot_cols).reshape(dot_rows, dot_cols)
    base_x = np.tile(np.arange(dot_cols) * size, (dot_rows, 1))

    # Wrap into the cropped coordinate space first, then offset by the margin.
    # Wrapping after the margin would be wrong: the margin enlarges the
    # coordinate range, so a wrapped dot lands outside the region the crop
    # keeps and the last row of the tile comes out empty.
    span_y = dot_rows * size
    span_x = dot_cols * size
    origin_y = (base_y + shift_y) % span_y + margin
    origin_x = (base_x + shift_x) % span_x + margin

    filled_y, filled_x = np.nonzero(pattern)
    if filled_y.size:
        for ky, kx in np.argwhere(kernel):
            # Fold the kernel radius back inside the grid when it pushes a dot
            # past the far edge.
            sy = (origin_y[filled_y, filled_x] - margin + ky) % span_y + margin
            sx = (origin_x[filled_y, filled_x] - margin + kx) % span_x + margin
            np.maximum.at(strips, (sy, sx), np.uint8(255))

    # Drop the margin: the wrap means nothing meaningful was written out there.
    cropped = strips[margin : margin + dot_rows * size, margin : margin + dot_cols * size]
    return Image.fromarray(cropped, "L").resize((TILE_SIZE, TILE_SIZE), Image.LANCZOS)


def to_rgba(alpha: Image.Image, rgb: tuple[int, int, int]) -> Image.Image:
    """Composite the coverage buffer over a solid ink colour."""
    image = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), rgb + (255,))
    image.putalpha(alpha)
    return image


def save_png(image: Image.Image, path: str) -> int:
    """Write the tile and return its size in bytes.

    No palette reduction: the tile is one ink colour with a varying alpha
    channel, so the colour data is constant and quantising it saves nothing.
    Only the alpha channel carries detail, and dithered alpha is close to
    incompressible either way.
    """
    image.save(path, "PNG", optimize=True)
    return os.path.getsize(path)


# --- Verification -----------------------------------------------------------


def check_tile(alpha: np.ndarray) -> None:
    """Assert the tile is actually seamless and free of bald patches."""
    # The dots must be round here too, not just in the full-size render. This
    # reuses the parent module's measurement so both scripts agree on what
    # counts as circular.
    bg.check_dots()

    # Normalise to 0..1 so the thresholds read as density fractions.
    norm = alpha / 255.0

    # A seam shows as a discontinuity between opposite edges. Compare bands
    # rather than single rows: one row holds only ~80 dots, so its mean carries
    # a sampling error of ~0.05 on its own and would fail a tight threshold
    # even on a perfectly seamless tile. Even over a four-row band the residual
    # is ~0.08, so the bound is set above that and well below the 0.2+ a real
    # seam produced before the stamp coordinates were wrapped.
    band = bg.CELL_H
    edge_gap = abs(float(norm[:band].mean()) - float(norm[-band:].mean()))
    assert edge_gap < 0.12, f"top/bottom edge density differs by {edge_gap:.3f}"
    left_gap = abs(float(norm[:, :band].mean()) - float(norm[:, -band:].mean()))
    assert left_gap < 0.12, f"left/right edge density differs by {left_gap:.3f}"

    assert norm.max() > 0.98, f"no opaque coverage: {norm.max()}"
    assert norm.min() == 0.0, f"no empty coverage: {norm.min()}"

    # No perceptible patch should be far emptier than the tile average. The
    # window is 120px rather than a few cells: at cell scale the minimum is
    # dominated by ordinary dot-to-dot noise (a single absent cell is ~40% of a
    # small window) and would fail regardless of the figures.
    block = 120
    rows = norm.shape[0] // block
    cols = norm.shape[1] // block
    windows = norm[: rows * block, : cols * block].reshape(rows, block, cols, block).mean(axis=(1, 3))
    mean = norm.mean()
    assert windows.min() > mean * 0.5, f"bald patch: {windows.min():.3f} vs {mean:.3f}"


# --- Entry point ------------------------------------------------------------


CSS_SNIPPET = """
body {{
    background-color: #13131e;
    background-image:
        /* Gradient lives here, not in the image, so it costs no bytes and
           never bands at a tile repeat. */
        linear-gradient(rgba(19, 19, 45, 0.7), rgba(9, 9, 15, 0.85)),
        url("../img/{name}");
    background-size: auto, {size}px {size}px;   /* no-repeat on the gradient */
    background-repeat: no-repeat, repeat;
    background-attachment: local, local;
}}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", help="hex token; omit to draw a fresh one")
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    token = secrets.token_bytes(32)
    seed = args.seed or token.hex()
    if args.seed is None:
        print(f"seed: {seed}")

    rng = np.random.default_rng(
        np.random.SeedSequence(int.from_bytes(bytes.fromhex(seed), "big"))
    )

    cells_x, cells_y = tile_cells()
    print(f"tile {TILE_SIZE}x{TILE_SIZE} = {cells_x}x{cells_y} cells, cell {bg.CELL_W}x{bg.CELL_H}")

    shape = tile_shape()
    field = wrapped_figure_field(rng, shape)
    offsets = wrapped_warp_offsets(rng, shape)
    pattern = tile_pattern(rng, field)
    alpha = render_tile(pattern, offsets)

    check_tile(np.asarray(alpha, dtype=np.float32))

    rgba = to_rgba(alpha, (255, 255, 255))
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, OUT_NAME)
    size_kb = save_png(rgba, path) / 1024
    print(f"wrote {path}  {TILE_SIZE}x{TILE_SIZE}  {size_kb:.1f} KB")

    print("\nCSS:")
    print(CSS_SNIPPET.format(name=OUT_NAME, size=TILE_SIZE))


if __name__ == "__main__":
    main()
