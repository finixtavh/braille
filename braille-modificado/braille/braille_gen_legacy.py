"""Render gradient-damped random braille textures as RGBA PNGs.

The same random dot pattern is written twice, once in white and once in black,
so the two files are exact colorways of each other. Every pixel not covered by a
braille dot is fully transparent.
"""

from __future__ import annotations

import argparse
import os
import secrets

import numpy as np
from PIL import Image

# Braille cell: two columns of four dots.
CELL_DOTS_X = 2
CELL_DOTS_Y = 4

# --- Tunables ---------------------------------------------------------------

# Output image size in pixels.
OUTPUT_WIDTH = 2048
OUTPUT_HEIGHT =  10240

# Size of one braille cell, as its width in output pixels. Cells are always
# twice as tall as they are wide (2 columns of 4 dots at an even pitch), so
# this single number sets how large the braille reads. Bigger = bolder and
# chunkier, smaller = finer texture.
#
# Must leave DOT_PITCH at or above MIN_DOT_PITCH. Below that a circle cannot be
# rasterised: at a 6px pitch the dot's middle rows come out uniform across the
# full width, so it reads as a square with clipped corners rather than a circle,
# and supersampling does not help because the limit is in the final pixels.
BRAILLE_SIZE = 20

# Distance in output pixels between neighbouring dot centres, derived from the
# cell size so the 2x4 dot layout stays evenly spaced.
DOT_PITCH = BRAILLE_SIZE / CELL_DOTS_X

# Fraction of the canvas height that stays completely solid before the fade
# begins. Set to 0.0 for a gradient that starts declining at the very first row.
FULL_BAND = 0.4

# Falloff exponent applied below FULL_BAND. 1.0 is linear, > 1 keeps the image
# dense for longer then drops off a cliff near the bottom, < 1 thins out early.
FALLOFF_EXP = 1.6

# Smallest dot pitch, in output pixels, at which a disc still reads as a circle.
# Measured by counting uniform inner rows in the rasterised dot: 6px and 8px
# both give two, so the dot has square sides; 10px and up give none, so the disc
# has a real curve. Supersampling cannot rescue a pitch below this, because the
# limit is in the final pixels, not in how the disc was drawn.
MIN_DOT_PITCH = 10

# Render at least N times the output resolution, then downsample for antialiased
# edges. The actual scale is raised to the smallest multiple that makes
# DOT_PITCH a whole number of pixels, so the downscale is always an exact integer
# ratio and never softens the dots.
#
# It is also raised to MIN_DOT_BLOCK's requirement, which buys the disc enough
# pixels to show a circular profile before downsampling.
SUPERSAMPLE = 1

# Smallest dot block size, in pixels, at which the disc is round enough to
# survive downsampling to the output pitch. At a 10px pitch a scale of 2 gives
# a 20px block, enough for a clean curve at the 10x2 downscale.
MIN_DOT_BLOCK = 20

# Dot diameter as a fraction of DOT_PITCH. Below 1.0 leaves a visible gap
# between neighbouring dots. Very close to 1.0 merges neighbouring dots into a
# lattice, so a little space is kept.
DOT_FILL = 0.9

# --- Distortion figures -----------------------------------------------------

# Random curves scattered over the canvas, each of a random kind (arc, coil or
# S-curve). They modulate dot density in soft ridges rather than carving
# shapes, so no region is ever left empty.
FIGURE_COUNT = 90

# Feature size range in output pixels. Keep these small relative to the canvas
# for a fine, local distortion; large values read as sweeping blotches.
FIGURE_MIN_PX = 200
FIGURE_MAX_PX = 800

# Peak density change a figure can apply, as a fraction. Capped well below 1.0
# so figures can never punch holes in a dense area.
DENSITY_PULL = 0.30

# Maximum dot displacement from the grid warp, in output pixels. Small values
# smear the braille without hiding the cell structure.
WARP_PX = 3.0

# Scale of the smooth noise driving the warp, in output pixels. Larger values
# bend the grid over longer distances.
WARP_SCALE_PX = 900.0

# (filename, rgb) pairs written from the same random pattern.
COLORWAYS = (
    ("braille-white.png", (255, 255, 255)),
    ("braille-black.png", (0, 0, 0)),
)

OUT_DIR = "."


# --- Geometry ---------------------------------------------------------------

# A cell is measured in whole output pixels, so an odd BRAILLE_SIZE rounds the
# cell to the nearest even width and the pitch follows from that.
CELL_W = int(round(BRAILLE_SIZE / CELL_DOTS_X) * CELL_DOTS_X)
CELL_H = CELL_W * CELL_DOTS_Y // CELL_DOTS_X
DOT_PITCH = CELL_W / CELL_DOTS_X


def render_scale() -> int:
    """Smallest scale that makes the dot pitch whole *and* the disc round.

    Two constraints, both integer multiples of SUPERSAMPLE:

      * DOT_PITCH * scale must be a whole number of pixels. An odd BRAILLE_SIZE
        gives a fractional pitch (11px cells put dots 5.5px apart), and
        rendering fractional blocks loses dot area to the rescaler.
      * DOT_PITCH * scale must be at least MIN_DOT_BLOCK, so the disc has enough
        pixels across to show a circular profile instead of a rounded square.
    """
    for scale in range(SUPERSAMPLE, SUPERSAMPLE * 64 + 1):
        pitch = DOT_PITCH * scale
        if pitch.is_integer() and pitch >= MIN_DOT_BLOCK:
            return scale
    raise ValueError(
        f"no scale in range makes DOT_PITCH={DOT_PITCH} whole and >= {MIN_DOT_BLOCK}px"
    )


def cell_grid() -> tuple[int, int]:
    """Return the (columns, rows) of whole braille cells that fit the canvas."""
    return int(OUTPUT_WIDTH // CELL_W), int(OUTPUT_HEIGHT // CELL_H)


def grid_shape() -> tuple[int, int]:
    """Return the dot-grid size as (rows, cols), matching the stamped pattern."""
    cols, rows = cell_grid()
    return rows * CELL_DOTS_Y, cols * CELL_DOTS_X


def check_geometry() -> None:
    """Report any output pixels left over, which stay fully transparent."""
    used_w, used_h = cell_grid()[0] * CELL_W, cell_grid()[1] * CELL_H
    if used_w != OUTPUT_WIDTH or used_h != OUTPUT_HEIGHT:
        print(
            f"note: braille grid covers {used_w}x{used_h} of "
            f"{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}; the remainder is transparent"
        )


def row_density() -> np.ndarray:
    """Per-cell-row fill probability, 1.0 across FULL_BAND then fading to 0."""
    _, rows = cell_grid()
    y = np.arange(rows) / rows

    fade = np.clip((1.0 - y) / (1.0 - FULL_BAND), 0.0, 1.0)
    return np.where(y <= FULL_BAND, 1.0, fade**FALLOFF_EXP)


def figure_curve(rng: np.random.Generator) -> tuple[np.ndarray, float, float]:
    """Return (points, width, sign) for one random distortion curve.

    Three kinds are drawn: a wide arc, a tight coil, or an elongated S-curve.
    Each is a cubic sampled densely enough that the distance field below sees a
    smooth ridge rather than a polyline.
    """
    size = float(rng.uniform(FIGURE_MIN_PX, FIGURE_MAX_PX))
    cx, cy = rng.uniform(0.0, OUTPUT_WIDTH), rng.uniform(0.0, OUTPUT_HEIGHT)
    angle = rng.uniform(0.0, 2.0 * np.pi)
    along = np.array([np.cos(angle), np.sin(angle)])
    across = np.array([-np.sin(angle), np.cos(angle)])

    kind = rng.integers(0, 3)
    if kind == 0:
        # Arc: a single bow, offset sideways to one side.
        bend = size * rng.uniform(0.4, 0.9) * rng.choice((-1.0, 1.0))
        p0 = np.array([cx, cy]) - along * size
        p1 = np.array([cx, cy]) + along * size
        c0 = p0 + along * size * 0.5 + across * bend
        c1 = p1 - along * size * 0.5 + across * bend
    elif kind == 1:
        # Coil: a curl that wraps roughly a turn and a half.
        p0 = np.array([cx, cy])
        p1 = p0 + along * size
        c0 = p0 + np.array([across[0], across[1]]) * size * 0.6 + along * size * 0.2
        c1 = p1 - np.array([across[0], across[1]]) * size * 0.6 - along * size * 0.2
    else:
        # S-curve: a long stroke that reverses sideways once.
        p0 = np.array([cx, cy]) - along * size
        p1 = np.array([cx, cy]) + along * size
        swing = size * rng.uniform(0.5, 1.0) * rng.choice((-1.0, 1.0))
        c0 = p0 + along * size * 0.3 + across * swing
        c1 = p1 - along * size * 0.3 - across * swing

    t = np.linspace(0.0, 1.0, 96)[:, None]
    inv = 1.0 - t
    pts = (
        inv**3 * p0
        + 3 * inv**2 * t * c0
        + 3 * inv * t**2 * c1
        + t**3 * p1
    )

    # The ridge is much narrower than the curve is long, so each figure reads as
    # a fine band. A width near the curve size would blanket the canvas and act
    # as one global density shift instead of local texture.
    width = size * rng.uniform(0.04, 0.10)
    return pts, width, float(rng.choice((-1.0, 1.0)))


def figure_field(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    """Sum a soft ridge per figure into a density offset bounded by DENSITY_PULL.

    The ridge is a Gaussian falloff in distance to the curve, so figures read as
    soft bands of thicker and thinner braille rather than as hard outlines.
    Because every contribution is clamped to DENSITY_PULL, no figure can empty a
    region that the base ramp wants filled.
    """
    rows, cols = shape
    ys, xs = np.mgrid[0:rows, 0:cols].astype(np.float32)
    # Dot-grid coordinates to output pixels.
    px = xs * DOT_PITCH + DOT_PITCH / 2
    py = ys * DOT_PITCH + DOT_PITCH / 2

    field = np.zeros(shape, dtype=np.float32)
    for _ in range(FIGURE_COUNT):
        pts, width, sign = figure_curve(rng)
        qx = pts[:, 0]
        qy = pts[:, 1]

        # Squared distance to the nearest curve sample. Split into column bands
        # so the broadcast stays within a sane memory footprint.
        d2 = np.full(shape, np.inf, dtype=np.float32)
        band = max(1, cols // 8)
        for start in range(0, cols, band):
            stop = min(start + band, cols)
            dx = px[:, start:stop, None] - qx
            dy = py[:, start:stop, None] - qy
            d2[:, start:stop] = np.min(dx * dx + dy * dy, axis=2)

        ridge = np.exp(-d2 / (2.0 * width**2)).astype(np.float32)
        field += sign * ridge

    np.clip(field, -DENSITY_PULL, DENSITY_PULL, out=field)
    return field


def smooth_noise(rng: np.random.Generator, shape: tuple[int, int], scale: float) -> np.ndarray:
    """Smooth 2D noise in [-1, 1] from a coarse random lattice, smoothly upsampled."""
    rows, cols = shape
    coarse_r = max(2, int(rows * DOT_PITCH / scale) + 2)
    coarse_c = max(2, int(cols * DOT_PITCH / scale) + 2)

    lattice = rng.uniform(-1.0, 1.0, (coarse_r, coarse_c)).astype(np.float32)
    img = Image.fromarray(lattice, "F").resize((cols, rows), Image.BICUBIC)
    noise = np.asarray(img, dtype=np.float32)

    return np.clip(noise, -1.0, 1.0)


def warp_offsets(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    """Per-dot (dy, dx) displacement in dot-grid units, bounded by WARP_PX.

    Two noise fields are combined into a divergence-free-ish flow so the grid
    bends and smears like heat haze rather than sliding uniformly. Offsets are
    applied when each dot is stamped, so dots never blur or double.
    """
    field_y = smooth_noise(rng, shape, WARP_SCALE_PX)
    field_x = smooth_noise(rng, shape, WARP_SCALE_PX * 1.7)

    limit = WARP_PX / DOT_PITCH
    mag = np.hypot(field_y, field_x)
    over = mag > limit
    scale = np.where(over, limit / np.maximum(mag, 1e-6), 1.0)

    return np.stack([field_y * scale, field_x * scale])


def dot_pattern(rng: np.random.Generator, field: np.ndarray | None = None) -> np.ndarray:
    """Return the filled-dot mask at dot-grid resolution, shape (y, x).

    Dots are numbered the braille way: down the left column first, then the
    right, so the texture reads as braille rather than as random confetti.
    `field` is an optional per-dot density offset from the figure curves.
    """
    cols, rows = cell_grid()
    ramp = row_density()[:, None]
    if field is None:
        density = ramp
    else:
        # field arrives at dot resolution (rows*4, cols*2); reshape to cells.
        per_cell = field.reshape(rows, CELL_DOTS_Y, cols, CELL_DOTS_X).mean(axis=(1, 3))
        # Fade the figures out along with the ramp. Otherwise a positive figure
        # would refill rows the gradient has already emptied, and the fade would
        # stop reading as a fade.
        weight = np.clip(ramp / max(DENSITY_PULL, 1e-6), 0.0, 1.0)
        density = np.clip(ramp + per_cell * weight, 0.0, 1.0)

    samples = rng.random((rows, cols, CELL_DOTS_X * CELL_DOTS_Y))
    filled = samples < density[..., None]

    # Slots run down the left column, then the right: dots 1-4, then 5-8.
    slots = np.arange(CELL_DOTS_X * CELL_DOTS_Y)
    slot_row = slots % CELL_DOTS_Y
    slot_col = slots // CELL_DOTS_Y

    # (cell_row, cell_col, slot) -> (dot_row, dot_col)
    dot_row = np.arange(rows)[:, None, None] * CELL_DOTS_Y + slot_row
    dot_col = np.arange(cols)[None, :, None] * CELL_DOTS_X + slot_col

    grid = np.zeros((rows * CELL_DOTS_Y, cols * CELL_DOTS_X), dtype=bool)
    grid[dot_row, dot_col] = filled
    return grid


def dot_kernel() -> np.ndarray:
    """A filled circle inscribed in one supersampled DOT_PITCH square."""
    size = int(round(DOT_PITCH * render_scale()))
    radius = size * DOT_FILL / 2.0
    centre = (size - 1) / 2.0
    offsets = np.arange(size) - centre

    dist = np.hypot(offsets[:, None], offsets[None, :])
    return dist <= radius


def render_alpha(
    pattern: np.ndarray, offsets: np.ndarray | None = None
) -> Image.Image:
    """Stamp the dot mask into a coverage buffer and downsample it to final size.

    `offsets` is an optional (2, y, x) array of per-dot displacements in dot-grid
    units. Dots are stamped at their displaced position rather than resampling
    the finished image, so warping smears the grid without blurring or doubling
    individual dots.
    """
    size = int(round(DOT_PITCH * render_scale()))
    dot_rows, dot_cols = pattern.shape
    kernel = dot_kernel()

    if offsets is None:
        offsets = np.zeros((2, dot_rows, dot_cols), dtype=np.float32)
    shift_y = np.rint(offsets[0] * DOT_PITCH * size).astype(int)
    shift_x = np.rint(offsets[1] * DOT_PITCH * size).astype(int)

    # A warped dot can reach outside its own block, so stamp into a buffer with
    # a margin and crop it back out after. The margin comes from the measured
    # shift so it cannot fall short of the real displacement.
    margin = int(max(abs(shift_y).max(), abs(shift_x).max())) + 1
    height = dot_rows * size + 2 * margin
    width = dot_cols * size + 2 * margin
    strips = np.zeros((height, width), dtype=np.uint8)

    base_y = np.repeat(np.arange(dot_rows) * size, dot_cols).reshape(dot_rows, dot_cols)
    base_x = np.tile(np.arange(dot_cols) * size, (dot_rows, 1))
    origin_y = base_y + shift_y + margin
    origin_x = base_x + shift_x + margin

    # Scatter each filled dot once per nonzero kernel pixel. maximum.at keeps
    # overlapping dots from summing past full coverage.
    filled_y, filled_x = np.nonzero(pattern)
    if filled_y.size:
        for ky, kx in np.argwhere(kernel):
            np.maximum.at(
                strips,
                (origin_y[filled_y, filled_x] + ky, origin_x[filled_y, filled_x] + kx),
                np.uint8(255),
            )

    # Downsample to the whole-cell grid, not straight to the output size: a
    # partial cell at the edge would otherwise get stretched to fill the
    # remainder instead of staying absent.
    cells_x, cells_y = cell_grid()
    cropped = strips[margin : margin + dot_rows * size, margin : margin + dot_cols * size]
    grid = Image.fromarray(cropped, "L").resize(
        (cells_x * CELL_W, cells_y * CELL_H), Image.LANCZOS
    )

    canvas = Image.new("L", (OUTPUT_WIDTH, OUTPUT_HEIGHT), 0)
    canvas.paste(grid, (0, 0))
    return canvas


def write_colorway(alpha: Image.Image, rgb: tuple[int, int, int], path: str) -> None:
    """Composite the coverage buffer over a solid color and save it."""
    image = Image.new("RGBA", (OUTPUT_WIDTH, OUTPUT_HEIGHT), rgb + (255,))
    image.putalpha(alpha)
    image.save(path, "PNG")


# --- Verification -----------------------------------------------------------


def check_density(pattern: np.ndarray) -> None:
    """Assert the rendered pattern honours the requested fade.

    Thresholds carry DENSITY_PULL because the figures deliberately thin a
    nominally full row, so the top band is no longer expected to be 1.0.
    """
    rows = pattern.shape[0] // CELL_DOTS_Y
    per_row = pattern.reshape(rows, -1).mean(axis=1)
    expected = row_density()

    top = per_row[: max(1, rows // 20)].mean()
    bottom = per_row[-max(1, rows // 20) :].mean()

    # A row the ramp calls full may still be pulled down by DENSITY_PULL.
    assert top > 1.0 - DENSITY_PULL - 0.02, f"top band is not near-full: {top:.3f}"
    assert bottom < 0.02 + DENSITY_PULL * 0.1, f"bottom band is not empty: {bottom:.3f}"

    worst = float(np.max(np.abs(per_row - expected)))
    assert worst <= DENSITY_PULL + 0.05, f"per-row density deviates by {worst:.3f}"

    # Monotonic and seam-free, asserted against the measurement itself rather
    # than against row_density(), so a flaw in that function cannot hide here.
    # A seam shows up as one row dropping far more than the smooth trend.
    # Without figures the steepest legitimate step is ~0.033, but figures
    # legitimately vary a row by up to DENSITY_PULL, so the bound scales with
    # them. A mis-scaled fade drops 0.83 in a single row, far outside this.
    tol = DENSITY_PULL * 0.8
    step = np.diff(per_row)
    assert step.max() <= tol, f"density rises downward: {step.max():.3f}"
    assert step.min() >= -tol, f"density drops by {step.min():.3f} in one row (seam?)"


def check_render(alpha: Image.Image) -> None:
    """Assert the downsampled coverage buffer actually holds 0..255 coverage."""
    coverage = np.asarray(alpha, dtype=np.uint8)

    assert coverage.max() > 250, f"no opaque coverage rendered: max {coverage.max()}"
    assert coverage.min() == 0, f"no empty pixels rendered: min {coverage.min()}"

    # This checks the renderer, not the art: a solid full-density pattern with no
    # figures and no warp must come back at the disc's own area fraction. The
    # 6% slack absorbs LANCZOS edge weighting across the supersample downscale,
    # measured at 3.6% for the round 18px disc. A 6px block loses far less
    # because its disc is already a blocky square.
    dot_area = float(dot_kernel().mean())
    solid = render_alpha(np.ones(grid_shape(), dtype=bool))
    solid_cov = np.asarray(solid, dtype=np.uint8)
    cell_rows = int(OUTPUT_HEIGHT // CELL_H)
    band = cell_rows // 8
    top = solid_cov[: band * CELL_H].mean() / 255
    assert abs(top - dot_area) < 0.06 * dot_area, f"solid coverage {top:.3f} vs {dot_area:.3f}"

    # The real render must not be emptier than the figures allow, and must not
    # exceed a solid field.
    top_art = coverage[: band * CELL_H].mean() / 255
    floor = dot_area * (1.0 - DENSITY_PULL) - 0.02
    assert top_art > floor, f"top art coverage {top_art:.3f} below floor {floor:.3f}"
    assert top_art <= dot_area + 0.02, f"top art coverage {top_art:.3f} above solid {dot_area:.3f}"


def check_dots() -> None:
    """Assert the dot a viewer actually sees is a circle, not a rounded square.

    Measured on the *final* output pixels, not on the pre-downsample kernel: a
    disc drawn large and downsampled still lands on a rounded square if the
    output pitch is too small, because the limit is in the final pixels.

    The test counts uniform inner rows in an isolated rendered dot. A disc has
    no row that is fully opaque across its whole width; a rasterised square does.
    """
    pitch = int(round(DOT_PITCH))
    assert pitch >= MIN_DOT_PITCH, (
        f"DOT_PITCH={pitch} is below MIN_DOT_PITCH={MIN_DOT_PITCH}: a circle "
        f"cannot be rasterised that small. Raise BRAILLE_SIZE to "
        f"{MIN_DOT_PITCH * CELL_DOTS_X} or more."
    )

    # Render one isolated dot at output size through the real pipeline.
    size = int(round(DOT_PITCH * render_scale()))
    off = np.arange(size) - (size - 1) / 2.0
    disc = np.hypot(off[:, None], off[None, :]) <= size * DOT_FILL / 2.0

    buffer = np.zeros((size, 3 * size), dtype=np.uint8)
    for i in range(3):
        buffer[:, i * size : (i + 1) * size] = np.where(disc, 255, 0)
    strip = Image.fromarray(buffer, "L").resize((3 * pitch, pitch), Image.LANCZOS)
    dot = np.asarray(strip)[:, :pitch]

    uniform = [
        r
        for r in range(1, pitch - 1)
        if dot[r].min() > 200 and dot[r].max() > 240
    ]
    assert not uniform, (
        f"dot rasterises as a square: rows {uniform} are uniform across the full "
        f"width at a {pitch}px pitch. Raise BRAILLE_SIZE."
    )


def check_figures(field: np.ndarray) -> None:
    """Assert the figure curves cannot punch holes or swamp the base ramp.

    The figures are meant to be a small perturbation. This is the guard against
    them growing strong enough to blank out a dense region, which would read as
    a bald patch rather than as texture.
    """
    ramp = row_density()[:, None]
    assert np.abs(field).max() <= DENSITY_PULL + 1e-6, (
        f"figure field exceeds DENSITY_PULL: {np.abs(field).max():.3f}"
    )

    # Mirror dot_pattern exactly: average the figure field to per cell, weight
    # it by the ramp, then apply. Measuring a different formula than the one
    # under test would let a hole slip through.
    # cell_grid() returns (columns, rows).
    cols, rows = cell_grid()
    base = np.repeat(ramp, CELL_DOTS_Y, axis=0)
    per_cell = field.reshape(rows, CELL_DOTS_Y, cols, CELL_DOTS_X).mean(axis=(1, 3))
    weight = np.clip(ramp / max(DENSITY_PULL, 1e-6), 0.0, 1.0)
    per_row = np.clip(ramp + per_cell * weight, 0.0, 1.0)

    # A figure may only ever nudge a row's density by DENSITY_PULL, so the
    # per-row deviation from the base ramp has a hard ceiling.
    dev = np.abs(per_row - ramp).max()
    assert dev <= DENSITY_PULL + 1e-6, f"figures shift row density by {dev:.3f}"

    # Nothing may be emptied where the base ramp is dense. A hole shows up as a
    # cell row whose density is far below what the ramp asked for.
    dense = ramp[:, 0] > 0.5
    if dense.any():
        lost = (ramp[dense, 0, None] - per_row[dense]).max()
        assert lost <= DENSITY_PULL + 1e-6, f"figures empty a dense region by {lost:.3f}"


# --- Entry point ------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        help="hex token from secrets; omit to draw a fresh one and print it",
    )
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    token = secrets.token_bytes(32)
    seed = args.seed or token.hex()
    if args.seed is None:
        print(f"seed: {seed}")

    rng = np.random.default_rng(
        np.random.SeedSequence(int.from_bytes(bytes.fromhex(seed), "big"))
    )

    check_geometry()

    # Figures and warp consume the same generator, so a given seed reproduces
    # the exact same texture end to end.
    field = figure_field(rng, grid_shape())
    check_figures(field)
    offsets = warp_offsets(rng, grid_shape())

    check_dots()
    pattern = dot_pattern(rng, field)
    check_density(pattern)
    alpha = render_alpha(pattern, offsets)
    check_render(alpha)

    os.makedirs(args.out_dir, exist_ok=True)
    for filename, rgb in COLORWAYS:
        path = f"{args.out_dir}/{filename}"
        write_colorway(alpha, rgb, path)
        print(f"wrote {path}  {OUTPUT_WIDTH}x{OUTPUT_HEIGHT}")


if __name__ == "__main__":
    main()
