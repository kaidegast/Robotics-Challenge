"""2D raycasting visibility for the accurate-scan sensor model.

Mirrors what a real rotating high-accuracy 2D laser would see from a fixed stop:
cast a dense ring of rays, walk each one cell-by-cell (DDA) until it hits an
occupied (wall) cell or exceeds max_range, and record that wall cell as
"observed" from this stop, with a quality score that falls off with range.

Quality combines the existing range falloff with a wall-incidence term: a real
scanner sees an obliquely struck wall less accurately than one struck head-on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .map_io import FREE, OCCUPIED, OccupancyGrid


_FACE_DIRECTIONS = tuple(
    (row_offset, col_offset)
    for row_offset in (-1, 0, 1)
    for col_offset in (-1, 0, 1)
    if row_offset != 0 or col_offset != 0
)
_INCIDENCE_EXPONENT = 1.0


@dataclass
class SensorModel:
    max_range_m: float = 16
    num_rays: int = 720  # 0.5 degree angular resolution
    min_quality: float = 0.5  # a wall cell only counts as "scanned" if observed at >= this quality


def quality_at_range(range_m: float, max_range_m: float,
                     incidence_cos: float | np.ndarray = 1.0) -> float | np.ndarray:
    """Quadratic falloff: 1.0 at range 0, 0.0 at max_range. Depth uncertainty
    for real range sensors (stereo, ToF) grows with range^2, so quality drops
    off faster near max_range than a linear model would. ``incidence_cos`` is
    the cosine between an exposed wall-face normal and the wall-to-scanner
    direction: 1.0 is head-on and 0.0 is grazing. The product models the
    weaker return and poorer surface localization of oblique measurements.
    Works for scalar or numpy-array inputs alike."""
    distance_quality = np.maximum(0.0, 1.0 - (range_m / max_range_m) ** 2)
    incidence_quality = np.clip(incidence_cos, 0.0, 1.0) ** _INCIDENCE_EXPONENT
    return distance_quality * incidence_quality


def _wall_face_masks(grid: OccupancyGrid) -> np.ndarray:
    """Return cached exposed wall faces indexed by cell and neighbor direction."""
    cached = getattr(grid, "_wall_face_masks", None)
    if cached is not None:
        return cached

    free = grid.data == FREE
    padded_free = np.pad(free, 1, mode="constant", constant_values=False)
    occupied = grid.data == OCCUPIED
    faces = np.zeros((*grid.data.shape, len(_FACE_DIRECTIONS)), dtype=bool)
    for index, (row_offset, col_offset) in enumerate(_FACE_DIRECTIONS):
        neighbor_free = padded_free[
            1 + row_offset:1 + row_offset + grid.height,
            1 + col_offset:1 + col_offset + grid.width,
        ]
        faces[:, :, index] = occupied & neighbor_free
    grid._wall_face_masks = faces  # type: ignore[attr-defined]
    return faces


def _incidence_cosines(grid: OccupancyGrid, origin_row: int, origin_col: int,
                        hit_rows: np.ndarray, hit_cols: np.ndarray,
                        range_px: np.ndarray) -> np.ndarray:
    """Return the most head-on exposed-face cosine for each wall hit."""
    faces = _wall_face_masks(grid)
    to_sensor_row = origin_row - hit_rows
    to_sensor_col = origin_col - hit_cols
    best_cosine = np.zeros(len(hit_rows), dtype=float)
    for face_index, (row_offset, col_offset) in enumerate(_FACE_DIRECTIONS):
        face_exists = faces[hit_rows, hit_cols, face_index]
        if not face_exists.any():
            continue
        face_length = np.hypot(row_offset, col_offset)
        cosine = (row_offset * to_sensor_row + col_offset * to_sensor_col) / (face_length * range_px)
        best_cosine = np.maximum(best_cosine, np.where(face_exists, cosine, 0.0))
    return np.clip(best_cosine, 0.0, 1.0)


def scan_from_stop(grid: OccupancyGrid, stop_xy: tuple[float, float],
                    sensor: SensorModel) -> dict[tuple[int, int], float]:
    """Simulate one full rotate-and-scan from a stop. Returns {(row, col):
    quality} for every wall cell seen at quality > 0, keyed by the pixel it
    hit (best quality wins if a cell is somehow hit by more than one ray).

    All of a stop's rays are marched together as numpy arrays (DDA, one step
    per iteration across every ray at once) rather than one Python-level ray
    at a time - this is called once per candidate stop a planner considers,
    so its cost adds up quickly across a large candidate set on a large map."""
    origin_row, origin_col = grid.world_to_pixel(*stop_xy)
    max_range_px = sensor.max_range_m / grid.resolution
    steps = int(max_range_px)
    h, w = grid.data.shape
    occ_mask = grid.data == OCCUPIED

    angles = 2 * np.pi * np.arange(sensor.num_rays) / sensor.num_rays
    dr = -np.sin(angles)  # row decreases as world y increases (see map_io convention)
    dc = np.cos(angles)
    r = np.full(sensor.num_rays, float(origin_row))
    c = np.full(sensor.num_rays, float(origin_col))
    active = np.ones(sensor.num_rays, dtype=bool)

    seen: dict[tuple[int, int], float] = {}
    for _ in range(steps):
        if not active.any():
            break
        r += dr
        c += dc
        ir = r.astype(int)
        ic = c.astype(int)
        in_bounds = active & (ir >= 0) & (ir < h) & (ic >= 0) & (ic < w)
        active &= in_bounds

        occ_now = np.zeros(sensor.num_rays, dtype=bool)
        occ_now[active] = occ_mask[ir[active], ic[active]]
        hit_now = active & occ_now
        if hit_now.any():
            range_px = np.hypot(ir[hit_now] - origin_row, ic[hit_now] - origin_col)
            incidence_cos = _incidence_cosines(
                grid, origin_row, origin_col, ir[hit_now], ic[hit_now], range_px,
            )
            quality = quality_at_range(
                range_px * grid.resolution, sensor.max_range_m, incidence_cos,
            )
            for row, col, q in zip(ir[hit_now], ic[hit_now], quality):
                cell = (int(row), int(col))
                if q > seen.get(cell, 0.0):
                    seen[cell] = q
            active &= ~hit_now
    return seen


def observable_wall_cells(grid: OccupancyGrid) -> np.ndarray:
    """Boolean (height, width) mask of wall cells that have at least one FREE
    8-neighbor inside the grid.

    A ray in `scan_from_stop` always stops at the *first* occupied cell it hits, so
    a cell buried inside a thick wall (e.g. a multi-pixel-deep exterior wall)
    can never be the hit cell for any ray from any free-space stop, no matter
    how the stops are placed. Counting those interior cells toward a coverage
    denominator makes 100% coverage geometrically impossible.

    Neighbor check is against FREE specifically, not merely non-occupied:
    UNKNOWN cells are never a valid stop (`is_stop_valid` requires FREE), and
    a wall cell facing only UNKNOWN space (e.g. the outer face of an exterior
    wall ring, with the unmapped/unreachable area beyond it) is not actually
    reachable either - any ray from inside the free interior hits the wall's
    free-facing side first, exactly like an interior cell would be. A cell
    directly adjacent to a FREE cell is always reachable, at minimum by a ray
    cast from that neighbor itself."""
    occ = grid.data == OCCUPIED
    free = grid.data == FREE
    padded = np.pad(free, 1, mode="constant", constant_values=False)
    neighbor_free = np.zeros_like(occ, dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            neighbor_free |= padded[1 + dr: 1 + dr + occ.shape[0], 1 + dc: 1 + dc + occ.shape[1]]
    return occ & neighbor_free


def is_stop_valid(grid: OccupancyGrid, stop_xy: tuple[float, float], robot_radius_m: float = 0.2) -> bool:
    """A stop is only physically realizable if it's in free space with room for
    the robot's footprint around it."""
    row, col = grid.world_to_pixel(*stop_xy)
    if not grid.is_free(row, col):
        return False
    radius_px = int(np.ceil(robot_radius_m / grid.resolution))
    for dr in range(-radius_px, radius_px + 1):
        for dc in range(-radius_px, radius_px + 1):
            if dr * dr + dc * dc > radius_px * radius_px:
                continue
            r, c = row + dr, col + dc
            if grid.in_bounds(r, c) and grid.is_occupied(r, c):
                return False
    return True
