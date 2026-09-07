"""Pure frontier extraction and ranking for the explore-and-return node.

The functions in this module deliberately depend only on an OccupancyGrid-like
object and NumPy.  Keeping the map work outside the ROS node makes the
exploration policy cheap to test and prevents it from accidentally using any
simulator-only ground-truth information.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import heapq
import math
from typing import Sequence

import numpy as np


FREE_MAX = 25
OCCUPIED_MIN = 65


@dataclass(frozen=True)
class ExclusionZone:
    """A temporary world-coordinate keep-out region for a frontier target."""

    x: float
    y: float
    expires_at_s: float


@dataclass(frozen=True)
class FrontierTarget:
    """A safe, known-free navigation target representing one frontier."""

    x: float
    y: float
    yaw: float
    path_length_m: float
    frontier_length_m: float
    frontier_points: tuple[tuple[float, float], ...] = ()
    needs_nav2_path: bool = False
    information_gain_m2: float = 0.0
    information_gain_weight: float = 0.0
    distance_penalty_weight: float = 1.0
    has_nav2_path_heading: bool = False

    @property
    def score(self) -> float:
        """Exploration utility, trading expected map gain against travel cost.

        ``information_gain_weight`` converts the ray-cast unknown area (m²)
        into a frontier-length-equivalent quantity (m). A zero weight exactly
        reproduces the original frontier-length-only score.
        """
        expected_gain = self.frontier_length_m + self.information_gain_weight * self.information_gain_m2
        return expected_gain / (1.0 + self.distance_penalty_weight * self.path_length_m)


@dataclass(frozen=True)
class InvalidFrontier:
    """A raw frontier component and the first validation rule it failed."""

    reason: str
    points: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class FrontierPlan:
    """All valid targets plus the unfiltered component count for diagnostics."""

    frontier_components_found: int
    valid_targets: tuple[FrontierTarget, ...]
    blacklisted_components: tuple[tuple[tuple[float, float], ...], ...]
    invalid_components: tuple[InvalidFrontier, ...]
    information_candidates_total: int = 0
    information_candidates_evaluated: int = 0
    has_more_information_candidates: bool = False


@dataclass(frozen=True)
class PlannerConfig:
    planning_resolution_m: float = 0.06
    clearance_radius_m: float = 0.20
    min_frontier_length_m: float = 0.25
    min_goal_distance_m: float = 0.50
    exclusion_radius_m: float = 0.75
    unknown_bridge_distance_m: float = 0.30
    allow_nav2_path_fallback: bool = True
    raycast_max_range_m: float = 3.0
    raycast_rays: int = 120
    information_gain_weight: float = 0.50
    min_information_gain_m2: float = 0.50
    distance_penalty_weight: float = 1.25
    raycast_candidate_limit: int = 20


@dataclass(frozen=True)
class _GridGeometry:
    origin_x: float
    origin_y: float
    origin_yaw: float
    resolution: float
    height: int
    width: int

    def cell_to_world(self, row: int, col: int) -> tuple[float, float]:
        local_x = (col + 0.5) * self.resolution
        local_y = (row + 0.5) * self.resolution
        cos_yaw = math.cos(self.origin_yaw)
        sin_yaw = math.sin(self.origin_yaw)
        return (
            self.origin_x + cos_yaw * local_x - sin_yaw * local_y,
            self.origin_y + sin_yaw * local_x + cos_yaw * local_y,
        )

    def world_to_cell(self, x: float, y: float) -> tuple[int, int] | None:
        dx = x - self.origin_x
        dy = y - self.origin_y
        cos_yaw = math.cos(self.origin_yaw)
        sin_yaw = math.sin(self.origin_yaw)
        local_x = cos_yaw * dx + sin_yaw * dy
        local_y = -sin_yaw * dx + cos_yaw * dy
        col = int(math.floor(local_x / self.resolution))
        row = int(math.floor(local_y / self.resolution))
        if 0 <= row < self.height and 0 <= col < self.width:
            return row, col
        return None


def _yaw_from_quaternion(quaternion) -> float:
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
    )


def _coarsen_map(map_msg, config: PlannerConfig) -> tuple[np.ndarray, np.ndarray, _GridGeometry]:
    """Return conservative free/occupied masks at the planning resolution.

    A planning cell is usable when it contains observed free space and has no
    occupied evidence. Mixed cells are intentionally retained: an early SLAM
    map consists of thin laser-cleared rays, so requiring every source pixel
    to be free would erase nearly all of the navigable evidence. Obstacle
    clearance is applied separately and conservatively below.
    """
    info = map_msg.info
    raw = np.asarray(map_msg.data, dtype=np.int16).reshape((info.height, info.width))
    factor = max(1, int(math.ceil(config.planning_resolution_m / info.resolution)))
    coarse_height = int(math.ceil(info.height / factor))
    coarse_width = int(math.ceil(info.width / factor))
    padded = np.full((coarse_height * factor, coarse_width * factor), -1, dtype=np.int16)
    padded[: info.height, : info.width] = raw
    blocks = padded.reshape(coarse_height, factor, coarse_width, factor)
    occupied = (blocks >= OCCUPIED_MIN).any(axis=(1, 3))
    free = ((blocks >= 0) & (blocks <= FREE_MAX)).any(axis=(1, 3))
    free &= ~occupied
    origin = info.origin
    geometry = _GridGeometry(
        origin_x=origin.position.x,
        origin_y=origin.position.y,
        origin_yaw=_yaw_from_quaternion(origin.orientation),
        resolution=info.resolution * factor,
        height=coarse_height,
        width=coarse_width,
    )
    return free, occupied, geometry


def _clearance_mask(occupied: np.ndarray, resolution: float, clearance_radius_m: float) -> np.ndarray:
    """Cells whose circular footprint does not intersect an obstacle."""
    radius_cells = int(math.ceil(clearance_radius_m / resolution))
    clear = np.ones_like(occupied, dtype=bool)
    padded = np.pad(occupied, radius_cells, mode="constant", constant_values=True)
    height, width = occupied.shape
    for dr in range(-radius_cells, radius_cells + 1):
        for dc in range(-radius_cells, radius_cells + 1):
            if dr * dr + dc * dc > (clearance_radius_m / resolution) ** 2:
                continue
            row_start = radius_cells + dr
            col_start = radius_cells + dc
            clear &= ~padded[row_start : row_start + height, col_start : col_start + width]
    return clear


def _safe_free_mask(free: np.ndarray, occupied: np.ndarray, resolution: float, clearance_radius_m: float) -> np.ndarray:
    """Remove free cells whose circular footprint intersects an obstacle."""
    return free & _clearance_mask(occupied, resolution, clearance_radius_m)


def _dilate(mask: np.ndarray, radius_m: float, resolution: float) -> np.ndarray:
    """Return cells no farther than radius_m from a true source cell."""
    radius_cells = int(math.ceil(radius_m / resolution))
    padded = np.pad(mask, radius_cells, mode="constant", constant_values=False)
    height, width = mask.shape
    expanded = np.zeros_like(mask, dtype=bool)
    for dr in range(-radius_cells, radius_cells + 1):
        for dc in range(-radius_cells, radius_cells + 1):
            if dr * dr + dc * dc > (radius_m / resolution) ** 2:
                continue
            row_start = radius_cells + dr
            col_start = radius_cells + dc
            expanded |= padded[row_start : row_start + height, col_start : col_start + width]
    return expanded


def _adjacent_to(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    adjacent = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            adjacent |= padded[1 + dr : 1 + dr + height, 1 + dc : 1 + dc + width]
    return adjacent


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return 8-connected cells without requiring scipy/opencv."""
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for row, col in np.argwhere(mask):
        row = int(row)
        col = int(col)
        if seen[row, col]:
            continue
        seen[row, col] = True
        component: list[tuple[int, int]] = []
        stack = [(row, col)]
        while stack:
            current_row, current_col = stack.pop()
            component.append((current_row, current_col))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    next_row = current_row + dr
                    next_col = current_col + dc
                    if (
                        0 <= next_row < height
                        and 0 <= next_col < width
                        and mask[next_row, next_col]
                        and not seen[next_row, next_col]
                    ):
                        seen[next_row, next_col] = True
                        stack.append((next_row, next_col))
        components.append(component)
    return components


def _nearest_safe_cell(safe: np.ndarray, desired: tuple[int, int] | None) -> tuple[int, int] | None:
    if desired is not None and safe[desired]:
        return desired
    cells = np.argwhere(safe)
    if len(cells) == 0:
        return None
    if desired is None:
        return tuple(int(value) for value in cells[0])
    distances_sq = (cells[:, 0] - desired[0]) ** 2 + (cells[:, 1] - desired[1]) ** 2
    return tuple(int(value) for value in cells[int(np.argmin(distances_sq))])


def _dijkstra(safe: np.ndarray, start: tuple[int, int], resolution: float) -> np.ndarray:
    """Known-free path lengths in metres, with diagonal corner-cutting blocked."""
    distances = np.full(safe.shape, np.inf, dtype=float)
    distances[start] = 0.0
    queue: list[tuple[float, int, int]] = [(0.0, start[0], start[1])]
    height, width = safe.shape
    while queue:
        distance, row, col = heapq.heappop(queue)
        if distance != distances[row, col]:
            continue
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                next_row = row + dr
                next_col = col + dc
                if not (0 <= next_row < height and 0 <= next_col < width and safe[next_row, next_col]):
                    continue
                if dr != 0 and dc != 0 and (not safe[row + dr, col] or not safe[row, col + dc]):
                    continue
                step = resolution * (math.sqrt(2.0) if dr != 0 and dc != 0 else 1.0)
                candidate = distance + step
                if candidate < distances[next_row, next_col]:
                    distances[next_row, next_col] = candidate
                    heapq.heappush(queue, (candidate, next_row, next_col))
    return distances


def _bresenham_cells(
    start_row: int, start_col: int, end_row: int, end_col: int
) -> list[tuple[int, int]]:
    """Integer cells intersected by a ray, including both endpoints."""
    cells: list[tuple[int, int]] = []
    row, col = start_row, start_col
    step_row = 1 if end_row >= row else -1
    step_col = 1 if end_col >= col else -1
    delta_row = abs(end_row - row)
    delta_col = abs(end_col - col)
    error = delta_col - delta_row
    while True:
        cells.append((row, col))
        if row == end_row and col == end_col:
            break
        doubled_error = 2 * error
        if doubled_error > -delta_row:
            error -= delta_row
            col += step_col
        if doubled_error < delta_col:
            error += delta_col
            row += step_row
    return cells


def raycast_information_gain_m2(
    unknown: np.ndarray,
    occupied: np.ndarray,
    start: tuple[int, int],
    resolution_m: float,
    max_range_m: float,
    ray_count: int,
) -> float:
    """Estimate uniquely visible unknown area from a prospective goal.

    The cast is deliberately optimistic about *unknown* cells: a laser placed
    at the goal may observe through them until a known obstacle or the sensor
    range is reached. Known occupied cells block rays, and a cell seen by
    several rays is counted just once. This is a planning heuristic only; it
    never changes the safety or reachability masks.
    """
    if max_range_m <= 0.0 or ray_count <= 0:
        return 0.0
    max_cells = int(math.ceil(max_range_m / resolution_m))
    visible_unknown = np.zeros_like(unknown, dtype=bool)
    height, width = unknown.shape
    for ray_index in range(ray_count):
        angle = 2.0 * math.pi * ray_index / ray_count
        end_row = start[0] + int(round(math.sin(angle) * max_cells))
        end_col = start[1] + int(round(math.cos(angle) * max_cells))
        for row, col in _bresenham_cells(start[0], start[1], end_row, end_col)[1:]:
            if not (0 <= row < height and 0 <= col < width):
                break
            if occupied[row, col]:
                break
            if unknown[row, col]:
                visible_unknown[row, col] = True
    return float(np.count_nonzero(visible_unknown)) * resolution_m * resolution_m


def is_excluded(
    x: float, y: float, now_s: float, exclusions: Sequence[ExclusionZone], radius_m: float
) -> bool:
    radius_sq = radius_m * radius_m
    return any(
        exclusion.expires_at_s > now_s
        and (exclusion.x - x) ** 2 + (exclusion.y - y) ** 2 <= radius_sq
        for exclusion in exclusions
    )


def rank_frontier_targets(targets: Sequence[FrontierTarget]) -> FrontierTarget | None:
    ranked = ranked_frontier_targets(targets)
    return ranked[0] if ranked else None


def ranked_frontier_targets(targets: Sequence[FrontierTarget]) -> list[FrontierTarget]:
    """Order valid targets from highest to lowest exploration utility."""
    return sorted(targets, key=lambda target: (-target.score, -target.frontier_length_m, target.path_length_m))


def find_frontier_targets(
    map_msg,
    robot_xy: tuple[float, float],
    now_s: float,
    exclusions: Sequence[ExclusionZone],
    config: PlannerConfig = PlannerConfig(),
) -> list[FrontierTarget]:
    """Return every valid frontier component, sorted by exploration utility."""
    return list(plan_frontiers(map_msg, robot_xy, now_s, exclusions, config).valid_targets)


def plan_frontiers(
    map_msg,
    robot_xy: tuple[float, float],
    now_s: float,
    exclusions: Sequence[ExclusionZone],
    config: PlannerConfig = PlannerConfig(),
    raycast_candidate_offset: int = 0,
) -> FrontierPlan:
    """Build ranked candidates and retain counts useful for operator diagnostics."""
    free, occupied, geometry = _coarsen_map(map_msg, config)
    clearance = _clearance_mask(occupied, geometry.resolution, config.clearance_radius_m)
    safe = free & clearance
    unknown = ~(free | occupied)
    frontier = safe & _adjacent_to(unknown)
    components = _connected_components(frontier)
    start = _nearest_safe_cell(safe, geometry.world_to_cell(*robot_xy))
    if start is None:
        return FrontierPlan(
            frontier_components_found=len(components),
            valid_targets=(),
            blacklisted_components=(),
            invalid_components=tuple(
                InvalidFrontier(
                    reason="no_safe_start",
                    points=tuple(geometry.cell_to_world(row, col) for row, col in component),
                )
                for component in components
            ),
        )
    # Nav2 may plan into unknown space near a frontier. Permit only a short,
    # obstacle-clear bridge across SLAM gaps so that one unmapped strip does
    # not falsely isolate an otherwise reachable frontier. The bridge is
    # bounded from known safe space, rather than allowing arbitrary traversal
    # through the entire unknown map.
    bridge = unknown & clearance & _dilate(safe, config.unknown_bridge_distance_m, geometry.resolution)
    traversable = safe | bridge
    distances = _dijkstra(traversable, start, geometry.resolution)
    unscored_targets: list[tuple[FrontierTarget, int, int]] = []
    blacklisted: list[tuple[tuple[float, float], ...]] = []
    invalid: list[InvalidFrontier] = []
    for component in components:
        component_points = tuple(geometry.cell_to_world(row, col) for row, col in component)
        frontier_length = len(component) * geometry.resolution
        if frontier_length < config.min_frontier_length_m:
            invalid.append(InvalidFrontier(reason="too_small", points=component_points))
            continue
        reachable = [(distances[row, col], row, col) for row, col in component if math.isfinite(distances[row, col])]
        if not reachable:
            # The local planner is deliberately more conservative than Nav2.
            # Keep a safe goal pending and ask Nav2 for a real path before it
            # can participate in ranking; never substitute straight-line
            # distance for a route around walls.
            if not config.allow_nav2_path_fallback:
                invalid.append(InvalidFrontier(reason="unreachable", points=component_points))
                continue
            reachable = []
            for row, col in component:
                target_x, target_y = geometry.cell_to_world(row, col)
                # This distance is used only to reject Nav2 no-op goals. The
                # final score is replaced by ComputePathToPose's route length.
                distance_from_robot = math.hypot(target_x - robot_xy[0], target_y - robot_xy[1])
                if distance_from_robot >= config.min_goal_distance_m:
                    reachable.append((distance_from_robot, row, col))
            if not reachable:
                invalid.append(InvalidFrontier(reason="too_close", points=component_points))
                continue
            needs_nav2_path = True
        else:
            needs_nav2_path = False
        # Nav2 considers goals within 0.25 m reached. Sending the closest
        # frontier cell can therefore produce a successful no-op at the
        # robot's current map boundary, so require a meaningful outward move.
        meaningful = [entry for entry in reachable if entry[0] >= config.min_goal_distance_m]
        if not meaningful:
            invalid.append(InvalidFrontier(reason="too_close", points=component_points))
            continue
        # The geometric centroid itself may be unknown or occupied. Project it
        # onto the closest eligible frontier cell instead of choosing a
        # component endpoint, which avoids a persistent wall-side bias while
        # retaining the minimum-distance and safety guarantees.
        centroid_x, centroid_y = np.mean(component_points, axis=0)
        path_length, target_row, target_col = min(
            meaningful,
            key=lambda entry: (
                (geometry.cell_to_world(entry[1], entry[2])[0] - centroid_x) ** 2
                + (geometry.cell_to_world(entry[1], entry[2])[1] - centroid_y) ** 2,
                entry[0],
            ),
        )
        target_x, target_y = geometry.cell_to_world(target_row, target_col)
        if is_excluded(target_x, target_y, now_s, exclusions, config.exclusion_radius_m):
            blacklisted.append(component_points)
            continue
        unscored_targets.append(
            (
                FrontierTarget(
                    x=target_x,
                    y=target_y,
                    # The planner does not prescribe a frontier-facing yaw.
                    # explorer_node.py replaces this placeholder with Nav2's
                    # final predicted path tangent before navigation.
                    yaw=0.0,
                    path_length_m=float(path_length) if not needs_nav2_path else math.inf,
                    frontier_length_m=frontier_length,
                    frontier_points=component_points,
                    needs_nav2_path=needs_nav2_path,
                    information_gain_weight=config.information_gain_weight,
                    distance_penalty_weight=config.distance_penalty_weight,
                ),
                target_row,
                target_col,
            )
        )

    # Ray casting every raw frontier makes planning cost scale poorly with
    # noisy maps. First rank every eligible component using only the cheap
    # size-versus-distance score, then perform expensive visibility estimates
    # for one bounded batch. The node advances the offset if a batch yields no
    # usable goal, so untested candidates never imply frontier exhaustion.
    ranked_unscored = sorted(
        unscored_targets,
        key=lambda entry: (-entry[0].score, -entry[0].frontier_length_m, entry[0].path_length_m),
    )
    batch_start = min(len(ranked_unscored), max(0, raycast_candidate_offset))
    batch_limit = max(1, config.raycast_candidate_limit)
    batch_end = min(len(ranked_unscored), batch_start + batch_limit)
    targets: list[FrontierTarget] = []
    for target, target_row, target_col in ranked_unscored[batch_start:batch_end]:
        information_gain_m2 = raycast_information_gain_m2(
            unknown,
            occupied,
            (target_row, target_col),
            geometry.resolution,
            config.raycast_max_range_m,
            config.raycast_rays,
        )
        if information_gain_m2 < config.min_information_gain_m2:
            invalid.append(InvalidFrontier(reason="insufficient_information_gain", points=target.frontier_points))
            continue
        targets.append(replace(target, information_gain_m2=information_gain_m2))
    return FrontierPlan(
        frontier_components_found=len(components),
        valid_targets=tuple(ranked_frontier_targets(targets)),
        blacklisted_components=tuple(blacklisted),
        invalid_components=tuple(invalid),
        information_candidates_total=len(ranked_unscored),
        information_candidates_evaluated=batch_end - batch_start,
        has_more_information_candidates=batch_end < len(ranked_unscored),
    )


def select_frontier_target(
    map_msg,
    robot_xy: tuple[float, float],
    now_s: float,
    exclusions: Sequence[ExclusionZone],
    config: PlannerConfig = PlannerConfig(),
) -> FrontierTarget | None:
    """Select the highest-utility safe frontier reachable through known free space."""
    return rank_frontier_targets(find_frontier_targets(map_msg, robot_xy, now_s, exclusions, config))
