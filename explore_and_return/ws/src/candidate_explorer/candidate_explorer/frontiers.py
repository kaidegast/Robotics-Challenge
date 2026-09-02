"""Pure map-grid helpers used by the exploration node.

The helpers deliberately do not import ROS messages so their behaviour can be
tested with ordinary pytest.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


FREE_MAX = 25
OCCUPIED_MIN = 65


@dataclass(frozen=True)
class Frontier:
    """A navigable frontier target in the map frame."""

    x: float
    y: float
    yaw: float
    row: int = -1
    column: int = -1


@dataclass(frozen=True)
class FrontierCluster:
    """An 8-connected region of frontier cells."""

    cells: tuple[Frontier, ...]


@dataclass(frozen=True)
class ClusterTarget:
    """A navigable representative of a frontier cluster."""

    frontier: Frontier
    cluster_size: int
    cells: tuple[Frontier, ...]
    information_gain: int = 0
    score: float = 0.0
    vicinity_cost: float = 0.0


def is_free(value: int) -> bool:
    return 0 <= value <= FREE_MAX


def is_unknown(value: int) -> bool:
    return value < 0


def is_occupied(value: int) -> bool:
    return value >= OCCUPIED_MIN


def grid_cell_to_world(
    row: int,
    column: int,
    *,
    resolution: float,
    origin_x: float,
    origin_y: float,
    origin_yaw: float,
) -> tuple[float, float]:
    """Return the centre of an OccupancyGrid cell in its map frame."""
    local_x = (column + 0.5) * resolution
    local_y = (row + 0.5) * resolution
    cos_yaw = math.cos(origin_yaw)
    sin_yaw = math.sin(origin_yaw)
    return (
        origin_x + cos_yaw * local_x - sin_yaw * local_y,
        origin_y + sin_yaw * local_x + cos_yaw * local_y,
    )


def find_frontiers(
    data: Sequence[int],
    *,
    width: int,
    height: int,
    resolution: float,
    origin_x: float,
    origin_y: float,
    origin_yaw: float,
) -> list[Frontier]:
    """Find 4-connected free/unknown boundaries and aim each at unknown space."""
    if width <= 0 or height <= 0 or len(data) != width * height:
        return []

    frontiers: list[Frontier] = []
    for row in range(height):
        for column in range(width):
            index = row * width + column
            if not is_free(data[index]):
                continue

            for row_delta, column_delta in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                unknown_row = row + row_delta
                unknown_column = column + column_delta
                if not (0 <= unknown_row < height and 0 <= unknown_column < width):
                    continue
                if not is_unknown(data[unknown_row * width + unknown_column]):
                    continue

                x, y = grid_cell_to_world(
                    row,
                    column,
                    resolution=resolution,
                    origin_x=origin_x,
                    origin_y=origin_y,
                    origin_yaw=origin_yaw,
                )
                unknown_x, unknown_y = grid_cell_to_world(
                    unknown_row,
                    unknown_column,
                    resolution=resolution,
                    origin_x=origin_x,
                    origin_y=origin_y,
                    origin_yaw=origin_yaw,
                )
                frontiers.append(
                    Frontier(x, y, math.atan2(unknown_y - y, unknown_x - x), row, column)
                )
                break
    return frontiers


def cluster_frontiers(frontiers: Sequence[Frontier]) -> list[FrontierCluster]:
    """Group frontier cells that touch horizontally, vertically, or diagonally."""
    by_cell = {(frontier.row, frontier.column): frontier for frontier in frontiers}
    remaining = set(by_cell)
    clusters: list[FrontierCluster] = []
    while remaining:
        start = remaining.pop()
        queue = [start]
        cells: list[Frontier] = []
        while queue:
            row, column = queue.pop()
            cells.append(by_cell[(row, column)])
            for row_delta in (-1, 0, 1):
                for column_delta in (-1, 0, 1):
                    neighbor = (row + row_delta, column + column_delta)
                    if neighbor != (row, column) and neighbor in remaining:
                        remaining.remove(neighbor)
                        queue.append(neighbor)
        clusters.append(FrontierCluster(tuple(cells)))
    return clusters


def cluster_targets(
    clusters: Sequence[FrontierCluster],
    robot_x: float,
    robot_y: float,
    blocked_locations: Sequence[tuple[float, float]],
    *,
    blacklist_radius_m: float,
) -> list[ClusterTarget]:
    """Choose a centroid-nearest target for every cluster, then distance-rank it."""
    radius_squared = blacklist_radius_m * blacklist_radius_m
    targets: list[ClusterTarget] = []
    for cluster in clusters:
        eligible = [
            frontier
            for frontier in cluster.cells
            if all(
                (frontier.x - blocked_x) ** 2 + (frontier.y - blocked_y) ** 2 > radius_squared
                for blocked_x, blocked_y in blocked_locations
            )
        ]
        if not eligible:
            continue
        centroid_x = sum(frontier.x for frontier in cluster.cells) / len(cluster.cells)
        centroid_y = sum(frontier.y for frontier in cluster.cells) / len(cluster.cells)
        target = min(
            eligible,
            key=lambda frontier: (frontier.x - centroid_x) ** 2 + (frontier.y - centroid_y) ** 2,
        )
        targets.append(ClusterTarget(target, len(cluster.cells), cluster.cells))
    return targets


def local_information_gain(
    data: Sequence[int], width: int, height: int, frontier: Frontier, radius_cells: int
) -> int:
    """Count unknown cells near a target, without modelling line of sight."""
    unknown_cells = 0
    radius_squared = radius_cells * radius_cells
    for row in range(max(0, frontier.row - radius_cells), min(height, frontier.row + radius_cells + 1)):
        for column in range(max(0, frontier.column - radius_cells), min(width, frontier.column + radius_cells + 1)):
            if (row - frontier.row) ** 2 + (column - frontier.column) ** 2 <= radius_squared:
                unknown_cells += int(is_unknown(data[row * width + column]))
    return unknown_cells


def raycast_information_gain(
    data: Sequence[int], width: int, height: int, frontier: Frontier, max_range_cells: int, ray_count: int
) -> int:
    """Count unique unknown cells visible from a target before known obstacles block a ray."""
    visible_unknown: set[tuple[int, int]] = set()
    for ray in range(ray_count):
        angle = 2.0 * math.pi * ray / ray_count
        for step in range(1, max_range_cells + 1):
            row = frontier.row + round(-math.sin(angle) * step)
            column = frontier.column + round(math.cos(angle) * step)
            if not (0 <= row < height and 0 <= column < width):
                break
            value = data[row * width + column]
            if is_occupied(value):
                break
            if is_unknown(value):
                visible_unknown.add((row, column))
    return len(visible_unknown)


def add_information_gain(
    targets: Sequence[ClusterTarget], data: Sequence[int], width: int, height: int, *, mode: str, range_cells: int,
    ray_count: int,
) -> list[ClusterTarget]:
    """Attach either local-area or raycast visibility gain to each cluster target."""
    estimator = local_information_gain if mode == "local" else raycast_information_gain
    return [
        ClusterTarget(
            target.frontier,
            target.cluster_size,
            target.cells,
            estimator(data, width, height, target.frontier, range_cells)
            if mode == "local"
            else estimator(data, width, height, target.frontier, range_cells, ray_count),
            target.score,
            target.vicinity_cost,
        )
        for target in targets
    ]


def rank_cluster_targets(
    targets: Sequence[ClusterTarget], vicinity_costs: Sequence[float], *, size_weight: float, vicinity_weight: float,
    information_gain_weight: float,
) -> list[ClusterTarget]:
    """Rank clusters by normalized size, inverse travel cost, and information gain."""
    if not targets:
        return []
    max_size = max(target.cluster_size for target in targets)
    max_gain = max(target.information_gain for target in targets)
    max_cost = max(vicinity_costs) or 1.0
    scored: list[tuple[ClusterTarget, float]] = []
    for target, cost in zip(targets, vicinity_costs):
        score = (
            size_weight * target.cluster_size / max_size
            + vicinity_weight * (1.0 - cost / max_cost)
            + information_gain_weight * target.information_gain / max(max_gain, 1)
        )
        scored.append(
            (
                ClusterTarget(
                    target.frontier,
                    target.cluster_size,
                    target.cells,
                    target.information_gain,
                    score,
                    cost,
                ),
                score,
            )
        )
    return [
        target
        for target, _ in sorted(
            scored,
            key=lambda item: -item[1],
        )
    ]


def nearest_frontier(
    frontiers: Sequence[Frontier],
    robot_x: float,
    robot_y: float,
    blocked_locations: Sequence[tuple[float, float]],
    *,
    blacklist_radius_m: float,
) -> Frontier | None:
    """Choose the closest target that is not near a failed navigation goal."""
    ranked = rank_frontiers(
        frontiers,
        robot_x,
        robot_y,
        blocked_locations,
        blacklist_radius_m=blacklist_radius_m,
    )
    return ranked[0] if ranked else None


def rank_frontiers(
    frontiers: Sequence[Frontier],
    robot_x: float,
    robot_y: float,
    blocked_locations: Sequence[tuple[float, float]],
    *,
    blacklist_radius_m: float,
) -> list[Frontier]:
    """Rank eligible frontier targets by straight-line distance to the robot."""
    radius_squared = blacklist_radius_m * blacklist_radius_m
    eligible = [
        frontier
        for frontier in frontiers
        if all(
            (frontier.x - blocked_x) ** 2 + (frontier.y - blocked_y) ** 2 > radius_squared
            for blocked_x, blocked_y in blocked_locations
        )
    ]
    return sorted(eligible, key=lambda frontier: (frontier.x - robot_x) ** 2 + (frontier.y - robot_y) ** 2)
