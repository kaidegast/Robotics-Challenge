"""Your task: implement plan_viewpoints().

You are given the ground-truth occupancy grid (this is the ONLY input you get —
there is no live sensor loop in this challenge, the map is already known) and
the sensor model the accurate scanner uses. Return a list of (x, y) world-frame
stop poses, IN VISIT ORDER, such that a 360-degree rotate-and-scan from each
stop, together, observes as much of the wall/boundary as possible, using as
few stops and as little travel as you can.

See README.md for the full brief, scoring rubric, and how to run
`eval.py` against your solution.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from sim.map_io import FREE, OccupancyGrid
from sim.pathing import multi_target_shortest_paths, traversable_mask
from sim.visibility import SensorModel, observable_wall_cells, scan_from_stop


_ROBOT_RADIUS_M = 0.2
_MAX_REFINEMENT_CLUSTERS = 12
_WALL_STANDOFFS_M = (0.75, 1.5, 2.5, 3.5, 4.5)
_WALL_LATERAL_OFFSETS_M = (-0.3, 0.0, 0.3)
_PLATEAU_WARMUP_STOPS = 5
_PLATEAU_WINDOW_STOPS = 5
_PLATEAU_GAIN_FRACTION = 0.1

_LAST_CANDIDATE_DEBUG: dict[str, list[tuple[int, int]] | np.ndarray] = {}


def get_last_candidate_debug() -> dict[str, list[tuple[int, int]] | np.ndarray]:
    """Return copies of the most recently generated candidate categories."""
    return {name: value.copy() for name, value in _LAST_CANDIDATE_DEBUG.items()}


def _exterior_mask(valid_mask: np.ndarray) -> np.ndarray:
    """Return clearance-safe cells connected to the map boundary.

    Floor-plan images commonly use FREE for the white background outside the
    building.  It is physically clearance-safe but is not a valid operating
    area, so discard every valid cell reachable from the image border.  Use
    8-connectivity to match the path planner's movement model.
    """
    height, width = valid_mask.shape
    exterior = np.zeros_like(valid_mask, dtype=bool)
    frontier: deque[tuple[int, int]] = deque()

    def add_if_valid(row: int, col: int) -> None:
        if valid_mask[row, col] and not exterior[row, col]:
            exterior[row, col] = True
            frontier.append((row, col))

    for row in range(height):
        add_if_valid(row, 0)
        add_if_valid(row, width - 1)
    for col in range(width):
        add_if_valid(0, col)
        add_if_valid(height - 1, col)

    while frontier:
        row, col = frontier.popleft()
        for row_offset in (-1, 0, 1):
            for col_offset in (-1, 0, 1):
                if row_offset == 0 and col_offset == 0:
                    continue
                neighbor_row = row + row_offset
                neighbor_col = col + col_offset
                if not (0 <= neighbor_row < height and 0 <= neighbor_col < width):
                    continue
                add_if_valid(neighbor_row, neighbor_col)

    return exterior


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Keep the largest 8-connected traversable region in ``mask``.

    The challenge API provides no robot start pose.  We therefore use the
    largest enclosed, robot-traversable region as the deterministic operating
    area and reject sealed rooms and exterior pockets behind robot-sized
    bottlenecks.
    """
    height, width = mask.shape
    labels = np.zeros(mask.shape, dtype=np.int32)
    component_id = 0
    largest_component_id = 0
    largest_component_size = 0

    for row in range(height):
        for col in range(width):
            if not mask[row, col] or labels[row, col] != 0:
                continue

            component_id += 1
            component_size = 0
            frontier: deque[tuple[int, int]] = deque([(row, col)])
            labels[row, col] = component_id

            while frontier:
                current_row, current_col = frontier.popleft()
                component_size += 1
                for row_offset in (-1, 0, 1):
                    for col_offset in (-1, 0, 1):
                        if row_offset == 0 and col_offset == 0:
                            continue
                        neighbor_row = current_row + row_offset
                        neighbor_col = current_col + col_offset
                        if not (0 <= neighbor_row < height and 0 <= neighbor_col < width):
                            continue
                        if mask[neighbor_row, neighbor_col] and labels[neighbor_row, neighbor_col] == 0:
                            labels[neighbor_row, neighbor_col] = component_id
                            frontier.append((neighbor_row, neighbor_col))

            if component_size > largest_component_size:
                largest_component_id = component_id
                largest_component_size = component_size

    return labels == largest_component_id


def _distance_ridge_candidates(grid: OccupancyGrid, valid_mask: np.ndarray,
                               coverage_range_m: float) -> list[tuple[int, int]]:
    """Sample high-clearance room centres and corridor centre lines.

    A multi-source grid distance transform is sufficient here: the local
    distance ridges approximate a medial-axis skeleton without introducing a
    new dependency. Long ridge components are sampled only when their extent
    exceeds the sensor's useful coverage range, rather than on an arbitrary
    map-grid spacing.
    """
    height, width = valid_mask.shape
    distance = np.full(valid_mask.shape, -1, dtype=np.int32)
    frontier: deque[tuple[int, int]] = deque()
    for row, col in np.argwhere(valid_mask):
        row, col = int(row), int(col)
        for row_offset in (-1, 0, 1):
            for col_offset in (-1, 0, 1):
                neighbor_row, neighbor_col = row + row_offset, col + col_offset
                if not (0 <= neighbor_row < height and 0 <= neighbor_col < width) or not valid_mask[neighbor_row, neighbor_col]:
                    distance[row, col] = 0
                    frontier.append((row, col))
                    break
            if distance[row, col] == 0:
                break

    while frontier:
        row, col = frontier.popleft()
        for row_offset in (-1, 0, 1):
            for col_offset in (-1, 0, 1):
                if row_offset == 0 and col_offset == 0:
                    continue
                neighbor_row, neighbor_col = row + row_offset, col + col_offset
                if (0 <= neighbor_row < height and 0 <= neighbor_col < width
                        and valid_mask[neighbor_row, neighbor_col]
                        and distance[neighbor_row, neighbor_col] == -1):
                    distance[neighbor_row, neighbor_col] = distance[row, col] + 1
                    frontier.append((neighbor_row, neighbor_col))

    ridge = np.zeros_like(valid_mask, dtype=bool)
    for row, col in np.argwhere(valid_mask):
        row, col = int(row), int(col)
        value = distance[row, col]
        ridge[row, col] = all(
            not (0 <= row + row_offset < height and 0 <= col + col_offset < width)
            or distance[row + row_offset, col + col_offset] <= value
            for row_offset in (-1, 0, 1)
            for col_offset in (-1, 0, 1)
            if row_offset != 0 or col_offset != 0
        )

    candidates: list[tuple[int, int]] = []
    visited = np.zeros_like(ridge, dtype=bool)
    coverage_range_px = coverage_range_m / grid.resolution
    for row, col in np.argwhere(ridge):
        row, col = int(row), int(col)
        if visited[row, col]:
            continue
        component: list[tuple[int, int]] = []
        frontier: deque[tuple[int, int]] = deque([(row, col)])
        visited[row, col] = True
        while frontier:
            current_row, current_col = frontier.popleft()
            component.append((current_row, current_col))
            for row_offset in (-1, 0, 1):
                for col_offset in (-1, 0, 1):
                    if row_offset == 0 and col_offset == 0:
                        continue
                    neighbor_row, neighbor_col = current_row + row_offset, current_col + col_offset
                    if (0 <= neighbor_row < height and 0 <= neighbor_col < width
                            and ridge[neighbor_row, neighbor_col] and not visited[neighbor_row, neighbor_col]):
                        visited[neighbor_row, neighbor_col] = True
                        frontier.append((neighbor_row, neighbor_col))

        selected = [max(component, key=lambda point: (distance[point], -point[0], -point[1]))]
        while True:
            candidate = max(
                component,
                key=lambda point: (
                    min((point[0] - chosen[0]) ** 2 + (point[1] - chosen[1]) ** 2 for chosen in selected),
                    distance[point], -point[0], -point[1],
                ),
            )
            nearest_distance_px = min(
                np.hypot(candidate[0] - chosen[0], candidate[1] - chosen[1])
                for chosen in selected
            )
            if nearest_distance_px <= coverage_range_px:
                break
            selected.append(candidate)
        candidates.extend(selected)
    return candidates


def _thin_topology(binary: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning for a small, topology-scale free-space raster."""
    image = binary.astype(np.uint8).copy()
    while True:
        changed = False
        for phase in (0, 1):
            padded = np.pad(image, 1, mode="constant", constant_values=0)
            neighbors = [
                padded[:-2, 1:-1], padded[:-2, 2:], padded[1:-1, 2:], padded[2:, 2:],
                padded[2:, 1:-1], padded[2:, :-2], padded[1:-1, :-2], padded[:-2, :-2],
            ]
            neighbor_count = sum(neighbors)
            transitions = sum(
                (neighbors[index] == 0) & (neighbors[(index + 1) % 8] == 1)
                for index in range(8)
            )
            if phase == 0:
                preserve = (neighbors[0] * neighbors[2] * neighbors[4] == 0) & (
                    neighbors[2] * neighbors[4] * neighbors[6] == 0
                )
            else:
                preserve = (neighbors[0] * neighbors[2] * neighbors[6] == 0) & (
                    neighbors[0] * neighbors[4] * neighbors[6] == 0
                )
            remove = (image == 1) & (neighbor_count >= 2) & (neighbor_count <= 6) & (transitions == 1) & preserve
            if remove.any():
                image[remove] = 0
                changed = True
        if not changed:
            return image.astype(bool)


def _topology_candidates(grid: OccupancyGrid, valid_mask: np.ndarray,
                         coverage_range_m: float) -> list[tuple[int, int]]:
    """Generate room/corridor features from a simplified skeleton graph."""
    # Use a topology map at one quarter of the robot radius (5 cm for the
    # challenge robot), removing pixel-scale wall noise before skeletonization.
    factor = max(1, int(round((_ROBOT_RADIUS_M / 4) / grid.resolution)))
    padded_height = int(np.ceil(grid.height / factor)) * factor
    padded_width = int(np.ceil(grid.width / factor)) * factor
    padded = np.pad(valid_mask, ((0, padded_height - grid.height), (0, padded_width - grid.width)))
    topology_free = padded.reshape(padded_height // factor, factor, padded_width // factor, factor).any(axis=(1, 3))
    skeleton = _thin_topology(topology_free)
    height, width = skeleton.shape

    padded_skeleton = np.pad(skeleton, 1, mode="constant", constant_values=False)
    degree = np.zeros_like(skeleton, dtype=np.int8)
    for row_offset in (-1, 0, 1):
        for col_offset in (-1, 0, 1):
            if row_offset != 0 or col_offset != 0:
                degree += padded_skeleton[
                    1 + row_offset:1 + row_offset + height,
                    1 + col_offset:1 + col_offset + width,
                ]
    node_mask = skeleton & (degree != 2)
    node_points = {tuple(map(int, point)) for point in np.argwhere(node_mask)}
    if not node_points and skeleton.any():
        node_points.add(tuple(map(int, np.argwhere(skeleton)[0])))

    def neighbors(point: tuple[int, int]) -> list[tuple[int, int]]:
        row, col = point
        return [
            (row + row_offset, col + col_offset)
            for row_offset in (-1, 0, 1)
            for col_offset in (-1, 0, 1)
            if (row_offset != 0 or col_offset != 0)
            and 0 <= row + row_offset < height and 0 <= col + col_offset < width
            and skeleton[row + row_offset, col + col_offset]
        ]

    def edge_key(first: tuple[int, int], second: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
        return (first, second) if first <= second else (second, first)

    feature_points = set(node_points)
    visited_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    topology_resolution = factor * grid.resolution
    for start in sorted(node_points):
        for first_neighbor in neighbors(start):
            if edge_key(start, first_neighbor) in visited_edges:
                continue
            path = [start]
            previous, current = start, first_neighbor
            while True:
                visited_edges.add(edge_key(previous, current))
                path.append(current)
                if current in node_points:
                    break
                next_points = [point for point in neighbors(current) if point != previous]
                if not next_points:
                    break
                previous, current = current, next_points[0]

            segment_length = sum(
                np.hypot(path[index + 1][0] - path[index][0], path[index + 1][1] - path[index][1])
                for index in range(len(path) - 1)
            ) * topology_resolution
            segment_count = int(np.ceil(segment_length / coverage_range_m))
            for index in range(1, segment_count):
                target_length = segment_length * index / segment_count
                accumulated = 0.0
                for path_index in range(len(path) - 1):
                    step_length = np.hypot(
                        path[path_index + 1][0] - path[path_index][0],
                        path[path_index + 1][1] - path[path_index][1],
                    ) * topology_resolution
                    accumulated += step_length
                    if accumulated >= target_length:
                        feature_points.add(path[path_index + 1])
                        break

    candidates: list[tuple[int, int]] = []
    for topology_row, topology_col in sorted(feature_points):
        row_start, col_start = topology_row * factor, topology_col * factor
        row_end = min(row_start + factor, grid.height)
        col_end = min(col_start + factor, grid.width)
        valid_points = np.argwhere(valid_mask[row_start:row_end, col_start:col_end])
        if valid_points.size == 0:
            continue
        center_row = (row_start + row_end - 1) / 2
        center_col = (col_start + col_end - 1) / 2
        rows, cols = valid_points[:, 0] + row_start, valid_points[:, 1] + col_start
        best = int(np.argmin((rows - center_row) ** 2 + (cols - center_col) ** 2))
        candidates.append((int(rows[best]), int(cols[best])))
    return candidates


def _cluster_representatives(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return wall-cell representatives of the largest 8-connected gaps."""
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    clusters: list[tuple[int, int, int, int]] = []  # size, first row, first col, packed centroid
    for row, col in np.argwhere(mask):
        row, col = int(row), int(col)
        if visited[row, col]:
            continue
        frontier: deque[tuple[int, int]] = deque([(row, col)])
        visited[row, col] = True
        size = row_sum = col_sum = 0
        first_row, first_col = row, col
        points: list[tuple[int, int]] = []
        while frontier:
            current_row, current_col = frontier.popleft()
            points.append((current_row, current_col))
            size += 1
            row_sum += current_row
            col_sum += current_col
            for row_offset in (-1, 0, 1):
                for col_offset in (-1, 0, 1):
                    if row_offset == 0 and col_offset == 0:
                        continue
                    neighbor_row, neighbor_col = current_row + row_offset, current_col + col_offset
                    if (0 <= neighbor_row < height and 0 <= neighbor_col < width
                            and mask[neighbor_row, neighbor_col] and not visited[neighbor_row, neighbor_col]):
                        visited[neighbor_row, neighbor_col] = True
                        frontier.append((neighbor_row, neighbor_col))
        centroid_row = row_sum / size
        centroid_col = col_sum / size
        representative_row, representative_col = min(
            points,
            key=lambda point: ((point[0] - centroid_row) ** 2 + (point[1] - centroid_col) ** 2, point),
        )
        clusters.append((size, first_row, first_col, representative_row * width + representative_col))

    clusters.sort(key=lambda cluster: (-cluster[0], cluster[1], cluster[2]))
    return [(packed // width, packed % width) for _, _, _, packed in clusters[:_MAX_REFINEMENT_CLUSTERS]]


def _wall_face_directions(grid: OccupancyGrid, row: int, col: int) -> list[tuple[int, int]]:
    """Return directions from a wall cell toward its adjacent free-space faces."""
    directions: list[tuple[int, int]] = []
    for row_offset in (-1, 0, 1):
        for col_offset in (-1, 0, 1):
            if row_offset == 0 and col_offset == 0:
                continue
            neighbor_row, neighbor_col = row + row_offset, col + col_offset
            if grid.in_bounds(neighbor_row, neighbor_col) and grid.data[neighbor_row, neighbor_col] == FREE:
                directions.append((row_offset, col_offset))
    return directions


def _refinement_candidates(grid: OccupancyGrid, valid_mask: np.ndarray,
                           gap_mask: np.ndarray, existing: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Generate head-on candidate views for the largest uncovered wall gaps."""
    candidates: list[tuple[int, int]] = []
    for wall_row, wall_col in _cluster_representatives(gap_mask):
        for normal_row, normal_col in _wall_face_directions(grid, wall_row, wall_col):
            normal_length = np.hypot(normal_row, normal_col)
            unit_row, unit_col = normal_row / normal_length, normal_col / normal_length
            lateral_row, lateral_col = -unit_col, unit_row
            for standoff_m in _WALL_STANDOFFS_M:
                standoff_px = standoff_m / grid.resolution
                for lateral_offset_m in _WALL_LATERAL_OFFSETS_M:
                    lateral_px = lateral_offset_m / grid.resolution
                    row = int(round(wall_row + unit_row * standoff_px + lateral_row * lateral_px))
                    col = int(round(wall_col + unit_col * standoff_px + lateral_col * lateral_px))
                    if not (0 <= row < grid.height and 0 <= col < grid.width) or not valid_mask[row, col]:
                        continue
                    point = (row, col)
                    if point not in existing:
                        existing.add(point)
                        candidates.append(point)
    return candidates


def _route_distance_matrix(
    grid: OccupancyGrid,
    stops: list[tuple[float, float]],
    traversable: np.ndarray,
) -> np.ndarray:
    """Compute clearance-safe travel distances between all selected stops."""
    count = len(stops)
    distances = np.zeros((count, count), dtype=float)
    progress_interval = max(1, count // 10)
    print("[planner] computing clearance-safe route distances", flush=True)
    for source_index, source_stop in enumerate(stops):
        paths = multi_target_shortest_paths(grid, source_stop, stops, traversable)
        for target_index, (distance_m, _) in enumerate(paths):
            distances[source_index, target_index] = distance_m
        if (source_index == 0 or source_index + 1 == count
                or (source_index + 1) % progress_interval == 0):
            print(f"[planner] route distances: {source_index + 1}/{count}", flush=True)
    return distances


def _route_length(route: list[int], distances: np.ndarray) -> float:
    return sum(distances[route[index], route[index + 1]] for index in range(len(route) - 1))


def _best_nearest_neighbor_route(distances: np.ndarray) -> list[int]:
    """Try every start and retain the shortest deterministic NN open route."""
    count = len(distances)
    best_route: list[int] | None = None
    best_length = float("inf")
    for start in range(count):
        route = [start]
        unvisited = set(range(count))
        unvisited.remove(start)
        while unvisited:
            current = route[-1]
            next_stop = min(unvisited, key=lambda index: (distances[current, index], index))
            route.append(next_stop)
            unvisited.remove(next_stop)
        length = _route_length(route, distances)
        if length < best_length:
            best_route = route
            best_length = length
    return best_route or []


def _two_opt(route: list[int], distances: np.ndarray) -> tuple[list[int], int]:
    """Apply deterministic first-improvement 2-opt to an open route."""
    route = route.copy()
    improvements = 0
    while True:
        improved = False
        for first in range(len(route) - 1):
            for last in range(first + 1, len(route)):
                before = distances[route[first - 1], route[first]] if first else 0.0
                after = distances[route[last], route[last + 1]] if last + 1 < len(route) else 0.0
                existing_cost = before + after
                replacement_before = distances[route[first - 1], route[last]] if first else 0.0
                replacement_after = distances[route[first], route[last + 1]] if last + 1 < len(route) else 0.0
                replacement_cost = replacement_before + replacement_after
                if replacement_cost + 1e-9 < existing_cost:
                    route[first:last + 1] = reversed(route[first:last + 1])
                    improvements += 1
                    improved = True
                    break
            if improved:
                break
        if not improved:
            return route, improvements


def _scan_candidates(grid: OccupancyGrid, sensor: SensorModel,
                     candidate_pixels: list[tuple[int, int]], observable: np.ndarray,
                     label: str) -> tuple[list[tuple[float, float]], list[set[tuple[int, int]]]]:
    """Raycast candidates once and retain only scorer-qualifying wall cells."""
    stops = [grid.pixel_to_world(row, col) for row, col in candidate_pixels]
    coverage: list[set[tuple[int, int]]] = []
    interval = max(1, len(stops) // 10)
    for index, stop in enumerate(stops, start=1):
        scan = scan_from_stop(grid, stop, sensor)
        coverage.append({
            cell for cell, quality in scan.items()
            if quality >= sensor.min_quality and observable[cell]
        })
        if index == 1 or index == len(stops) or index % interval == 0:
            print(f"[planner] raycast {label}: {index}/{len(stops)}", flush=True)
    return stops, coverage


def _greedy_selection(candidate_coverage: list[set[tuple[int, int]],], total_targets: int) -> tuple[list[int], set[tuple[int, int]]]:
    """Run greedy set cover until the marginal-coverage plateau."""
    covered: set[tuple[int, int]] = set()
    selected_indices: set[int] = set()
    selection_order: list[int] = []
    marginal_gains: list[int] = []
    covering_candidates: dict[tuple[int, int], list[int]] = {}
    for candidate_index, coverage in enumerate(candidate_coverage):
        for target in coverage:
            covering_candidates.setdefault(target, []).append(candidate_index)
    marginal_gain = [len(coverage) for coverage in candidate_coverage]
    print("[planner] selecting stops greedily", flush=True)

    while True:
        best_index: int | None = None
        best_gain = 0
        for index, gain in enumerate(marginal_gain):
            if index not in selected_indices and gain > best_gain:
                best_index, best_gain = index, gain
        if best_index is None:
            break

        selected_indices.add(best_index)
        selection_order.append(best_index)
        marginal_gains.append(best_gain)
        newly_covered = candidate_coverage[best_index] - covered
        covered.update(newly_covered)
        for target in newly_covered:
            for candidate_index in covering_candidates[target]:
                if candidate_index not in selected_indices:
                    marginal_gain[candidate_index] -= 1
        if len(selection_order) == 1 or len(selection_order) % 50 == 0:
            print(f"[planner] selected {len(selection_order)} stops; covered {len(covered)}/{total_targets} wall cells", flush=True)

        if len(marginal_gains) >= _PLATEAU_WARMUP_STOPS + _PLATEAU_WINDOW_STOPS:
            reference = sum(marginal_gains[:_PLATEAU_WARMUP_STOPS]) / _PLATEAU_WARMUP_STOPS
            recent = sum(marginal_gains[-_PLATEAU_WINDOW_STOPS:]) / _PLATEAU_WINDOW_STOPS
            if recent < _PLATEAU_GAIN_FRACTION * reference:
                print(
                    f"[planner] coverage plateau after {len(selection_order)} stops: "
                    f"recent gain {recent:.1f} cells/stop is below "
                    f"{_PLATEAU_GAIN_FRACTION:.0%} of initial gain {reference:.1f}",
                    flush=True,
                )
                break
    return selection_order, covered


def plan_viewpoints(grid: OccupancyGrid, sensor: SensorModel) -> list[tuple[float, float]]:
    """Plan stops from floor-plan topology and visibility-driven refinement."""
    clearance_safe_mask = traversable_mask(grid, _ROBOT_RADIUS_M)
    exterior_mask = _exterior_mask(clearance_safe_mask)
    enclosed_mask = clearance_safe_mask & ~exterior_mask
    valid_mask = _largest_connected_component(enclosed_mask)
    observable = observable_wall_cells(grid)

    # Skeleton graph features supply room centres, corridor junctions and
    # sensor-range samples on long corridors without uniform map sampling.
    effective_range_m = sensor.max_range_m * np.sqrt(1.0 - sensor.min_quality)
    topology_pixels = _topology_candidates(grid, valid_mask, effective_range_m)
    candidate_pixels = topology_pixels.copy()
    existing_pixels = set(candidate_pixels)
    print(
        f"[planner] largest enclosed component has {int(valid_mask.sum())} cells; "
        f"{len(candidate_pixels)} skeleton-topology candidates "
        f"(effective range {effective_range_m:.2f} m) for "
        f"{int(observable.sum())} observable wall cells",
        flush=True,
    )

    candidate_stops, candidate_coverage = _scan_candidates(
        grid, sensor, candidate_pixels, observable, "initial candidates",
    )
    _, initial_covered = _greedy_selection(candidate_coverage, int(observable.sum()))

    # Only refine targets already known to be observable from the operating
    # region. This avoids spending work on exterior or sealed-room wall faces.
    initially_attainable = set().union(*candidate_coverage) if candidate_coverage else set()
    gap_mask = np.zeros_like(observable, dtype=bool)
    for row, col in initially_attainable - initial_covered:
        gap_mask[row, col] = True
    refinement_pixels = _refinement_candidates(grid, valid_mask, gap_mask, existing_pixels)
    if refinement_pixels:
        print(f"[planner] adding {len(refinement_pixels)} wall-normal candidates for coverage gaps", flush=True)
        refinement_stops, refinement_coverage = _scan_candidates(
            grid, sensor, refinement_pixels, observable, "refinement candidates",
        )
        candidate_pixels.extend(refinement_pixels)
        candidate_stops.extend(refinement_stops)
        candidate_coverage.extend(refinement_coverage)

    global _LAST_CANDIDATE_DEBUG
    _LAST_CANDIDATE_DEBUG = {
        "topology": topology_pixels,
        "refinement": refinement_pixels,
        "all": candidate_pixels.copy(),
        "clearance_safe": clearance_safe_mask.copy(),
        "exterior": exterior_mask.copy(),
        "enclosed": enclosed_mask.copy(),
        "operating": valid_mask.copy(),
    }

    selection_order, covered = _greedy_selection(candidate_coverage, int(observable.sum()))

    # A later greedy choice can make an earlier stop redundant.  Reverse
    # deletion keeps the same union of wall cells while removing every stop
    # whose complete contribution is duplicated by the remaining plan.
    target_counts: dict[tuple[int, int], int] = {}
    for candidate_index in selection_order:
        for target in candidate_coverage[candidate_index]:
            target_counts[target] = target_counts.get(target, 0) + 1

    retained_indices = set(selection_order)
    for candidate_index in reversed(selection_order):
        if all(target_counts[target] > 1 for target in candidate_coverage[candidate_index]):
            retained_indices.remove(candidate_index)
            for target in candidate_coverage[candidate_index]:
                target_counts[target] -= 1

    removed_count = len(selection_order) - len(retained_indices)
    selected = [
        candidate_stops[candidate_index]
        for candidate_index in selection_order
        if candidate_index in retained_indices
    ]
    print(
        f"[planner] pruned {removed_count} redundant stops; "
        f"finished with {len(selected)} stops; "
        f"covered {len(covered)}/{int(observable.sum())} wall cells",
        flush=True,
    )

    if len(selected) > 1:
        route_distances = _route_distance_matrix(grid, selected, valid_mask)
        route = _best_nearest_neighbor_route(route_distances)
        nearest_neighbor_length = _route_length(route, route_distances)
        route, two_opt_improvements = _two_opt(route, route_distances)
        optimized_length = _route_length(route, route_distances)
        selected = [selected[index] for index in route]
        print(
            f"[planner] route optimization: {nearest_neighbor_length:.1f} m -> "
            f"{optimized_length:.1f} m with {two_opt_improvements} 2-opt improvements",
            flush=True,
        )
    return selected
