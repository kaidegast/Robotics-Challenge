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

from sim.map_io import OccupancyGrid
from sim.pathing import multi_target_shortest_paths, traversable_mask
from sim.visibility import SensorModel, observable_wall_cells, scan_from_stop


_ROBOT_RADIUS_M = 0.2
_CANDIDATE_SPACING_M = 0.45
_PLATEAU_WARMUP_STOPS = 5
_PLATEAU_WINDOW_STOPS = 5
_PLATEAU_GAIN_FRACTION = 0.1


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


def _tile_candidates(grid: OccupancyGrid, valid_mask: np.ndarray) -> list[tuple[int, int]]:
    """Pick one clearance-safe cell nearest the centre of each non-empty tile.

    Choosing from each tile rather than requiring the lattice intersection
    itself to be valid avoids silently skipping narrow corridors when the
    lattice happens to land on a wall.
    """
    tile_size_px = max(1, int(round(_CANDIDATE_SPACING_M / grid.resolution)))
    candidates: list[tuple[int, int]] = []

    for row_start in range(0, grid.height, tile_size_px):
        row_end = min(row_start + tile_size_px, grid.height)
        for col_start in range(0, grid.width, tile_size_px):
            col_end = min(col_start + tile_size_px, grid.width)
            valid_in_tile = np.argwhere(valid_mask[row_start:row_end, col_start:col_end])
            if valid_in_tile.size == 0:
                continue

            center_row = (row_start + row_end - 1) / 2
            center_col = (col_start + col_end - 1) / 2
            rows = valid_in_tile[:, 0] + row_start
            cols = valid_in_tile[:, 1] + col_start
            squared_distance = (rows - center_row) ** 2 + (cols - center_col) ** 2
            # np.argmin keeps the first tied element. argwhere is row-major,
            # yielding a stable (row, col) tie break.
            best = int(np.argmin(squared_distance))
            candidates.append((int(rows[best]), int(cols[best])))

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


def plan_viewpoints(grid: OccupancyGrid, sensor: SensorModel) -> list[tuple[float, float]]:
    """Select clearance-safe scan stops with deterministic greedy set cover.

    A candidate's set contains the observable wall cells it scans at the
    scorer's required quality.  Repeatedly selecting the candidate with the
    greatest marginal set size maximizes coverage greedily. The selected
    stops are then reordered into a short clearance-safe open route.
    """
    clearance_safe_mask = traversable_mask(grid, _ROBOT_RADIUS_M)
    # The free background outside a closed floor plan can also satisfy the
    # clearance test. Keep only enclosed clearance-safe regions for stops.
    enclosed_mask = clearance_safe_mask & ~_exterior_mask(clearance_safe_mask)
    valid_mask = _largest_connected_component(enclosed_mask)
    candidate_pixels = _tile_candidates(grid, valid_mask)
    observable = observable_wall_cells(grid)
    print(
        f"[planner] largest enclosed component has {int(valid_mask.sum())} cells; "
        f"{len(candidate_pixels)} interior candidates for "
        f"{int(observable.sum())} observable wall cells",
        flush=True,
    )

    candidate_stops = [grid.pixel_to_world(row, col) for row, col in candidate_pixels]
    candidate_coverage: list[set[tuple[int, int]]] = []
    scan_progress_interval = max(1, len(candidate_stops) // 10)
    for index, stop_xy in enumerate(candidate_stops, start=1):
        scan = scan_from_stop(grid, stop_xy, sensor)
        candidate_coverage.append({
            cell for cell, quality in scan.items()
            if quality >= sensor.min_quality and observable[cell]
        })
        if index == 1 or index == len(candidate_stops) or index % scan_progress_interval == 0:
            print(f"[planner] raycast candidates: {index}/{len(candidate_stops)}", flush=True)

    selected: list[tuple[float, float]] = []
    covered: set[tuple[int, int]] = set()
    selected_indices: set[int] = set()
    selection_order: list[int] = []
    marginal_gains: list[int] = []

    # Maintain each candidate's exact marginal gain incrementally.  Recomputing
    # ``len(coverage - covered)`` for every candidate after every selection is
    # correct but dominates runtime on the larger maps.
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
            if index in selected_indices:
                continue
            if gain > best_gain:
                best_index = index
                best_gain = gain

        if best_index is None:
            break

        selected_indices.add(best_index)
        selection_order.append(best_index)
        marginal_gains.append(best_gain)
        newly_covered = candidate_coverage[best_index] - covered
        covered.update(newly_covered)
        selected.append(candidate_stops[best_index])
        for target in newly_covered:
            for candidate_index in covering_candidates[target]:
                if candidate_index not in selected_indices:
                    marginal_gain[candidate_index] -= 1
        if len(selected) == 1 or len(selected) % 50 == 0:
            print(
                f"[planner] selected {len(selected)} stops; "
                f"covered {len(covered)}/{int(observable.sum())} wall cells",
                flush=True,
            )

        # A coverage percentage is not meaningful here because the scorer's
        # denominator includes inaccessible wall faces. Instead, stop once
        # recent stops consistently add only a small fraction of the useful
        # coverage added at the start of the greedy run.
        enough_history = len(marginal_gains) >= _PLATEAU_WARMUP_STOPS + _PLATEAU_WINDOW_STOPS
        if enough_history:
            reference_gain = sum(marginal_gains[:_PLATEAU_WARMUP_STOPS]) / _PLATEAU_WARMUP_STOPS
            recent_gain = sum(marginal_gains[-_PLATEAU_WINDOW_STOPS:]) / _PLATEAU_WINDOW_STOPS
            if recent_gain < _PLATEAU_GAIN_FRACTION * reference_gain:
                print(
                    f"[planner] coverage plateau after {len(selected)} stops: "
                    f"recent gain {recent_gain:.1f} cells/stop is below "
                    f"{_PLATEAU_GAIN_FRACTION:.0%} of initial gain {reference_gain:.1f}",
                    flush=True,
                )
                break

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
