"""Scores a candidate's chosen stops against the ground-truth map: wall
coverage %, stop count, tour length, and a validity check (stops must be in
free space, clear of walls by the robot's footprint radius)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .map_io import OccupancyGrid
from .pathing import shortest_path, traversable_mask
from .visibility import SensorModel, is_stop_valid, observable_wall_cells, scan_from_stop


@dataclass
class ScoreReport:
    coverage_fraction: float
    num_stops: int
    tour_length_m: float
    invalid_stops: list[tuple[float, float]] = field(default_factory=list)
    covered_cells: set[tuple[int, int]] = field(default_factory=set)
    total_wall_cells: int = 0
    unreachable_wall_cells: int = 0
    # (row, col) path per consecutive stop pair, through free space and clear
    # of walls by the robot's footprint; None for a pair the path search
    # couldn't route (e.g. an invalid stop), drawn as a straight line instead.
    path_segments: list[list[tuple[int, int]] | None] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Wall coverage:      {self.coverage_fraction * 100:.1f}%  "
            f"({len(self.covered_cells)}/{self.total_wall_cells} observable wall cells at required quality)",
            f"Stops used:         {self.num_stops}",
            f"Tour length:        {self.tour_length_m:.1f} m (through free space, not a straight line)",
        ]
        if self.unreachable_wall_cells:
            lines.append(
                f"Note:               {self.unreachable_wall_cells} wall cells excluded from coverage "
                f"denominator (buried inside wall thickness, unreachable by any ray)"
            )
        if self.invalid_stops:
            lines.append(
                f"INVALID stops (not in free space, or too close to a wall): {self.invalid_stops}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "coverage_fraction": self.coverage_fraction,
            "covered_wall_cells": len(self.covered_cells),
            "total_wall_cells": self.total_wall_cells,
            "unreachable_wall_cells": self.unreachable_wall_cells,
            "num_stops": self.num_stops,
            "tour_length_m": self.tour_length_m,
            "invalid_stops": [list(s) for s in self.invalid_stops],
        }


def score_solution(grid: OccupancyGrid, stops: list[tuple[float, float]],
                    sensor: SensorModel, robot_radius_m: float = 0.2,
                    progress: bool = False) -> ScoreReport:
    def log(message: str) -> None:
        if progress:
            print(f"[scorer] {message}", flush=True)

    invalid = [s for s in stops if not is_stop_valid(grid, s, robot_radius_m)]

    covered: dict[tuple[int, int], float] = {}
    score_progress_interval = max(1, len(stops) // 10)
    for index, stop in enumerate(stops, start=1):
        if stop in invalid:
            continue  # an invalid stop doesn't get to contribute a scan
        seen = scan_from_stop(grid, stop, sensor)
        for cell, quality in seen.items():
            if quality > covered.get(cell, 0.0):
                covered[cell] = quality
        if progress and (index == 1 or index == len(stops) or index % score_progress_interval == 0):
            log(f"scoring scans: {index}/{len(stops)}")

    total_occupied_cells = int((grid.data == 1).sum())
    total_wall_cells = int(observable_wall_cells(grid).sum())
    unreachable_wall_cells = total_occupied_cells - total_wall_cells
    covered_at_quality = {c for c, q in covered.items() if q >= sensor.min_quality}
    coverage_fraction = len(covered_at_quality) / total_wall_cells if total_wall_cells else 0.0

    traversable = traversable_mask(grid, robot_radius_m)
    tour_length = 0.0
    path_segments: list[list[tuple[int, int]] | None] = []
    path_count = len(stops) - 1
    path_progress_interval = max(1, path_count // 10)
    for i in range(path_count):
        length_m, path = shortest_path(grid, stops[i], stops[i + 1], traversable)
        tour_length += length_m
        path_segments.append(path)
        if progress and (i == 0 or i + 1 == path_count or (i + 1) % path_progress_interval == 0):
            log(f"routing tour segments: {i + 1}/{path_count}")

    return ScoreReport(
        coverage_fraction=coverage_fraction,
        num_stops=len(stops),
        tour_length_m=tour_length,
        invalid_stops=invalid,
        covered_cells=covered_at_quality,
        total_wall_cells=total_wall_cells,
        unreachable_wall_cells=unreachable_wall_cells,
        path_segments=path_segments,
    )


def render_report(grid: OccupancyGrid, stops: list[tuple[float, float]], report: ScoreReport,
                   out_path: str) -> None:
    """Save a PNG: map with wall cells colored green (covered) / red (missed but
    reachable) / dark gray (buried inside wall thickness, unreachable by any
    ray - excluded from the coverage score), and stop locations numbered in
    visit order. Requires matplotlib."""
    import matplotlib.pyplot as plt

    from .visibility import observable_wall_cells

    observable = observable_wall_cells(grid)

    rgb = np.zeros((grid.height, grid.width, 3), dtype=np.uint8)
    rgb[grid.data == 0] = (255, 255, 255)  # free
    rgb[grid.data == 2] = (205, 205, 205)  # unknown
    rgb[grid.data == 1] = (60, 60, 60)  # wall, default = unreachable interior (dark gray)
    rgb[(grid.data == 1) & observable] = (220, 60, 60)  # reachable but missed (red)
    for row, col in report.covered_cells:
        rgb[row, col] = (60, 180, 75)  # covered (green)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb)
    for i, (x, y) in enumerate(stops):
        row, col = grid.world_to_pixel(x, y)
        color = "black" if (x, y) not in report.invalid_stops else "orange"
        ax.plot(col, row, "o", color=color, markersize=8)
        ax.annotate(str(i + 1), (col, row), color="white", fontsize=8,
                    ha="center", va="center")
    for i, path in enumerate(report.path_segments):
        if path is not None:
            ax.plot([c for _, c in path], [r for r, _ in path], "b-", linewidth=1.5, alpha=0.6)
        else:
            # No free-space route (e.g. an invalid stop) - fall back to a
            # dashed straight line so the gap is still visible, but marked
            # as not a real drivable path.
            (x0, y0), (x1, y1) = stops[i], stops[i + 1]
            r0, c0 = grid.world_to_pixel(x0, y0)
            r1, c1 = grid.world_to_pixel(x1, y1)
            ax.plot([c0, c1], [r0, r1], "r--", linewidth=1, alpha=0.6)

    ax.set_title(
        f"Coverage {report.coverage_fraction * 100:.1f}%  |  "
        f"{report.num_stops} stops  |  {report.tour_length_m:.1f} m tour (free-space path)"
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
