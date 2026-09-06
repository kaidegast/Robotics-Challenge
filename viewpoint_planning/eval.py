#!/usr/bin/env python3
"""CLI entry point: load a map, run a candidate's plan_viewpoints(), score it,
print a report, and (unless --no-viz) save a coverage visualization PNG.

Usage:
    python eval.py --map maps/room_a/room.yaml --solution candidate_solution.solution
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sim.map_io import load_occupancy_grid
from sim.scorer import render_report, score_solution
from sim.visibility import SensorModel

# Fixed for every candidate/map so scores are directly comparable - not CLI flags.
MAX_RANGE_M = 8.0
MIN_QUALITY = 0.5
ROBOT_RADIUS_M = 0.2


def render_candidate_debug(grid, candidates: dict[str, object], out_path: Path) -> None:
    """Render the complete candidate pool exposed by a solution module."""
    import matplotlib.pyplot as plt
    import numpy as np

    from sim.map_io import FREE, OCCUPIED, UNKNOWN

    rgb = np.zeros((grid.height, grid.width, 3), dtype=np.uint8)
    rgb[grid.data == FREE] = (255, 255, 255)
    rgb[grid.data == OCCUPIED] = (45, 45, 45)
    rgb[grid.data == UNKNOWN] = (205, 205, 205)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb)

    styles = (
        ("topology", "#1f77b4", "skeleton topology"),
        ("refinement", "#d62728", "wall-normal refinement"),
    )
    for category, color, label in styles:
        pixels = candidates.get(category, [])
        if isinstance(pixels, list) and pixels:
            ax.scatter([col for _, col in pixels], [row for row, _ in pixels],
                       s=9, c=color, label=f"{label} ({len(pixels)})", alpha=0.75)
    all_candidates = candidates.get("all", [])
    total = len(all_candidates) if isinstance(all_candidates, list) else 0
    ax.set_title(f"Candidate viewpoints: {total} total")
    ax.legend(loc="upper right", fontsize=8)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def render_area_debug(grid, debug: dict[str, object], out_path: Path) -> None:
    """Render the operating-area classification used by the candidate planner."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    def mask(name: str) -> np.ndarray:
        value = debug.get(name)
        if not isinstance(value, np.ndarray):
            raise ValueError(f"Candidate debug data has no {name!r} mask")
        return value

    clearance_safe = mask("clearance_safe")
    exterior = mask("exterior")
    enclosed = mask("enclosed")
    operating = mask("operating")
    rgb = np.zeros((grid.height, grid.width, 3), dtype=np.uint8)
    rgb[grid.data == 0] = (232, 232, 232)  # free, but no footprint clearance
    rgb[grid.data == 1] = (45, 45, 45)
    rgb[grid.data == 2] = (160, 160, 160)
    rgb[clearance_safe] = (195, 195, 195)
    rgb[exterior] = (247, 196, 75)
    rgb[enclosed & ~operating] = (230, 126, 34)
    rgb[operating] = (46, 204, 113)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb)
    points = debug.get("all", [])
    if isinstance(points, list) and points:
        ax.scatter([col for _, col in points], [row for row, _ in points],
                   s=7, c="black", alpha=0.7, label=f"candidates ({len(points)})")
    legend = [
        Patch(color="#2ecc71", label="operating region / candidate area"),
        Patch(color="#e67e22", label="enclosed but disconnected"),
        Patch(color="#f7c44b", label="boundary-connected exterior"),
        Patch(color="#c3c3c3", label="clearance-safe, unclassified"),
        Patch(color="#e8e8e8", label="free but too close to wall"),
        Patch(color="#2d2d2d", label="wall"),
        Patch(color="#a0a0a0", label="unknown"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=7)
    ax.set_title("Robot operating-area classification")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True, help="Path to a room.yaml")
    parser.add_argument("--solution", default="candidate_solution.solution",
                         help="Python module exposing plan_viewpoints(grid, sensor)")
    parser.add_argument("--results-dir", default="results",
                         help="Parent directory for timestamped session output")
    parser.add_argument("--out", default=None,
                        help="Explicit PNG output path, overriding the timestamped results dir")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--debug-candidates", action="store_true",
                        help="Save a candidate_positions.png debug image alongside the report")
    parser.add_argument("--debug-areas", action="store_true",
                        help="Save an area_classification.png planner-mask debug image")
    parser.add_argument("--progress", action="store_true",
                        help="Print scorer progress while evaluating scans and route segments")
    args = parser.parse_args()

    grid = load_occupancy_grid(args.map)
    sensor = SensorModel(max_range_m=MAX_RANGE_M, min_quality=MIN_QUALITY)

    module = importlib.import_module(args.solution)
    print("[eval] planning viewpoints", flush=True)
    t0 = time.perf_counter()
    stops = module.plan_viewpoints(grid, sensor)
    elapsed = time.perf_counter() - t0

    if not stops:
        print("plan_viewpoints() returned no stops.")
        sys.exit(1)

    print("[eval] scoring coverage and tour", flush=True)
    report = score_solution(
        grid, stops, sensor, robot_radius_m=ROBOT_RADIUS_M, progress=args.progress,
    )

    print(f"Map:              {args.map}")
    print(f"Planning time:    {elapsed:.2f}s")
    print(report.summary())

    if args.out is not None:
        png_path = Path(args.out)
    else:
        session_dir = Path(args.results_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir.mkdir(parents=True, exist_ok=True)
        png_path = session_dir / "coverage_report.png"
    png_path.parent.mkdir(parents=True, exist_ok=True)

    json_path = png_path.with_suffix(".json")
    json_path.write_text(json.dumps(
        {"map": args.map, "planning_time_s": elapsed, **report.to_dict()}, indent=2))
    print(f"Report JSON:      {json_path}")

    if args.debug_candidates or args.debug_areas:
        debug_getter = getattr(module, "get_last_candidate_debug", None)
        if debug_getter is None:
            print("Debug images unavailable: solution does not expose candidate data.")
        else:
            debug_data = debug_getter()
            if args.debug_candidates:
                debug_path = png_path.with_name("candidate_positions.png")
                render_candidate_debug(grid, debug_data, debug_path)
                print(f"Candidate debug:  {debug_path}")
            if args.debug_areas:
                area_path = png_path.with_name("area_classification.png")
                render_area_debug(grid, debug_data, area_path)
                print(f"Area debug:       {area_path}")

    if not args.no_viz:
        render_report(grid, stops, report, str(png_path))
        print(f"Visualization:    {png_path}")


if __name__ == "__main__":
    main()
