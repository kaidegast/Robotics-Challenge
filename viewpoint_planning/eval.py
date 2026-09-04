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
        grid, stops, sensor, robot_radius_m=ROBOT_RADIUS_M, progress=True,
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

    if not args.no_viz:
        render_report(grid, stops, report, str(png_path))
        print(f"Visualization:    {png_path}")


if __name__ == "__main__":
    main()
