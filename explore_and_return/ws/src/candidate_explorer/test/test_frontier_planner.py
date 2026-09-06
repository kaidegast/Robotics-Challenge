from pathlib import Path
from types import SimpleNamespace
import math
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from candidate_explorer.frontier_planner import (
    ExclusionZone,
    FrontierTarget,
    PlannerConfig,
    _coarsen_map,
    _safe_free_mask,
    is_excluded,
    plan_frontiers,
    rank_frontier_targets,
    raycast_information_gain_m2,
    select_frontier_target,
)


def make_map(data: np.ndarray, resolution: float = 0.06, origin=(0.0, 0.0, 0.0)):
    height, width = data.shape
    yaw = origin[2]
    return SimpleNamespace(
        data=data.flatten().tolist(),
        info=SimpleNamespace(
            width=width,
            height=height,
            resolution=resolution,
            origin=SimpleNamespace(
                position=SimpleNamespace(x=origin[0], y=origin[1]),
                orientation=SimpleNamespace(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2)),
            ),
        ),
    )


def test_coarsening_keeps_unknown_and_occupied_conservative():
    raw = np.zeros((4, 4), dtype=np.int16)
    raw[:2, :2] = -1
    raw[3, 3] = 100
    free, occupied, _ = _coarsen_map(make_map(raw, resolution=0.03), PlannerConfig(planning_resolution_m=0.06))
    assert not free[0, 0]
    assert occupied[1, 1]


def test_rotated_grid_geometry_round_trips_world_cells():
    raw = np.zeros((5, 5), dtype=np.int16)
    _, _, geometry = _coarsen_map(make_map(raw, origin=(3.0, -2.0, math.pi / 2)), PlannerConfig())
    point = geometry.cell_to_world(2, 3)
    assert geometry.world_to_cell(*point) == (2, 3)


def test_clearance_excludes_cells_close_to_obstacles():
    free = np.ones((9, 9), dtype=bool)
    occupied = np.zeros((9, 9), dtype=bool)
    occupied[4, 4] = True
    safe = _safe_free_mask(free, occupied, resolution=0.1, clearance_radius_m=0.21)
    assert not safe[4, 6]
    assert safe[2, 2]


def test_selects_reachable_frontier_cluster_and_respects_threshold():
    raw = np.full((20, 30), -1, dtype=np.int16)
    raw[2:18, 1:15] = 0
    raw[2:18, 22:29] = 0  # disconnected free island should not be selected
    target = select_frontier_target(
        make_map(raw), robot_xy=(0.12, 0.30), now_s=0.0, exclusions=[], config=PlannerConfig(clearance_radius_m=0.0)
    )
    assert target is not None
    assert target.x < 0.9
    assert target.path_length_m >= 0.5
    assert target.frontier_length_m >= 0.25


def test_small_frontiers_are_ignored():
    raw = np.full((5, 5), -1, dtype=np.int16)
    raw[2, 2] = 0
    assert select_frontier_target(make_map(raw), (0.15, 0.15), 0.0, [], PlannerConfig(clearance_radius_m=0.0)) is None


def test_ranking_prefers_frontier_length_then_distance_without_raycast_gain():
    close_small = FrontierTarget(0.0, 0.0, 0.0, path_length_m=0.5, frontier_length_m=0.5)
    far_large = FrontierTarget(1.0, 0.0, 0.0, path_length_m=1.0, frontier_length_m=1.2)
    assert rank_frontier_targets([close_small, far_large]) == far_large


def test_raycast_counts_unknown_cells_until_a_known_obstacle():
    unknown = np.zeros((7, 7), dtype=bool)
    occupied = np.zeros((7, 7), dtype=bool)
    unknown[3, 4] = True
    unknown[3, 5] = True
    unknown[3, 6] = True
    occupied[3, 5] = True
    gain = raycast_information_gain_m2(
        unknown, occupied, start=(3, 3), resolution_m=1.0, max_range_m=3.0, ray_count=4
    )
    assert gain == 1.0


def test_ranking_can_prefer_more_visible_unknown_area():
    low_gain = FrontierTarget(
        0.0, 0.0, 0.0, path_length_m=1.0, frontier_length_m=0.5,
        information_gain_m2=0.1, information_gain_weight=0.5,
    )
    high_gain = FrontierTarget(
        1.0, 0.0, 0.0, path_length_m=1.0, frontier_length_m=0.5,
        information_gain_m2=2.0, information_gain_weight=0.5,
    )
    assert rank_frontier_targets([low_gain, high_gain]) == high_gain


def test_distance_penalty_weight_strengthens_preference_for_nearby_frontiers():
    gain = 2.0
    near = FrontierTarget(
        0.0, 0.0, 0.0, path_length_m=1.0, frontier_length_m=gain, distance_penalty_weight=1.25
    )
    far = FrontierTarget(
        1.0, 0.0, 0.0, path_length_m=5.0, frontier_length_m=gain, distance_penalty_weight=1.25
    )
    assert near.score == gain / 2.25
    assert rank_frontier_targets([far, near]) == near


def test_raycast_is_limited_to_a_ranked_candidate_batch():
    raw = np.full((9, 9), -1, dtype=np.int16)
    raw[1, 1] = 0
    raw[4, 4] = 0
    raw[7, 7] = 0
    plan = plan_frontiers(
        make_map(raw, resolution=0.1),
        robot_xy=(0.15, 0.15),
        now_s=0.0,
        exclusions=[],
        config=PlannerConfig(
            planning_resolution_m=0.1,
            clearance_radius_m=0.0,
            min_frontier_length_m=0.0,
            min_goal_distance_m=0.0,
            min_information_gain_m2=0.0,
            raycast_candidate_limit=1,
        ),
    )
    assert plan.information_candidates_total == 3
    assert plan.information_candidates_evaluated == 1
    assert plan.has_more_information_candidates
    assert len(plan.valid_targets) == 1


def test_exclusion_zones_expire_and_suppress_nearby_targets():
    zone = ExclusionZone(1.0, 1.0, expires_at_s=10.0)
    assert is_excluded(1.5, 1.0, now_s=5.0, exclusions=[zone], radius_m=0.75)
    assert not is_excluded(1.5, 1.0, now_s=10.0, exclusions=[zone], radius_m=0.75)
