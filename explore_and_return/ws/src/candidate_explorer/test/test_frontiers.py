import math

from candidate_explorer.frontiers import (
    Frontier,
    add_information_gain,
    cluster_frontiers,
    cluster_targets,
    find_frontiers,
    grid_cell_to_world,
    is_free,
    is_occupied,
    is_unknown,
    nearest_frontier,
    rank_cluster_targets,
    rank_frontiers,
)


def test_occupancy_classification_matches_nav_map_thresholds() -> None:
    assert is_unknown(-1)
    assert is_free(0)
    assert is_free(25)
    assert not is_free(26)
    assert is_occupied(65)
    assert is_occupied(100)


def test_find_frontiers_uses_free_cells_adjacent_to_unknown() -> None:
    # -1 is unknown, 0 is free, and 100 is occupied.
    data = [100, -1, 100, 100, 0, 100, 100, 100, 100]

    frontiers = find_frontiers(
        data, width=3, height=3, resolution=1.0, origin_x=0.0, origin_y=0.0, origin_yaw=0.0
    )

    assert len(frontiers) == 1
    assert (frontiers[0].x, frontiers[0].y) == (1.5, 1.5)
    assert frontiers[0].yaw == -math.pi / 2.0


def test_grid_cell_to_world_respects_a_rotated_origin() -> None:
    x, y = grid_cell_to_world(
        0, 1, resolution=2.0, origin_x=10.0, origin_y=-3.0, origin_yaw=math.pi / 2.0
    )

    assert math.isclose(x, 9.0)
    assert math.isclose(y, 0.0, abs_tol=1e-12)


def test_nearest_frontier_skips_blacklisted_locations() -> None:
    close = Frontier(1.0, 0.0, 0.0)
    far = Frontier(3.0, 0.0, 0.0)

    selected = nearest_frontier([far, close], 0.0, 0.0, [(1.1, 0.0)], blacklist_radius_m=0.25)

    assert selected == far


def test_rank_frontiers_orders_all_eligible_candidates() -> None:
    near = Frontier(1.0, 0.0, 0.0)
    middle = Frontier(2.0, 0.0, 0.0)
    far = Frontier(3.0, 0.0, 0.0)

    ranked = rank_frontiers([far, near, middle], 0.0, 0.0, [], blacklist_radius_m=0.5)

    assert ranked == [near, middle, far]


def test_cluster_frontiers_groups_diagonal_neighbors() -> None:
    first = Frontier(0.0, 0.0, 0.0, 2, 2)
    diagonal = Frontier(1.0, 1.0, 0.0, 3, 3)
    separate = Frontier(5.0, 5.0, 0.0, 8, 8)

    clusters = cluster_frontiers([first, separate, diagonal])

    assert sorted(len(cluster.cells) for cluster in clusters) == [1, 2]


def test_cluster_ranking_favors_large_regions_for_equal_travel_cost() -> None:
    small = cluster_frontiers([Frontier(2.0, 0.0, 0.0, 0, 0)])[0]
    large = cluster_frontiers(
        [Frontier(2.0, 0.0, 0.0, 2, 2), Frontier(2.0, 1.0, 0.0, 2, 3)]
    )[0]

    targets = cluster_targets([small, large], 0.0, 0.0, [], blacklist_radius_m=0.5)
    ranked = rank_cluster_targets(
        targets,
        [2.0, 2.0],
        size_weight=1.0,
        vicinity_weight=0.0,
        information_gain_weight=0.0,
    )

    assert ranked[0].cluster_size == 2


def test_cluster_target_is_the_frontier_cell_nearest_the_cluster_centroid() -> None:
    left = Frontier(1.0, 0.0, 0.0, 4, 4)
    middle = Frontier(5.0, 0.0, 0.0, 4, 5)
    right = Frontier(9.0, 0.0, 0.0, 4, 6)
    cluster = cluster_frontiers([left, middle, right])[0]

    target = cluster_targets([cluster], 0.0, 0.0, [], blacklist_radius_m=0.5)[0]

    assert target.frontier == middle


def test_information_gain_modes_measure_unknown_space() -> None:
    target = cluster_targets(
        cluster_frontiers([Frontier(1.0, 1.0, 0.0, 2, 2)]), 0.0, 0.0, [], blacklist_radius_m=0.5
    )
    # An occupied cell to the east hides the unknown cells beyond it from raycasts.
    data = [0] * 25
    data[2 * 5 + 3] = 100
    data[2 * 5 + 4] = -1
    data[1 * 5 + 2] = -1

    local = add_information_gain(target, data, 5, 5, mode="local", range_cells=3, ray_count=8)
    raycast = add_information_gain(target, data, 5, 5, mode="raycast", range_cells=3, ray_count=72)

    assert local[0].information_gain > raycast[0].information_gain
