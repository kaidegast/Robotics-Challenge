# Write-Up: Autonomous Explore-and-Return

## Approach

### Map Downsampling

- The explorer uses only the live `/map`, TF, and Nav2. It does not read the
  hidden maps or evaluation reports while running.
- It subscribes to `/map` with reliable, transient-local QoS and enables
  simulation time internally. TF provides `map -> base_link` for planning and
  a fresh `map -> odom` transform for returning home.
- The occupancy grid is coarsened to a 0.06 m planning grid. Values `<= 25`
  are free, values `>= 65` are occupied, and the rest are unknown. A coarse
  cell is free only when it has free evidence and no occupied evidence.

### How Frontiers Are Identified

- The planner removes free cells within the robot clearance radius
  (`robot_radius_m + safety_margin_m`, normally 0.20 m) of an obstacle. A
  frontier is a remaining safe-free cell adjacent to unknown space. Frontier
  cells are clustered with 8-connectivity; components shorter than 0.10 m are
  discarded.

### How Frontiers Are Scored

- One 8-connected Dijkstra search estimates paths from the robot to all safe
  frontier cells. It prevents diagonal corner-cutting and permits only a
  bounded 0.30 m obstacle-clear bridge through unknown map gaps. If this local
  model cannot connect an otherwise safe goal, Nav2's `/compute_path_to_pose`
  verifies a real route and supplies its path length; no straight-line route
  estimate is used for final scoring.
- Candidates first receive the inexpensive provisional score:

  ```text
  frontier_length / (1 + distance_penalty_weight * path_length)
  ```

  Only the top `raycast_candidate_limit` candidates (20 by default) are then
  ray-cast. Each selected goal casts 120 rays through the planning grid to a
  5.0 m range, counting unique unknown cells visible before known obstacles.
  Candidates below 0.50 m² potential information gain are rejected. If a
  batch produces no usable target, the next batch is evaluated rather than
  treating untested candidates as exhausted.
- Ray-cast candidates are ranked by:

  ```text
  score = (frontier_length + information_gain_weight * information_gain)
          / (1 + distance_penalty_weight * path_length)
  ```

  Defaults are `information_gain_weight=0.50` and
  `distance_penalty_weight=1.25`, modestly favouring nearer frontiers.

  Larger frontiers and goals that reveal more unknown area increase the
  numerator. A longer route increases the denominator, so it lowers the
  score. This makes the score a gain-per-travel-cost estimate rather than a
  nearest-frontier rule.

### How a Goal for Nav2 Is Selected

- A component goal is the eligible safe frontier cell nearest the component
  centroid. The centroid itself is never sent as a goal because it may be
  unknown or occupied. Goals closer than 0.50 m are ignored to avoid no-op
  navigation success.
- Before navigation, the selected target obtains a Nav2 predicted path. The
  yaw sent in `NavigateToPose` is the direction of that path's final segment,
  so the robot normally arrives aligned with its required final yaw instead of
  rotating in place. The supplied Nav2 controller and goal-checker settings
  are not modified.
- The candidate costmaps use a 0.23 m Nav2 robot radius. Their inflation
  layers use `inflation_radius=1.75` and `cost_scaling_factor=2.58`, making
  obstacle-adjacent routes less attractive and reducing collision risk.

### Execution, Blacklisting, and Return

- Exactly one navigation action runs at a time. Successful and failed goals
  are blacklisted in world coordinates for 600 simulated seconds within the
  configured 0.75 m radius. The blacklist applies to the goal, not the whole
  frontier component, so nearby frontiers remain available.
- Exploration ends after all ray-cast batches have no eligible frontier for
  three map revisions, or when the return reserve begins. The return goal is
  derived from `map -> odom` immediately before every attempt. Only a
  successful Nav2 return permits calling `/finish_exploration`.

### Debugging Logs and Frontier Visualisation

- The node publishes `/frontier_candidates` as a
  `visualization_msgs/MarkerArray`. The supplied RViz configuration displays
  this topic in the `map` frame. Each valid component is shown as points and
  its selected Nav2 goal as a sphere.
- Valid candidates are ranked and coloured by score: the best is red, ranks
  two through five fade from red toward grey, and all lower-ranked valid
  frontiers are grey.
- A temporarily blacklisted goal is shown in black. Invalid frontier
  components use a reason-specific colour: purple for `too_small`, blue for
  `unreachable`, orange for `too_close`, magenta for `no_safe_start`, and
  green for `insufficient_information_gain`.
- For each new map revision, the node logs the total number of raw frontier
  components, the valid, blacklisted, and invalid counts, plus invalid-reason
  counts. It prints the top five valid candidates with their score, frontier
  length, raw ray-cast information gain, weighted information-gain term, path
  length, and world-coordinate goal. This makes it possible to compare the
  numeric ranking directly with the RViz colours.

## Design Decisions & Tradeoffs

- **Coverage versus safety:** The planner uses observed safe-free cells and
  clearance filtering. This avoids risky wall-adjacent goals, but a large
  clearance margin can hide narrow doorways. The default margin is therefore
  zero.
- **Information gain versus planning cost:** Ray-casting is more informative
  than frontier length alone but scales with the number of candidates. The
  two-stage ranking caps normal work at 20 ray-casts per batch while still
  allowing later candidates to be considered.
- **Local planning versus Nav2:** Dijkstra cheaply evaluates the map once per
  cycle. Nav2 is reserved for candidates that the conservative local grid
  cannot connect and for deriving the selected goal's arrival yaw.
- **Short travel versus map gain:** The weighted path term discourages
  back-and-forth trips, but does not make distant high-gain frontiers
  impossible to select.
- **Repeated goals versus changed maps:** World-coordinate goal blacklisting
  prevents immediate success/failure loops without permanently excluding an
  area after SLAM changes.

## Conclusions

### Performance

The current implementation has been built with:

```bash
colcon build --symlink-install --packages-select candidate_explorer
```

The pure planner test suite currently contains 11 tests and passes in the
challenge container. It covers map coarsening, coordinate conversion,
clearance, frontier thresholding, reachability, ranking, exclusion zones,
ray-casting, and the bounded ray-cast batch limit.

No final coverage claim is recorded here. Full evaluation should use fixed
map/seed combinations and record coverage, return error, elapsed simulated
time, collisions, and success from the generated report.

### What I'd Do With More Time

- Compare the ray-cast estimate with achieved coverage across fixed seeds and
  tune the ray range, candidate-batch limit, and score weights from results.
- Use a clearance preference in the score rather than a larger hard safety
  margin, so wall-adjacent goals are discouraged without hiding narrow areas.
- Cache or incrementally update ray-cast results when the map changes only
  slightly.
- Add integration tests covering Nav2 path requests, arrival-yaw selection,
  goal failure recovery, and return behaviour.

### Known Limitations / Where I Expect This to Break

- A noisy or incomplete SLAM map can make a real doorway appear unsafe or
  disconnected.
- The ray-cast gain is optimistic: unknown cells may contain obstacles, and
  it is a proxy for future coverage rather than ground-truth coverage.
- The final predicted path can change when Nav2 replans during execution, so
  an arrival-yaw query reduces but cannot mathematically eliminate all final
  rotation.
- Goal-only blacklisting can still allow a materially shifted version of the
  same frontier to be reconsidered after mapping changes.

## Reproducibility

The planner is deterministic for identical `/map`, TF, Nav2 path responses,
and parameters. The simulator spawn can vary by seed, so comparisons should
use explicit fixed seeds. For example:

```bash
cd /challenge
./eval_runner.sh maps/4/room.yaml 1
```

The selected time limit is forwarded to the explorer as `time_limit_s`. Use
the same map, seed, time limit, time scale, and node parameters when comparing
runs.

## Modified/Added Files or Packages

- `candidate_explorer/frontier_planner.py` — pure NumPy occupancy-grid,
  frontier, reachability, ray-cast, scoring, and exclusion logic.
- `candidate_explorer/explorer_node.py` — ROS interfaces, state machine, TF,
  Nav2 actions, logs, marker publication, and return logic.
- `candidate_explorer/test/test_frontier_planner.py` — pure planner tests.
- `candidate_explorer/package.xml` — NumPy and visualization dependencies.
- `challenge_sim/config/challenge.rviz` — displays `/frontier_candidates`.
- `eval_runner.sh` — forwards the selected time limit to the explorer.
