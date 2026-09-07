# Write-Up: Autonomous Explore-and-Return

## Approach

- The explorer consumes only the live SLAM occupancy grid, `map` TF, Nav2's
  navigation action, and the finish service. It never reads the supplied
  ground-truth room images or evaluation reports while running.
- It conservatively coarsens `/map` to a 0.06 m planning grid. A cell is
  usable only when it contains observed free space and no occupied evidence;
  obstacle clearance remains conservative.
- Candidate frontiers are safe known-free cells adjacent to unknown space.
  A 0.25 m obstacle-clearance filter accounts for the 0.20 m robot radius and
  a small margin. Tiny frontier components below 0.25 m are ignored.
- One Dijkstra search through safe known-free cells estimates the travel cost
  to every frontier. The selected goal maximizes `frontier length / (1 + path
  length)`, balancing expected information gain against transit time. Goals
  closer than 0.5 m are skipped so Nav2 cannot report a no-op as exploration;
  each component uses its furthest reachable safe point to advance the initial
  scan boundary rather than stopping at its near edge.
- Reached and failed targets are temporarily excluded in world coordinates.
  This makes the policy robust to SLAM grid resizing/correction and prevents
  oscillation at already observed or unreachable locations.
- The run returns after three successive map revisions have no eligible
  frontier, or when the configured return reserve starts. Home is transformed
  from odom origin into the current map frame immediately before every return
  attempt.

## Design Decisions & Tradeoffs

I optimized for robust high coverage rather than chasing every remaining map
pixel. Conservative free-space and clearance checks trade some last-mile
coverage for substantially fewer collision-prone goals. Frontier exhaustion is
used as a coverage proxy because the actual reachable-space denominator is not
available to the candidate. Nav2's provided costmaps and recovery behavior are
left unchanged; bad targets are handled at the explorer layer instead.

## Conclusions

### Performance

Record reproducible evaluation reports here after final runs, including map,
seed, coverage fraction, return distance, elapsed simulated time, and success.
The required success threshold is at least 80% coverage and return within
0.3 m before timeout.

### What I'd Do With More Time

- Estimate expected information gain with ray visibility rather than frontier
  length alone.
- Use Nav2 plan feedback or a planning service for a more exact route cost.
- Adapt safety margins and goal suppression from measured recovery outcomes.

### Known Limitations / Where I Expect This to Break

- Narrow doorways or noisy SLAM can eliminate a valid approach from the
  conservative clearance grid.
- Long occluded regions may have small frontier boundaries and therefore lose
  to nearer, broader frontiers.
- Map-frame localization error can still affect the physical final return,
  although recomputing `map -> odom` minimizes stale-home error.

## Reproducibility

The explorer itself is deterministic for identical `/map` and TF streams. Use
fixed simulator seeds for comparisons; test the final solution across multiple
seeds because the spawn position changes with the seed.

## Modified Files or Packages

- `candidate_explorer/frontier_planner.py`: pure map/frontier planning logic.
- `candidate_explorer/explorer_node.py`: ROS state machine and Nav2/TF wiring.
- `candidate_explorer/test/test_frontier_planner.py`: planner unit tests.
- `eval_runner.sh`: forwards the selected time limit to the explorer.
