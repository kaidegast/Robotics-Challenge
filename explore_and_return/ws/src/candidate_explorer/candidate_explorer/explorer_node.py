"""Your task: replace the EXPLORING state's body (marked below) with real
exploration + coverage logic. Everything else in this file is infrastructure
you're free to use as-is, restructure, or throw away entirely — the only
external contract that matters is:

  - subscribe /map (nav_msgs/OccupancyGrid) — published live by slam_toolbox
  - send goals via the /navigate_to_pose action (nav2_msgs/action/NavigateToPose)
  - call the /finish_exploration service (std_srvs/srv/Trigger) when done

See README.md for the full brief and scoring rubric.

Note: "home" is the odom frame's origin (see get_home_pose_in_map_frame
below) — re-derive its pose in the map frame every time you need it, don't
cache it from t=0. slam_toolbox corrects the map->odom transform as it
refines the pose graph, so where home sits in the map frame can drift even
though the robot itself never moves in the odom frame.
"""
from __future__ import annotations

import math

import rclpy
import tf2_ros
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.node import Node
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from .frontiers import (
    ClusterTarget,
    Frontier,
    add_information_gain,
    cluster_frontiers,
    cluster_targets,
    find_frontiers,
    rank_cluster_targets,
)


FRONTIER_EXHAUSTION_MAP_UPDATES = 3
FAILED_GOAL_BLACKLIST_RADIUS_M = 0.5
NUM_VISUALIZED_FRONTIERS = 5
MAX_PATH_COST_QUERIES = 12
FRONTIER_COLORS = (
    (1.0, 0.0, 0.0),  # best: red
    (1.0, 0.5, 0.0),  # orange
    (1.0, 1.0, 0.0),  # yellow
    (0.0, 1.0, 0.0),  # green
    (0.0, 0.4, 1.0),  # blue
)
FRONTIER_COLOR_NAMES = ("Red", "Orange", "Yellow", "Green", "Blue")


def yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


class ExplorerNode(Node):
    def __init__(self) -> None:
        super().__init__("candidate_explorer")

        self.latest_map: OccupancyGrid | None = None
        self.map_sub = self.create_subscription(OccupancyGrid, "/map", self._on_map, 10)
        self.declare_parameter("visualize_frontiers", True)
        self.declare_parameter("use_nav2_path_cost", False)
        self.declare_parameter("information_gain_mode", "local")
        self.declare_parameter("information_gain_radius_m", 2.0)
        self.declare_parameter("information_gain_ray_count", 72)
        self.declare_parameter("frontier_size_weight", 0.25)
        self.declare_parameter("frontier_vicinity_weight", 0.35)
        self.declare_parameter("frontier_information_gain_weight", 0.40)
        self.declare_parameter("min_frontier_cluster_size", 20)
        self.declare_parameter("minimum_goal_distance_m", 0.45)
        self.declare_parameter("minimum_progress_m", 0.25)
        self.declare_parameter("bootstrap_offset_m", 0.50)
        self.declare_parameter("min_information_gain_cells", 1000)
        self.frontier_marker_pub = self.create_publisher(MarkerArray, "/frontier_candidates", 10)

        self.nav_client = ActionClient(self, NavigateToPose, "/navigate_to_pose")
        self.path_client = ActionClient(self, ComputePathToPose, "/compute_path_to_pose")
        self.finish_client = self.create_client(Trigger, "/finish_exploration")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.state = "WAITING_FOR_MAP"
        self._goal_in_progress = False
        self._map_generation = 0
        self._last_empty_map_generation = -1
        self._consecutive_empty_maps = 0
        self._failed_goal_locations: list[tuple[float, float]] = []
        self._frontier_markers_visible = False
        self._path_scoring_in_progress = False
        self._path_candidates: list[ClusterTarget] = []
        self._path_batch: list[ClusterTarget] = []
        self._path_scores: list[tuple[ClusterTarget, float]] = []
        self._pending_frontier_target: Frontier | None = None
        self._pending_goal_mode = "NORMAL"
        self._goal_start_odom_position: tuple[float, float] | None = None

        self.timer = self.create_timer(1.0, self._tick)

    def _on_map(self, msg: OccupancyGrid) -> None:
        self.latest_map = msg
        self._map_generation += 1

    def get_robot_position_in_map_frame(self) -> tuple[float, float] | None:
        try:
            transform = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
        except tf2_ros.TransformException as ex:
            self.get_logger().warn(f"no map->base_link transform yet: {ex}")
            return None
        return transform.transform.translation.x, transform.transform.translation.y

    def get_robot_position_in_odom_frame(self) -> tuple[float, float] | None:
        try:
            transform = self.tf_buffer.lookup_transform("odom", "base_link", rclpy.time.Time())
        except tf2_ros.TransformException as ex:
            self.get_logger().warn(f"no odom->base_link transform yet: {ex}")
            return None
        return transform.transform.translation.x, transform.transform.translation.y

    def _cluster_targets(self) -> list[ClusterTarget] | None:
        if self.latest_map is None:
            return None
        robot_position = self.get_robot_position_in_map_frame()
        if robot_position is None:
            return None

        info = self.latest_map.info
        orientation = info.origin.orientation
        origin_yaw = math.atan2(
            2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z),
        )
        frontiers = find_frontiers(
            self.latest_map.data,
            width=info.width,
            height=info.height,
            resolution=info.resolution,
            origin_x=info.origin.position.x,
            origin_y=info.origin.position.y,
            origin_yaw=origin_yaw,
        )
        clusters = cluster_frontiers(frontiers)
        min_cluster_size = max(1, int(self.get_parameter("min_frontier_cluster_size").value))
        valid_clusters = [cluster for cluster in clusters if len(cluster.cells) >= min_cluster_size]
        targets = cluster_targets(
            valid_clusters,
            *robot_position,
            self._failed_goal_locations,
            blacklist_radius_m=FAILED_GOAL_BLACKLIST_RADIUS_M,
        )
        mode = self.get_parameter("information_gain_mode").value
        if mode not in ("local", "raycast"):
            self.get_logger().warn(f"Unknown information_gain_mode={mode!r}; using local.")
            mode = "local"
        radius_cells = max(
            1,
            round(self.get_parameter("information_gain_radius_m").value / info.resolution),
        )
        targets = add_information_gain(
            targets,
            self.latest_map.data,
            info.width,
            info.height,
            mode=mode,
            range_cells=radius_cells,
            ray_count=int(self.get_parameter("information_gain_ray_count").value),
        )
        min_information_gain = max(0, int(self.get_parameter("min_information_gain_cells").value))
        low_gain_count = sum(target.information_gain < min_information_gain for target in targets)
        targets = [target for target in targets if target.information_gain >= min_information_gain]
        vicinity_costs = [
            math.hypot(target.frontier.x - robot_position[0], target.frontier.y - robot_position[1])
            for target in targets
        ]
        ranked = self._rank_targets(targets, vicinity_costs)
        self.get_logger().info(
            f"Frontiers: cells={len(frontiers)}, clusters={len(clusters)}, "
            f"valid_cells={sum(len(cluster.cells) for cluster in valid_clusters)}, "
            f"valid_clusters={len(ranked)}, low_gain_filtered={low_gain_count}, "
            f"min_cluster_size={min_cluster_size}, min_gain={min_information_gain}."
        )
        self._log_top_frontiers(ranked, "geometric")
        return ranked

    def _rank_targets(
        self, targets: list[ClusterTarget], vicinity_costs: list[float]
    ) -> list[ClusterTarget]:
        return rank_cluster_targets(
            targets,
            vicinity_costs,
            size_weight=float(self.get_parameter("frontier_size_weight").value),
            vicinity_weight=float(self.get_parameter("frontier_vicinity_weight").value),
            information_gain_weight=float(self.get_parameter("frontier_information_gain_weight").value),
        )

    def _log_top_frontiers(self, targets: list[ClusterTarget], scoring: str) -> None:
        if not targets:
            self.get_logger().info(f"Top frontiers ({scoring}): none.")
            return
        summary = ", ".join(
            f"#{index} {FRONTIER_COLOR_NAMES[index - 1]}: score={target.score:.3f}, "
            f"size={target.cluster_size}, distance={target.vicinity_cost:.2f}m, "
            f"info_gain={target.information_gain}"
            for index, target in enumerate(targets[:NUM_VISUALIZED_FRONTIERS], start=1)
        )
        self.get_logger().info(f"Top frontiers ({scoring}): {summary}")

    @staticmethod
    def _path_length(path) -> float:
        return sum(
            math.hypot(
                second.pose.position.x - first.pose.position.x,
                second.pose.position.y - first.pose.position.y,
            )
            for first, second in zip(path.poses, path.poses[1:])
        )

    def _start_path_scoring(self, candidates: list[ClusterTarget]) -> None:
        self._path_batch = candidates[:MAX_PATH_COST_QUERIES]
        if not self.path_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn("/compute_path_to_pose unavailable; using geometric cluster ranking")
            self._pending_frontier_target = self._path_batch[0].frontier
            self._pending_goal_mode = "NORMAL"
            self._publish_frontier_markers(self._path_batch)
            return
        self._path_candidates = list(self._path_batch)
        self._path_scores = []
        self._path_scoring_in_progress = True
        self._request_next_path()

    def _request_next_path(self) -> None:
        if not self._path_candidates:
            self._path_scoring_in_progress = False
            if not self._path_scores:
                for candidate in self._path_batch:
                    self._failed_goal_locations.append((candidate.frontier.x, candidate.frontier.y))
                return
            ranked = self._rank_targets(
                [candidate for candidate, _ in self._path_scores],
                [cost for _, cost in self._path_scores],
            )
            self._log_top_frontiers(ranked, "Nav2 path")
            self._publish_frontier_markers(ranked)
            self._pending_frontier_target = ranked[0].frontier
            self._pending_goal_mode = "NORMAL"
            return

        candidate = self._path_candidates.pop(0)
        goal = ComputePathToPose.Goal()
        goal.goal = self._pose_for_frontier(candidate.frontier)
        goal.use_start = False
        send_future = self.path_client.send_goal_async(goal)

        def _on_goal_response(future) -> None:
            handle = future.result()
            if not handle.accepted:
                self._request_next_path()
                return
            result_future = handle.get_result_async()

            def _on_result(future2) -> None:
                result = future2.result()
                if result.status == GoalStatus.STATUS_SUCCEEDED:
                    self._path_scores.append((candidate, self._path_length(result.result.path)))
                self._request_next_path()

            result_future.add_done_callback(_on_result)

        send_future.add_done_callback(_on_goal_response)

    def _pose_for_frontier(self, target: Frontier) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = target.x
        pose.pose.position.y = target.y
        pose.pose.orientation = yaw_to_quaternion(target.yaw)
        return pose

    def _bootstrap_pose_for_frontier(self, target: Frontier) -> PoseStamped:
        pose = self._pose_for_frontier(target)
        offset = float(self.get_parameter("bootstrap_offset_m").value)
        pose.pose.position.x += offset * math.cos(target.yaw)
        pose.pose.position.y += offset * math.sin(target.yaw)
        return pose

    def _goal_progress_m(self) -> float | None:
        if self._goal_start_odom_position is None:
            return None
        end = self.get_robot_position_in_odom_frame()
        if end is None:
            return None
        return math.hypot(end[0] - self._goal_start_odom_position[0], end[1] - self._goal_start_odom_position[1])

    def _publish_frontier_markers(self, targets: list[ClusterTarget] | None) -> None:
        """Publish every cell in each of the five best frontier regions."""
        if not self.get_parameter("visualize_frontiers").value:
            if self._frontier_markers_visible:
                self._publish_marker_deletes()
            return

        markers = MarkerArray()
        displayed_targets = (targets or [])[:NUM_VISUALIZED_FRONTIERS]
        for index, target in enumerate(displayed_targets):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "best_frontiers"
            marker.id = index
            marker.type = Marker.POINTS
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = 0.12
            marker.color.r, marker.color.g, marker.color.b = FRONTIER_COLORS[index]
            marker.color.a = 1.0
            marker.points = [Point(x=frontier.x, y=frontier.y, z=0.05) for frontier in target.cells]
            markers.markers.append(marker)

            destination = Marker()
            destination.header.frame_id = "map"
            destination.header.stamp = self.get_clock().now().to_msg()
            destination.ns = "frontier_destinations"
            destination.id = index
            destination.type = Marker.SPHERE
            destination.action = Marker.ADD
            destination.pose.position.x = target.frontier.x
            destination.pose.position.y = target.frontier.y
            destination.pose.position.z = 0.1
            destination.pose.orientation.w = 1.0
            destination.scale.x = destination.scale.y = destination.scale.z = 0.25
            destination.color.r, destination.color.g, destination.color.b = FRONTIER_COLORS[index]
            destination.color.a = 1.0
            markers.markers.append(destination)

        for index in range(len(displayed_targets), NUM_VISUALIZED_FRONTIERS):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.ns = "best_frontiers"
            marker.id = index
            marker.action = Marker.DELETE
            markers.markers.append(marker)
            destination = Marker()
            destination.header.frame_id = "map"
            destination.ns = "frontier_destinations"
            destination.id = index
            destination.action = Marker.DELETE
            markers.markers.append(destination)

        self.frontier_marker_pub.publish(markers)
        self._frontier_markers_visible = bool(targets)

    def _publish_marker_deletes(self) -> None:
        markers = MarkerArray()
        for index in range(NUM_VISUALIZED_FRONTIERS):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.ns = "best_frontiers"
            marker.id = index
            marker.action = Marker.DELETE
            markers.markers.append(marker)
            destination = Marker()
            destination.header.frame_id = "map"
            destination.ns = "frontier_destinations"
            destination.id = index
            destination.action = Marker.DELETE
            markers.markers.append(destination)
        self.frontier_marker_pub.publish(markers)
        self._frontier_markers_visible = False

    def get_home_pose_in_map_frame(self) -> PoseStamped | None:
        try:
            t = self.tf_buffer.lookup_transform("map", "odom", rclpy.time.Time())
        except tf2_ros.TransformException as ex:
            self.get_logger().warn(f"no map->odom transform yet: {ex}")
            return None
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = t.transform.translation.x
        pose.pose.position.y = t.transform.translation.y
        pose.pose.orientation = t.transform.rotation
        return pose

    def send_nav_goal(self, pose: PoseStamped, on_done) -> None:
        if not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("/navigate_to_pose action server not available")
            on_done(False)
            return
        goal = NavigateToPose.Goal()
        goal.pose = pose
        self._goal_in_progress = True
        send_future = self.nav_client.send_goal_async(goal)

        def _on_goal_response(fut):
            handle = fut.result()
            if not handle.accepted:
                self.get_logger().warn("goal rejected")
                self._goal_in_progress = False
                on_done(False)
                return
            result_future = handle.get_result_async()

            def _on_result(fut2):
                status = fut2.result().status
                self._goal_in_progress = False
                on_done(status == GoalStatus.STATUS_SUCCEEDED)

            result_future.add_done_callback(_on_result)

        send_future.add_done_callback(_on_goal_response)

    def call_finish_exploration(self) -> None:
        if not self.finish_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("/finish_exploration service not available")
            return
        future = self.finish_client.call_async(Trigger.Request())

        def _on_response(fut):
            res = fut.result()
            self.get_logger().info(f"Session result: {res.message}")

        future.add_done_callback(_on_response)

    def _tick(self) -> None:
        if self.state == "WAITING_FOR_MAP":
            if self.latest_map is not None:
                self.get_logger().info("Got first /map.")
                self.state = "EXPLORING"
            return

        if self.state in ("EXPLORING", "BOOTSTRAPPING"):
            if self._goal_in_progress:
                return

            if self._pending_frontier_target is not None:
                target = self._pending_frontier_target
                self._pending_frontier_target = None
                robot_position = self.get_robot_position_in_map_frame()
                if robot_position is None:
                    return
                target_distance = math.hypot(target.x - robot_position[0], target.y - robot_position[1])
                goal_mode = self._pending_goal_mode
                if goal_mode == "NORMAL" and target_distance < float(
                    self.get_parameter("minimum_goal_distance_m").value
                ):
                    goal_mode = "BOOTSTRAP"
                    self.state = "BOOTSTRAPPING"
                    self.get_logger().info(
                        f"Frontier is only {target_distance:.2f}m away; sending bootstrap goal."
                    )
                pose = (
                    self._bootstrap_pose_for_frontier(target)
                    if goal_mode == "BOOTSTRAP"
                    else self._pose_for_frontier(target)
                )
                self._goal_start_odom_position = self.get_robot_position_in_odom_frame()
                if self._goal_start_odom_position is None:
                    self._pending_frontier_target = target
                    self._pending_goal_mode = goal_mode
                    return

                def _on_done(success: bool) -> None:
                    progress = self._goal_progress_m()
                    minimum_progress = float(self.get_parameter("minimum_progress_m").value)
                    made_progress = progress is not None and progress >= minimum_progress
                    if not success or not made_progress:
                        self._failed_goal_locations.append((target.x, target.y))
                        reason = "failed" if not success else (
                            "could not measure odometry" if progress is None else f"moved only {progress:.2f}m"
                        )
                        self.get_logger().warn(
                            f"{goal_mode.title()} frontier goal {reason}; blacklisting "
                            f"({target.x:.2f}, {target.y:.2f})."
                        )
                    else:
                        self.get_logger().info(
                            f"{goal_mode.title()} goal moved {progress:.2f}m toward frontier "
                            f"({target.x:.2f}, {target.y:.2f})."
                        )
                    self._goal_start_odom_position = None
                    self.state = "EXPLORING"

                self.send_nav_goal(pose, _on_done)
                return

            if self._path_scoring_in_progress:
                return

            targets = self._cluster_targets()
            if targets is None:
                self._publish_frontier_markers(None)
                return  # A missing TF is transient; do not mistake it for completion.
            if not targets:
                self._publish_frontier_markers([])
                if self._map_generation != self._last_empty_map_generation:
                    self._last_empty_map_generation = self._map_generation
                    self._consecutive_empty_maps += 1
                    self.get_logger().info(
                        f"No valid frontier on map update {self._consecutive_empty_maps}/"
                        f"{FRONTIER_EXHAUSTION_MAP_UPDATES}."
                    )
                if self._consecutive_empty_maps >= FRONTIER_EXHAUSTION_MAP_UPDATES:
                    self.get_logger().info("Frontiers exhausted; returning home.")
                    self.state = "RETURNING"
                return

            self._consecutive_empty_maps = 0
            self._last_empty_map_generation = -1
            if self.get_parameter("use_nav2_path_cost").value:
                self._start_path_scoring(targets)
            else:
                self._publish_frontier_markers(targets)
                self._pending_frontier_target = targets[0].frontier
                self._pending_goal_mode = "NORMAL"
            return

        if self.state == "RETURNING":
            if self._goal_in_progress:
                return
            home = self.get_home_pose_in_map_frame()
            if home is None:
                return  # try again next tick

            def _on_done(success: bool) -> None:
                self.get_logger().info(f"return-home goal finished, success={success}")
                self.state = "FINISHING"

            self.send_nav_goal(home, _on_done)
            return

        if self.state == "FINISHING":
            self.state = "DONE"
            self.call_finish_exploration()
            return

        # DONE: nothing left to do.


def main() -> None:
    rclpy.init()
    node = ExplorerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
