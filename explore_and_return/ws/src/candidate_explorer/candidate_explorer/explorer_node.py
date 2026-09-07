"""Autonomous frontier exploration followed by a transform-correct return home."""
from __future__ import annotations

import math
from dataclasses import replace

import rclpy
import tf2_ros
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from .frontier_planner import (
    ExclusionZone,
    FrontierTarget,
    InvalidFrontier,
    PlannerConfig,
    plan_frontiers,
    ranked_frontier_targets,
)


_NUMERIC = ParameterDescriptor(dynamic_typing=True)


def yaw_to_quaternion(yaw: float) -> Quaternion:
    quaternion = Quaternion()
    quaternion.z = math.sin(yaw / 2.0)
    quaternion.w = math.cos(yaw / 2.0)
    return quaternion


def yaw_from_quaternion(quaternion: Quaternion) -> float:
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
    )


class ExplorerNode(Node):
    """Coordinates ROS I/O while frontier_planner owns all map reasoning."""

    def __init__(self) -> None:
        super().__init__("candidate_explorer")

        # The explorer is normally started separately from the launch file.
        # Opt into challenge_sim's /clock here so `ros2 run ...` follows the
        # README workflow and never compares a wall-clock epoch to sim time.
        if self.has_parameter("use_sim_time"):
            self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        else:
            self.declare_parameter("use_sim_time", True)

        self.declare_parameter("planning_resolution_m", 0.06)
        self.declare_parameter("robot_radius_m", 0.20)
        self.declare_parameter("safety_margin_m", 0.00)
        self.declare_parameter("min_frontier_length_m", 0.10)
        self.declare_parameter("min_goal_distance_m", 0.50)
        self.declare_parameter("unknown_bridge_distance_m", 0.30)
        self.declare_parameter("allow_nav2_path_fallback", True)
        self.declare_parameter("raycast_max_range_m", 5.0)
        self.declare_parameter("raycast_rays", 120)
        self.declare_parameter("information_gain_weight", 0.50)
        self.declare_parameter("min_information_gain_m2", 0.50)
        self.declare_parameter("distance_penalty_weight", 1.25)
        self.declare_parameter("raycast_candidate_limit", 20)
        self.declare_parameter("visited_radius_m", 0.75)
        self.declare_parameter("visited_cooldown_s", 600.0)
        self.declare_parameter("failed_cooldown_s", 600.0)
        # Launch substitutions may parse an integer command-line value (e.g.
        # ``time_limit_s:=900``), so accept either ROS numeric parameter type.
        self.declare_parameter("time_limit_s", 5400.0, _NUMERIC)
        self.declare_parameter("return_reserve_s", 180.0)
        self.declare_parameter("publish_frontier_markers", True)

        self.planner_config = PlannerConfig(
            planning_resolution_m=float(self.get_parameter("planning_resolution_m").value),
            clearance_radius_m=(
                float(self.get_parameter("robot_radius_m").value)
                + float(self.get_parameter("safety_margin_m").value)
            ),
            min_frontier_length_m=float(self.get_parameter("min_frontier_length_m").value),
            min_goal_distance_m=float(self.get_parameter("min_goal_distance_m").value),
            unknown_bridge_distance_m=float(self.get_parameter("unknown_bridge_distance_m").value),
            allow_nav2_path_fallback=bool(self.get_parameter("allow_nav2_path_fallback").value),
            raycast_max_range_m=float(self.get_parameter("raycast_max_range_m").value),
            raycast_rays=int(self.get_parameter("raycast_rays").value),
            information_gain_weight=float(self.get_parameter("information_gain_weight").value),
            min_information_gain_m2=float(self.get_parameter("min_information_gain_m2").value),
            distance_penalty_weight=float(self.get_parameter("distance_penalty_weight").value),
            raycast_candidate_limit=int(self.get_parameter("raycast_candidate_limit").value),
            exclusion_radius_m=float(self.get_parameter("visited_radius_m").value),
        )

        self.latest_map: OccupancyGrid | None = None
        self._map_version = 0
        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.map_sub = self.create_subscription(OccupancyGrid, "/map", self._on_map, map_qos)
        self.frontier_marker_pub = self.create_publisher(MarkerArray, "/frontier_candidates", 1)

        self.nav_client = ActionClient(self, NavigateToPose, "/navigate_to_pose")
        self.path_client = ActionClient(self, ComputePathToPose, "/compute_path_to_pose")
        self.finish_client = self.create_client(Trigger, "/finish_exploration")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.state = "WAITING_FOR_MAP"
        self._goal_in_progress = False
        self._exploration_result: bool | None = None
        self._return_result: bool | None = None
        self._active_target: FrontierTarget | None = None
        self._visited: list[ExclusionZone] = []
        self._failed: list[ExclusionZone] = []
        self._no_frontier_cycles = 0
        self._last_no_frontier_map_version = -1
        self._last_frontier_log_map_version = -1
        self._path_queries_active = False
        self._path_queries_remaining = 0
        self._path_query_known_targets: list[FrontierTarget] = []
        self._path_query_resolved_targets: list[FrontierTarget] = []
        self._path_query_failed = 0
        self._path_query_components_found = 0
        self._path_query_blacklisted: tuple[tuple[tuple[float, float], ...], ...] = ()
        self._path_query_invalid: tuple[InvalidFrontier, ...] = ()
        self._path_query_has_more_information_candidates = False
        self._information_batch_offset = 0
        self._heading_query_active = False
        self._heading_query_target: FrontierTarget | None = None
        self.timer = self.create_timer(1.0, self._tick)

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds / 1_000_000_000.0

    def _on_map(self, msg: OccupancyGrid) -> None:
        self.latest_map = msg
        self._map_version += 1

    def _get_robot_xy_in_map_frame(self) -> tuple[float, float] | None:
        try:
            transform = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
        except tf2_ros.TransformException as ex:
            self.get_logger().debug(f"no map->base_link transform yet: {ex}")
            return None
        return transform.transform.translation.x, transform.transform.translation.y

    def get_home_pose_in_map_frame(self) -> PoseStamped | None:
        """Map frame drifts with SLAM, so transform odom origin every attempt."""
        try:
            transform = self.tf_buffer.lookup_transform("map", "odom", rclpy.time.Time())
        except tf2_ros.TransformException as ex:
            self.get_logger().debug(f"no map->odom transform yet: {ex}")
            return None
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = transform.transform.translation.x
        pose.pose.position.y = transform.transform.translation.y
        pose.pose.orientation = transform.transform.rotation
        return pose

    def _make_navigation_goal_pose(self, target: FrontierTarget) -> PoseStamped:
        """Make a goal whose final yaw follows Nav2's predicted arrival path."""
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = target.x
        pose.pose.position.y = target.y
        pose.pose.orientation = yaw_to_quaternion(target.yaw)
        return pose

    def _target_with_current_yaw(self, target: FrontierTarget) -> FrontierTarget:
        """Provide a valid query/fallback yaw without using frontier geometry."""
        try:
            transform = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
            return replace(target, yaw=yaw_from_quaternion(transform.transform.rotation))
        except tf2_ros.TransformException:
            return target

    def _make_path_query_pose(self, target: FrontierTarget) -> PoseStamped:
        """Use current yaw while requesting a path used to derive arrival yaw."""
        return self._make_navigation_goal_pose(self._target_with_current_yaw(target))

    @staticmethod
    def _path_length(path) -> float:
        return sum(
            math.hypot(
                current.pose.position.x - previous.pose.position.x,
                current.pose.position.y - previous.pose.position.y,
            )
            for previous, current in zip(path.poses, path.poses[1:])
        )

    @staticmethod
    def _final_path_yaw(path, fallback_yaw: float) -> float:
        if len(path.poses) < 2:
            return fallback_yaw
        previous = path.poses[-2].pose.position
        current = path.poses[-1].pose.position
        dx = current.x - previous.x
        dy = current.y - previous.y
        if math.hypot(dx, dy) < 1e-6:
            return fallback_yaw
        return math.atan2(dy, dx)

    def _start_nav2_path_queries(self, known: list[FrontierTarget], pending: list[FrontierTarget]) -> bool:
        """Resolve locally-unreachable candidates using Nav2's real global paths."""
        if not self.path_client.wait_for_server(timeout_sec=0.2):
            self.get_logger().warn("/compute_path_to_pose action server not available")
            return False
        self._path_queries_active = True
        self._path_queries_remaining = len(pending)
        self._path_query_known_targets = known
        self._path_query_resolved_targets = []
        self._path_query_failed = 0
        for target in pending:
            goal = ComputePathToPose.Goal()
            goal.goal = self._make_path_query_pose(target)
            goal.use_start = False
            send_future = self.path_client.send_goal_async(goal)

            def on_goal_response(future, candidate=target) -> None:
                try:
                    handle = future.result()
                except Exception:
                    handle = None
                if handle is None or not handle.accepted:
                    self._path_query_failed += 1
                    self._path_queries_remaining -= 1
                    return
                result_future = handle.get_result_async()

                def on_result(result_future, candidate=candidate) -> None:
                    try:
                        result = result_future.result()
                        length = self._path_length(result.result.path)
                        if result.status == GoalStatus.STATUS_SUCCEEDED and length > 0.0:
                            self._path_query_resolved_targets.append(
                                replace(
                                    candidate,
                                    path_length_m=length,
                                    needs_nav2_path=False,
                                    yaw=self._final_path_yaw(result.result.path, candidate.yaw),
                                    has_nav2_path_heading=True,
                                )
                            )
                        else:
                            self._path_query_failed += 1
                    except Exception:
                        self._path_query_failed += 1
                    self._path_queries_remaining -= 1

                result_future.add_done_callback(on_result)

            send_future.add_done_callback(on_goal_response)
        return True

    def _consume_nav2_path_queries(self) -> list[FrontierTarget] | None:
        if self._path_queries_remaining > 0:
            return None
        targets = ranked_frontier_targets([*self._path_query_known_targets, *self._path_query_resolved_targets])
        if self._path_query_failed:
            self.get_logger().info(f"Nav2 could not plan to {self._path_query_failed} frontier candidates")
        self._path_queries_active = False
        return targets

    def _start_final_heading_query(self, target: FrontierTarget) -> bool:
        """Get Nav2's arrival tangent before sending the navigation action."""
        if not self.path_client.wait_for_server(timeout_sec=0.2):
            return False
        self._heading_query_active = True
        self._heading_query_target = self._target_with_current_yaw(target)
        goal = ComputePathToPose.Goal()
        goal.goal = self._make_path_query_pose(target)
        goal.use_start = False
        send_future = self.path_client.send_goal_async(goal)

        def on_goal_response(future) -> None:
            try:
                handle = future.result()
            except Exception:
                handle = None
            if handle is None or not handle.accepted:
                self._heading_query_active = False
                return
            result_future = handle.get_result_async()

            def on_result(result_future) -> None:
                try:
                    result = result_future.result()
                    if result.status == GoalStatus.STATUS_SUCCEEDED and self._heading_query_target is not None:
                        self._heading_query_target = replace(
                            self._heading_query_target,
                            yaw=self._final_path_yaw(result.result.path, self._heading_query_target.yaw),
                            has_nav2_path_heading=True,
                        )
                except Exception:
                    pass
                self._heading_query_active = False

            result_future.add_done_callback(on_result)

        send_future.add_done_callback(on_goal_response)
        return True

    def _send_nav_goal(self, pose: PoseStamped, on_result) -> bool:
        """Start one action request; callbacks only record its final outcome."""
        if not self.nav_client.wait_for_server(timeout_sec=0.2):
            self.get_logger().debug("/navigate_to_pose action server not available yet")
            return False
        goal = NavigateToPose.Goal()
        goal.pose = pose
        self._goal_in_progress = True
        send_future = self.nav_client.send_goal_async(goal)

        def on_goal_response(future) -> None:
            try:
                handle = future.result()
            except Exception as ex:  # rclpy futures surface transport failures here.
                self.get_logger().warn(f"navigation goal request failed: {ex}")
                self._goal_in_progress = False
                on_result(False)
                return
            if handle is None or not handle.accepted:
                self.get_logger().warn("navigation goal rejected")
                self._goal_in_progress = False
                on_result(False)
                return
            result_future = handle.get_result_async()

            def on_action_result(result_future) -> None:
                try:
                    result = result_future.result()
                    succeeded = result.status == GoalStatus.STATUS_SUCCEEDED
                    if not succeeded:
                        self.get_logger().warn(f"navigation ended with status {result.status}")
                except Exception as ex:
                    self.get_logger().warn(f"navigation result unavailable: {ex}")
                    succeeded = False
                self._goal_in_progress = False
                on_result(succeeded)

            result_future.add_done_callback(on_action_result)

        send_future.add_done_callback(on_goal_response)
        return True

    def _send_exploration_goal(self, target: FrontierTarget) -> None:
        """Send a candidate whose yaw already follows the final path segment."""
        self._active_target = target
        if self._send_nav_goal(
            self._make_navigation_goal_pose(target),
            lambda success: setattr(self, "_exploration_result", success),
        ):
            self.get_logger().info(
                f"exploring frontier at ({target.x:.2f}, {target.y:.2f}), "
                f"score={target.score:.3f}, arrival_yaw={target.yaw:.2f}rad"
            )
        else:
            self._active_target = None

    def _call_finish_exploration(self) -> bool:
        if not self.finish_client.wait_for_service(timeout_sec=0.2):
            self.get_logger().debug("/finish_exploration service not available yet")
            return False
        future = self.finish_client.call_async(Trigger.Request())

        def on_response(response_future) -> None:
            try:
                response = response_future.result()
                self.get_logger().info(f"Session result: {response.message}")
            except Exception as ex:
                self.get_logger().error(f"finish_exploration request failed: {ex}")

        future.add_done_callback(on_response)
        return True

    def _prune_exclusions(self, now_s: float) -> None:
        self._visited = [zone for zone in self._visited if zone.expires_at_s > now_s]
        self._failed = [zone for zone in self._failed if zone.expires_at_s > now_s]

    def _raycast_batch_offset(self) -> int:
        return self._information_batch_offset

    def _advance_raycast_batch(self) -> None:
        self._information_batch_offset += max(1, self.planner_config.raycast_candidate_limit)

    @staticmethod
    def _frontier_color(rank: int) -> tuple[float, float, float]:
        """Red winner, ranks 2-5 fading to grey, then grey candidates."""
        if rank == 0:
            return 1.0, 0.0, 0.0
        if rank < 5:
            blend = rank / 4.0
            return 1.0 - 0.5 * blend, 0.5 * blend, 0.5 * blend
        return 0.5, 0.5, 0.5

    def _publish_frontier_markers(
        self,
        targets: list[FrontierTarget],
        blacklisted: tuple[tuple[tuple[float, float], ...], ...],
        invalid: tuple[InvalidFrontier, ...],
    ) -> None:
        """Visualize every valid component and its representative Nav2 goal."""
        if not bool(self.get_parameter("publish_frontier_markers").value):
            return
        now = self.get_clock().now().to_msg()
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers = [clear]
        for rank, target in enumerate(targets):
            red, green, blue = self._frontier_color(rank)
            component = Marker()
            component.header.frame_id = "map"
            component.header.stamp = now
            component.ns = "frontier_components"
            component.id = rank
            component.type = Marker.POINTS
            component.action = Marker.ADD
            component.scale.x = self.planner_config.planning_resolution_m
            component.scale.y = self.planner_config.planning_resolution_m
            component.color.r = red
            component.color.g = green
            component.color.b = blue
            component.color.a = 0.9
            component.points = [Point(x=x, y=y, z=0.04) for x, y in target.frontier_points]
            markers.append(component)

            goal = Marker()
            goal.header.frame_id = "map"
            goal.header.stamp = now
            goal.ns = "frontier_goals"
            goal.id = rank
            goal.type = Marker.SPHERE
            goal.action = Marker.ADD
            goal.pose.position.x = target.x
            goal.pose.position.y = target.y
            goal.pose.position.z = 0.08
            goal.scale.x = 0.18
            goal.scale.y = 0.18
            goal.scale.z = 0.18
            goal.color.r = red
            goal.color.g = green
            goal.color.b = blue
            goal.color.a = 1.0
            markers.append(goal)

        for component_id, points in enumerate(blacklisted):
            marker = self._component_marker(now, "blacklisted_frontiers", component_id, points, (0.0, 0.0, 0.0), 0.8)
            markers.append(marker)

        invalid_colors = {
            "too_small": (0.65, 0.0, 0.75),       # purple
            "unreachable": (0.10, 0.35, 1.0),     # blue
            "too_close": (1.0, 0.55, 0.0),        # orange
            "no_safe_start": (0.90, 0.0, 0.55),   # magenta
            "insufficient_information_gain": (0.0, 0.75, 0.35),  # green
        }
        for component_id, component in enumerate(invalid):
            color = invalid_colors.get(component.reason, (0.75, 0.75, 0.75))
            marker = self._component_marker(
                now,
                f"invalid_{component.reason}",
                component_id,
                component.points,
                color,
                0.7,
            )
            markers.append(marker)
        self.frontier_marker_pub.publish(MarkerArray(markers=markers))

    def _component_marker(
        self,
        stamp,
        namespace: str,
        marker_id: int,
        points: tuple[tuple[float, float], ...],
        color: tuple[float, float, float],
        alpha: float,
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = stamp
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = self.planner_config.planning_resolution_m
        marker.scale.y = self.planner_config.planning_resolution_m
        marker.color.r, marker.color.g, marker.color.b = color
        marker.color.a = alpha
        marker.points = [Point(x=x, y=y, z=0.035) for x, y in points]
        return marker

    def _log_frontier_plan(
        self,
        components_found: int,
        targets: list[FrontierTarget],
        blacklisted_count: int,
        invalid: tuple[InvalidFrontier, ...],
    ) -> None:
        """Emit a readable, bounded diagnostic summary for the current map."""
        if self._last_frontier_log_map_version == self._map_version:
            return
        self._last_frontier_log_map_version = self._map_version
        self.get_logger().info(
            f"frontiers found={components_found}, valid={len(targets)}, "
            f"blacklisted={blacklisted_count}, invalid={len(invalid)}, map_revision={self._map_version}"
        )
        invalid_by_reason: dict[str, int] = {}
        for component in invalid:
            invalid_by_reason[component.reason] = invalid_by_reason.get(component.reason, 0) + 1
        if invalid_by_reason:
            details = ", ".join(f"{reason}={count}" for reason, count in sorted(invalid_by_reason.items()))
            self.get_logger().info(f"  invalid reasons: {details}")
        for rank, target in enumerate(targets[:5], start=1):
            weighted_information_gain = target.information_gain_weight * target.information_gain_m2
            self.get_logger().info(
                f"  #{rank}: score={target.score:.3f}, "
                f"frontier_length={target.frontier_length_m:.2f}m, "
                f"information_gain={target.information_gain_m2:.2f}m^2, "
                f"weighted_information_gain={weighted_information_gain:.2f}m, "
                f"path_length={target.path_length_m:.2f}m, "
                f"goal=({target.x:.2f}, {target.y:.2f})"
            )

    def _record_exploration_result(self, now_s: float) -> None:
        if self._exploration_result is None or self._active_target is None:
            return
        target = self._active_target
        if self._exploration_result:
            self.get_logger().info(
                "frontier reached; blacklisting its goal to prevent a "
                f"successful no-progress loop: path={target.path_length_m:.1f}m "
                f"frontier={target.frontier_length_m:.1f}m"
            )
            self._visited.append(
                ExclusionZone(
                    target.x,
                    target.y,
                    now_s + float(self.get_parameter("visited_cooldown_s").value),
                )
            )
        else:
            self.get_logger().warn("frontier navigation failed; temporarily blacklisting its target")
            self._failed.append(
                ExclusionZone(
                    target.x,
                    target.y,
                    now_s + float(self.get_parameter("failed_cooldown_s").value),
                )
            )
        self._active_target = None
        self._exploration_result = None
        # A changed robot pose and a new exclusion set make the cheap ranking
        # different, so begin again with the best information-gain batch.
        self._information_batch_offset = 0

    def _return_due_to_time_guard(self, now_s: float) -> bool:
        limit = float(self.get_parameter("time_limit_s").value)
        reserve = float(self.get_parameter("return_reserve_s").value)
        # challenge_sim's clock begins with infrastructure bringup, before the
        # eval runner starts this node. Compare against absolute sim time so
        # the configured reserve really remains available for the trip home.
        return limit > 0.0 and now_s >= max(0.0, limit - reserve)

    def _tick_exploring(self, now_s: float) -> None:
        if self._return_due_to_time_guard(now_s):
            self.get_logger().info("return reserve reached; heading home")
            self.state = "RETURNING"
            return
        self._record_exploration_result(now_s)
        self._prune_exclusions(now_s)
        if self._goal_in_progress or self.latest_map is None:
            return
        if self._heading_query_active:
            return
        if self._heading_query_target is not None:
            target = self._heading_query_target
            self._heading_query_target = None
            self._send_exploration_goal(target)
            return
        robot_xy = self._get_robot_xy_in_map_frame()
        if robot_xy is None:
            return
        has_more_information_candidates = False
        if self._path_queries_active:
            queried_targets = self._consume_nav2_path_queries()
            if queried_targets is None:
                return
            targets = queried_targets
            components_found = self._path_query_components_found
            blacklisted = self._path_query_blacklisted
            invalid = self._path_query_invalid
            has_more_information_candidates = self._path_query_has_more_information_candidates
        else:
            plan = plan_frontiers(
                self.latest_map,
                robot_xy,
                now_s,
                [*self._visited, *self._failed],
                self.planner_config,
                raycast_candidate_offset=self._raycast_batch_offset(),
            )
            known_targets = [target for target in plan.valid_targets if not target.needs_nav2_path]
            pending_targets = [target for target in plan.valid_targets if target.needs_nav2_path]
            if pending_targets and self._start_nav2_path_queries(known_targets, pending_targets):
                self._path_query_components_found = plan.frontier_components_found
                self._path_query_blacklisted = plan.blacklisted_components
                self._path_query_invalid = plan.invalid_components
                self._path_query_has_more_information_candidates = plan.has_more_information_candidates
                self._publish_frontier_markers(known_targets, plan.blacklisted_components, plan.invalid_components)
                return
            if pending_targets and not known_targets:
                return
            targets = known_targets
            components_found = plan.frontier_components_found
            blacklisted = plan.blacklisted_components
            invalid = plan.invalid_components
            has_more_information_candidates = plan.has_more_information_candidates
        self._log_frontier_plan(components_found, targets, len(blacklisted), invalid)
        self._publish_frontier_markers(targets, blacklisted, invalid)
        target = targets[0] if targets else None
        if target is None:
            if has_more_information_candidates:
                self._advance_raycast_batch()
                self.get_logger().info("no usable target in this ray-cast batch; evaluating the next batch")
                return
            if self._last_no_frontier_map_version != self._map_version:
                self._last_no_frontier_map_version = self._map_version
                self._no_frontier_cycles += 1
                self.get_logger().info(
                    f"no eligible frontiers in map revision {self._map_version} "
                    f"({self._no_frontier_cycles}/3)"
                )
            if self._no_frontier_cycles >= 3:
                self.get_logger().info("frontier exhaustion is stable; returning home")
                self.state = "RETURNING"
            return
        self._no_frontier_cycles = 0
        if target.has_nav2_path_heading:
            self._send_exploration_goal(target)
        elif self._start_final_heading_query(target):
            self.get_logger().info("computing Nav2 arrival heading for selected frontier")
        else:
            self._send_exploration_goal(self._target_with_current_yaw(target))

    def _tick_returning(self) -> None:
        if self._goal_in_progress:
            return
        if self._return_result is not None:
            if self._return_result:
                self.get_logger().info("return-home goal succeeded")
                self.state = "FINISHING"
                self._return_result = None
                return
            self.get_logger().warn("return-home goal failed; retrying with a fresh map->odom transform")
            self._return_result = None
        home = self.get_home_pose_in_map_frame()
        if home is not None:
            self._send_nav_goal(home, lambda success: setattr(self, "_return_result", success))

    def _tick(self) -> None:
        if self.state == "WAITING_FOR_MAP":
            if self.latest_map is not None:
                self.get_logger().info("Got first /map; starting frontier exploration.")
                self.state = "EXPLORING"
            return
        if self.state == "EXPLORING":
            self._tick_exploring(self._now_s())
            return
        if self.state == "RETURNING":
            self._tick_returning()
            return
        if self.state == "FINISHING" and self._call_finish_exploration():
            self.state = "DONE"


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
