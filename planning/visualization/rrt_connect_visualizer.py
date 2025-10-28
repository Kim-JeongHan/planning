"""Visualization utilities for RRT-Connect algorithm."""

from typing import TYPE_CHECKING

import numpy as np

from .base_visualizer import BaseVisualizer

if TYPE_CHECKING:
    from ..sampling.rrt import RRTConnect


class RRTConnectVisualizer(BaseVisualizer):
    """Visualizer for RRT-Connect algorithm (bidirectional RRT)."""

    def visualize_branches(
        self,
        planner: "RRTConnect",
        success_color: tuple[int, int, int] = (100, 150, 255),
        failure_color: tuple[int, int, int] = (255, 100, 100),
        line_width: float = 1.5,
        prefix: str = "/branches",
        show_nodes: bool = True,
        node_radius: float = 0.08,
    ) -> None:
        """Visualize branches (compatible with RRT visualizer interface).

        Shows only:
        - Success path: Blue edges and nodes
        - Failed nodes: Red edges and nodes (includes disconnected tree nodes)
        - Connection points: Yellow spheres

        Args:
            planner: RRT-Connect planner instance
            success_color: RGB color for successful path (default: blue)
            failure_color: RGB color for failed attempts (default: red)
            line_width: Line width for edges
            prefix: Scene prefix
            show_nodes: Whether to show node markers
            node_radius: Radius of node markers
        """
        # Get connection path and collect success nodes
        connection_path = []
        success_nodes_set = set()
        if planner.connection_point_start is not None:
            connection_path = planner._extract_path()
            success_nodes_set = set(connection_path)

        # Collect all failed nodes (including disconnected tree nodes)
        failed_nodes = self._collect_failed_nodes(planner, success_nodes_set)

        # Visualize components
        self._visualize_success_path(
            connection_path, success_color, line_width, show_nodes, node_radius, prefix
        )
        self._visualize_failed_nodes(
            failed_nodes, failure_color, line_width, show_nodes, node_radius, prefix
        )
        self._visualize_connection_points(planner, prefix)
        self._print_stats(planner, connection_path, failed_nodes)

    def _collect_failed_nodes(self, planner: "RRTConnect", success_nodes_set: set) -> list:
        """Collect all failed nodes including disconnected tree nodes.

        Args:
            planner: RRT-Connect planner instance
            success_nodes_set: Set of nodes in the success path

        Returns:
            List of all failed nodes
        """
        failed_nodes = []

        # Add explicit failed connection attempts
        failed_nodes.extend(planner.failed_nodes)

        # Add disconnected nodes from start tree
        for node in planner.start_nodes:
            if node not in success_nodes_set:
                failed_nodes.append(node)

        # Add disconnected nodes from goal tree
        for node in planner.goal_nodes:
            if node not in success_nodes_set:
                failed_nodes.append(node)

        return failed_nodes

    def _visualize_success_path(
        self,
        connection_path: list,
        success_color: tuple[int, int, int],
        line_width: float,
        show_nodes: bool,
        node_radius: float,
        prefix: str,
    ) -> None:
        """Visualize the successful connection path."""
        if not connection_path or len(connection_path) <= 1:
            return

        # Draw success edges
        success_edges = []
        for i in range(len(connection_path) - 1):
            p1 = self._get_pos(connection_path[i].state)
            p2 = self._get_pos(connection_path[i + 1].state)
            success_edges.append([p1, p2])

        if success_edges:
            self.server.scene.add_line_segments(
                f"{prefix}/success_path",
                points=np.array(success_edges),
                colors=success_color,
                line_width=line_width * 2.0,
            )

        # Draw success nodes
        if show_nodes:
            for idx, node in enumerate(connection_path):
                pos = self._get_pos(node.state)
                self.server.scene.add_icosphere(
                    f"{prefix}/success_node_{idx}",
                    radius=node_radius,
                    position=tuple(pos),
                    color=success_color,
                )

    def _visualize_failed_nodes(
        self,
        failed_nodes: list,
        failure_color: tuple[int, int, int],
        line_width: float,
        show_nodes: bool,
        node_radius: float,
        prefix: str,
    ) -> None:
        """Visualize all failed nodes (including disconnected tree nodes)."""
        if not failed_nodes:
            return

        # Draw failed edges
        failed_edges = []
        for node in failed_nodes:
            if node.parent is not None:
                parent_pos = self._get_pos(node.parent.state)
                child_pos = self._get_pos(node.state)
                failed_edges.append([parent_pos, child_pos])

        if failed_edges:
            self.server.scene.add_line_segments(
                f"{prefix}/failed_attempts",
                points=np.array(failed_edges),
                colors=failure_color,
                line_width=line_width * 0.5,
            )

        # Draw failed nodes
        if show_nodes:
            for idx, node in enumerate(failed_nodes):
                pos = self._get_pos(node.state)
                self.server.scene.add_icosphere(
                    f"{prefix}/failed_node_{idx}",
                    radius=node_radius * 0.7,
                    position=tuple(pos),
                    color=failure_color,
                )

    def _visualize_connection_points(
        self,
        planner: "RRTConnect",
        prefix: str,
    ) -> None:
        """Visualize the connection points between trees."""
        if planner.connection_point_start is None or planner.connection_point_goal is None:
            return

        connection_color = (255, 200, 0)
        connection_start_pos = self._get_pos(planner.connection_point_start.state)
        connection_goal_pos = self._get_pos(planner.connection_point_goal.state)

        self.server.scene.add_icosphere(
            f"{prefix}/connection_start",
            radius=0.2,
            position=tuple(connection_start_pos),
            color=connection_color,
        )

        self.server.scene.add_icosphere(
            f"{prefix}/connection_goal",
            radius=0.2,
            position=tuple(connection_goal_pos),
            color=connection_color,
        )

    def _print_stats(
        self,
        planner: "RRTConnect",
        connection_path: list,
        failed_nodes: list,
    ) -> None:
        """Print visualization statistics."""
        total_nodes = len(planner.start_nodes) + len(planner.goal_nodes)
        connection_status = (
            f"✅ Connected with {len(connection_path)} nodes in path"
            if planner.connection_point_start
            else "❌ Trees not connected"
        )

        print(
            f"Visualized RRT-Connect:"
            f"\n  Total nodes explored: {total_nodes}"
            f"\n  Success path nodes: {len(connection_path)}"
            f"\n  Failed nodes: {len(failed_nodes)}"
            f"\n  {connection_status}"
        )

    def _get_pos(self, state: np.ndarray) -> np.ndarray:
        """Extract 3D coordinates from state.

        Args:
            state: Node state (N-dimensional)

        Returns:
            3D position array
        """
        return state[:3] if len(state) >= 3 else np.append(state, 0)
