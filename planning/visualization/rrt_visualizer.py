"""Visualization utilities for RRT algorithms."""

from typing import TYPE_CHECKING

import numpy as np
import viser

from ..graph.node import Node

if TYPE_CHECKING:
    from ..sampling.rrt import RRTBase


class RRTVisualizer:
    """Visualizer for RRT algorithms."""

    def __init__(self, server: viser.ViserServer) -> None:
        """Initialize the visualizer.

        Args:
            server: Viser server instance
        """
        self.server = server

    def visualize_start_goal(
        self,
        start_state: np.ndarray,
        goal_state: np.ndarray,
        start_color: tuple[int, int, int] = (0, 255, 0),
        goal_color: tuple[int, int, int] = (255, 0, 0),
        radius: float = 0.3,
    ) -> None:
        """Visualize start and goal positions.

        Args:
            start_state: Starting state
            goal_state: Goal state
            start_color: RGB color for start marker (default: green)
            goal_color: RGB color for goal marker (default: red)
            radius: Radius of the marker spheres
        """
        # Extract positions (first 3 dimensions)
        start_pos = start_state[:3] if len(start_state) >= 3 else np.append(start_state, 0)
        goal_pos = goal_state[:3] if len(goal_state) >= 3 else np.append(goal_state, 0)

        self.server.scene.add_icosphere(
            "/start",
            radius=radius,
            position=tuple(start_pos),
            color=start_color,
        )

        self.server.scene.add_icosphere(
            "/goal",
            radius=radius,
            position=tuple(goal_pos),
            color=goal_color,
        )

    def visualize_branches(
        self,
        planner: "RRTBase",
        success_color: tuple[int, int, int] = (100, 150, 255),
        failure_color: tuple[int, int, int] = (255, 100, 100),
        line_width: float = 1.5,
        prefix: str = "/branches",
    ) -> None:
        """visualize branches of the RRT tree."""
        nodes = planner.get_all_nodes()
        goal_node = planner.get_goal_node()
        leaf_nodes = sorted(
            [n for n in nodes if n.is_leaf()],
            key=lambda n: n.get_depth(),
            reverse=True,
        )

        success_nodes = self._collect_success_nodes(goal_node)
        success_leaves, failure_leaves = self._split_leaves(leaf_nodes, success_nodes)

        print(
            f"Visualizing {len(success_leaves) + len(failure_leaves)} branches "
            f"({len(success_leaves)} success, {len(failure_leaves)} failure)..."
        )

        self._draw_branches(
            failure_leaves, success_nodes, failure_color, line_width, f"{prefix}/failure"
        )
        self._draw_branches(
            success_leaves, success_nodes, success_color, line_width, f"{prefix}/success"
        )

    # === helper functions ===

    def _collect_success_nodes(self, goal_node: Node | None) -> set[Node]:
        """collect all nodes on the success path."""
        success_nodes = set()
        current = goal_node
        while current is not None:
            success_nodes.add(current)
            current = current.parent
        return success_nodes

    def _split_leaves(
        self, leaf_nodes: list[Node], success_nodes: set[Node]
    ) -> tuple[list[Node], list[Node]]:
        """separate success leaves and failure leaves."""
        success_leaves: list[Node] = []
        failure_leaves: list[Node] = []
        for leaf in leaf_nodes:
            (success_leaves if leaf in success_nodes else failure_leaves).append(leaf)
        return success_leaves, failure_leaves

    def _draw_branches(
        self,
        leaves: list[Node],
        success_nodes: set[Node],
        color: tuple[int, int, int],
        line_width: float,
        prefix: str,
    ) -> None:
        """draw branches and end markers using leaves."""
        segments, end_markers = [], []
        for leaf in leaves:
            path = leaf.get_path_from_root()
            for i in range(len(path) - 1):
                p, c = path[i], path[i + 1]
                if color != (255, 100, 100) and p not in success_nodes and c not in success_nodes:
                    continue
                parent_pos = self._get_pos(p.state)
                child_pos = self._get_pos(c.state)
                segments.append([parent_pos, child_pos])
            end_markers.append(self._get_pos(path[-1].state))

        if segments:
            self.server.scene.add_line_segments(
                f"{prefix}_branches",
                points=np.array(segments),
                colors=color,
                line_width=line_width,
            )

        for idx, pos in enumerate(end_markers):
            self.server.scene.add_icosphere(
                f"{prefix}_end_{idx}",
                radius=0.1,
                position=tuple(pos),
                color=color,
            )

    def _get_pos(self, state: np.ndarray) -> np.ndarray:
        """extract 3D coordinates from state."""
        return state[:3] if len(state) >= 3 else np.append(state, 0)
