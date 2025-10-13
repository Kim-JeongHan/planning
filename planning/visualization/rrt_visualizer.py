"""Visualization utilities for RRT algorithms."""

import numpy as np
import viser

from ..graph.node import Node


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
        nodes: list[Node],
        goal_node: Node | None = None,
        min_depth: int = 3,
        max_branches: int = 20,
        success_color: tuple[int, int, int] = (100, 150, 255),
        failure_color: tuple[int, int, int] = (255, 100, 100),
        line_width: float = 1.5,
        prefix: str = "/branches",
    ) -> None:
        """Visualize branches as success (blue) or failure (red) paths.

        Args:
            nodes: All nodes in the tree
            goal_node: The node that reached the goal (for identifying success path)
            min_depth: Minimum depth for a branch to be visualized
            max_branches: Maximum number of branches to show
            success_color: RGB color for the successful path (blue)
            failure_color: RGB color for failed paths (red)
            line_width: Width of the branch lines
            prefix: Prefix for scene node names
        """
        # Find leaf nodes with sufficient depth
        leaf_nodes = [node for node in nodes if node.is_leaf() and node.get_depth() >= min_depth]

        # Sort by depth (longer branches first)
        leaf_nodes.sort(key=lambda n: n.get_depth(), reverse=True)

        # Separate success and failure paths
        success_path_nodes = set()
        if goal_node is not None:
            # Mark all nodes in the success path
            current: Node | None = goal_node
            while current is not None:
                success_path_nodes.add(current)
                current = current.parent

        # Categorize leaf nodes
        success_leaves = []
        failure_leaves = []

        for leaf in leaf_nodes:
            if leaf in success_path_nodes:
                success_leaves.append(leaf)
            else:
                failure_leaves.append(leaf)

        # Limit number of branches (keep all success, subsample failures)
        if len(failure_leaves) > max_branches:
            failure_leaves = failure_leaves[:max_branches]

        total_branches = len(success_leaves) + len(failure_leaves)
        print(
            f"Visualizing {total_branches} exploration branches ({len(success_leaves)} success, {len(failure_leaves)} failures)..."
        )

        # Draw failure branches (red)
        for branch_idx, leaf_node in enumerate(failure_leaves):
            path = leaf_node.get_path_from_root()

            # Draw this branch
            for i in range(len(path) - 1):
                parent_state = path[i].state
                child_state = path[i + 1].state

                # Skip if part of success path
                if path[i] in success_path_nodes and path[i + 1] in success_path_nodes:
                    continue

                # Extract positions
                parent_pos = (
                    parent_state[:3] if len(parent_state) >= 3 else np.append(parent_state, 0)
                )
                child_pos = child_state[:3] if len(child_state) >= 3 else np.append(child_state, 0)

                points = np.array([parent_pos, child_pos])
                self.server.scene.add_spline_catmull_rom(
                    f"{prefix}/failure_{branch_idx}_segment_{i}",
                    positions=points,
                    color=failure_color,
                    line_width=line_width,
                )

            # Add marker at the end of the branch
            end_pos = (
                path[-1].state[:3] if len(path[-1].state) >= 3 else np.append(path[-1].state, 0)
            )
            self.server.scene.add_icosphere(
                f"{prefix}/failure_{branch_idx}_end",
                radius=0.1,
                position=tuple(end_pos),
                color=failure_color,
            )

        # Draw success branches (blue) - usually just one or a few
        for branch_idx, leaf_node in enumerate(success_leaves):
            path = leaf_node.get_path_from_root()

            # Draw this branch
            for i in range(len(path) - 1):
                parent_state = path[i].state
                child_state = path[i + 1].state

                # Extract positions
                parent_pos = (
                    parent_state[:3] if len(parent_state) >= 3 else np.append(parent_state, 0)
                )
                child_pos = child_state[:3] if len(child_state) >= 3 else np.append(child_state, 0)

                points = np.array([parent_pos, child_pos])
                self.server.scene.add_spline_catmull_rom(
                    f"{prefix}/success_{branch_idx}_segment_{i}",
                    positions=points,
                    color=success_color,
                    line_width=line_width,
                )

            # Add marker at the end of the branch
            end_pos = (
                path[-1].state[:3] if len(path[-1].state) >= 3 else np.append(path[-1].state, 0)
            )
            self.server.scene.add_icosphere(
                f"{prefix}/success_{branch_idx}_end",
                radius=0.1,
                position=tuple(end_pos),
                color=success_color,
            )

    def add_coordinate_frame(
        self,
        position: tuple[float, float, float] = (0, 0, 0),
        axes_length: float = 2.0,
        name: str = "/axes",
    ) -> None:
        """Add a coordinate frame to the scene.

        Args:
            position: Position of the frame
            axes_length: Length of the axes
            name: Name of the frame in the scene
        """
        self.server.scene.add_frame(
            name, wxyz=(1, 0, 0, 0), position=position, axes_length=axes_length
        )
