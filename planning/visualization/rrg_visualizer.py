"""Visualization utilities for RRG (Rapidly-exploring Random Graph) algorithms."""

import numpy as np
import viser

from ..graph import Graph
from ..sampling.base import RRGBase


class RRGVisualizer:
    """Visualizer for RRG algorithms."""

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
        """Visualize start and goal positions."""
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

    def visualize_graph(
        self,
        planner: RRGBase,
        success_color: tuple[int, int, int] = (100, 150, 255),
        failure_color: tuple[int, int, int] = (255, 100, 100),
        success_line_width: float = 3.0,
        failure_line_width: float = 1.2,
        prefix: str = "/graph",
    ) -> None:
        """Visualize all nodes and edges in the RRG graph.

        Args:
            planner: RRG planner instance
            success_color: RGB color for the final path (default: blue)
            failure_color: RGB color for failed edges (default: red)
            success_line_width: Line width for success path edges
            failure_line_width: Line width for failure edges
            prefix: Scene prefix
        """
        graph: Graph = planner.graph
        nodes = graph.nodes
        edges = graph.edges

        # Build set of edges in the success path
        success_edge_set = set()
        if planner.path:
            for i in range(len(planner.path) - 1):
                node1 = planner.path[i]
                node2 = planner.path[i + 1]
                # Add both directions since edges are undirected
                success_edge_set.add((id(node1), id(node2)))
                success_edge_set.add((id(node2), id(node1)))

        # Separate edges into success and failure
        success_edge_points = []
        failure_edge_points = []

        for edge in edges:
            p1 = edge.node1.state[:3]
            p2 = edge.node2.state[:3]
            edge_pair = (id(edge.node1), id(edge.node2))

            if edge_pair in success_edge_set:
                success_edge_points.append([p1, p2])
            else:
                failure_edge_points.append([p1, p2])

        # Draw failure edges (red)
        if failure_edge_points:
            failure_points_array = np.array(failure_edge_points)
            self.server.scene.add_line_segments(
                f"{prefix}/failure_edges",
                points=failure_points_array,
                colors=failure_color,
                line_width=failure_line_width,
            )

        # Draw success path edges (blue)
        if success_edge_points:
            success_points_array = np.array(success_edge_points)
            self.server.scene.add_line_segments(
                f"{prefix}/success_path",
                points=success_points_array,
                colors=success_color,
                line_width=success_line_width,
            )

            # Mark goal node
            goal_node = planner.goal_node
            if goal_node is not None:
                goal_pos = (
                    goal_node.state[:3]
                    if len(goal_node.state) >= 3
                    else np.append(goal_node.state, 0)
                )
                self.server.scene.add_icosphere(
                    f"{prefix}/goal_marker",
                    radius=0.15,
                    position=tuple(goal_pos),
                    color=success_color,
                )

        print(
            f"Visualized RRG with {len(nodes)} nodes and {len(edges)} edges."
            + (
                f" Success: {len(success_edge_points)}, Failure: {len(failure_edge_points)}"
                if edges
                else ""
            )
            + (f" | Path length: {len(planner.path)}" if planner.path else "")
        )
