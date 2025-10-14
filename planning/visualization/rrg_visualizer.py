"""Visualization utilities for RRG (Rapidly-exploring Random Graph) algorithms."""

from typing import TYPE_CHECKING

import numpy as np
import viser

from ..graph import Graph

if TYPE_CHECKING:
    from ..sampling.rrg import RRG


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
        planner: "RRG",
        edge_color: tuple[int, int, int] = (150, 150, 150),
        node_color: tuple[int, int, int] = (255, 255, 255),
        success_color: tuple[int, int, int] = (100, 150, 255),
        line_width: float = 1.2,
        prefix: str = "/graph",
    ) -> None:
        """Visualize all nodes and edges in the RRG graph.

        Args:
            planner: RRG planner instance
            edge_color: RGB color for general edges
            node_color: RGB color for general nodes
            success_color: RGB color for the final path
            line_width: Line width for edges
            prefix: Scene prefix
        """
        graph: Graph = planner.graph
        nodes = graph.nodes
        edges = graph.edges

        # Draw all edges
        for edge_idx, edge in enumerate(edges):
            p1 = edge.node1.state
            p2 = edge.node2.state
            p1_pos, p2_pos = p1[:3], p2[:3]
            points = np.array([p1_pos, p2_pos])

            self.server.scene.add_spline_catmull_rom(
                f"{prefix}/edge_{edge_idx}",
                points=points,
                color=edge_color,
                line_width=line_width,
            )

        # Draw all nodes
        for node_idx, node in enumerate(nodes):
            pos = node.state[:3]
            self.server.scene.add_icosphere(
                f"{prefix}/node_{node_idx}",
                radius=0.08,
                position=tuple(pos),
                color=node_color,
            )

        # If goal path exists, overlay it
        if planner.path:
            for i in range(len(planner.path) - 1):
                p1 = planner.path[i].state
                p2 = planner.path[i + 1].state
                p1_pos, p2_pos = p1[:3], p2[:3]
                points = np.array([p1_pos, p2_pos])

                self.server.scene.add_spline_catmull_rom(
                    f"{prefix}/success_segment_{i}",
                    points=points,
                    color=success_color,
                    line_width=line_width * 2.0,
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
            + (f" Path length: {len(planner.path)}" if planner.path else "")
        )
