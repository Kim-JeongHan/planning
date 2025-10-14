"""Graph class for managing planning environment."""

import numpy as np

from .node import Node


class Edge:
    """Edge class for managing planning environment."""

    def __init__(self, node1: Node, node2: Node, cost: float) -> None:
        """Initialize the edge."""
        self.node1 = node1
        self.node2 = node2
        self.cost = cost


class Graph:
    """Graph class for managing planning environment."""

    def __init__(self) -> None:
        """Initialize the graph."""
        self.nodes: list[Node] = []
        self.edges: list[Edge] = []

    def add_node(self, node: Node) -> None:
        """Add a node to the graph.

        Args:
            node: The node to add
        """
        self.nodes.append(node)

    def add_edge(self, node1: Node, node2: Node, cost: float) -> None:
        """Add an edge to the graph.

        Args:
            node1: The first node
            node2: The second node
        """
        self.edges.append(Edge(node1, node2, cost))

    def reset(self) -> None:
        """Reset the graph."""
        self.nodes = []
        self.edges = []

    def get_nodes_num(self) -> int:
        """Get the number of nodes in the graph."""
        return len(self.nodes)

    def get_edges_num(self) -> int:
        """Get the number of edges in the graph."""
        return len(self.edges)

    def steer(self, from_node: Node, to_node: Node, max_distance: float) -> tuple[Node, float]:
        """Steer from one node towards another with a maximum distance."""
        direction = to_node.state - from_node.state
        dist = float(np.linalg.norm(direction))

        if dist <= max_distance:
            # Target is within max_distance, return target state
            new_state = to_node.state
            new_cost = dist
        else:
            # Steer towards target with max_distance
            direction = direction / dist  # Normalize
            new_state = from_node.state + direction * max_distance
            new_cost = max_distance

        new_node = Node(state=new_state)

        return new_node, new_cost
