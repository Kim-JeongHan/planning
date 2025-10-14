"""Graph class for managing planning environment."""

import numpy as np

from .node import Node


class Edge:
    """Undirected edge class for managing planning environment."""

    def __init__(self, node1: Node, node2: Node, cost: float = 0.0) -> None:
        """Initialize the undirected edge.

        Args:
            node1: First node
            node2: Second node
            cost: Cost of the edge
        """
        self.node1 = node1
        self.node2 = node2
        self.cost = cost

    def __eq__(self, other: object) -> bool:
        """Check equality of two edges (undirected).

        Two edges are equal if they connect the same nodes regardless of order.
        """
        if not isinstance(other, Edge):
            return False
        return (self.node1 is other.node1 and self.node2 is other.node2) or (
            self.node1 is other.node2 and self.node2 is other.node1
        )

    def __hash__(self) -> int:
        """Return hash of the edge (undirected).

        Hash is the same regardless of node order.
        """
        # Use frozenset to ensure order-independent hashing
        return hash(frozenset([id(self.node1), id(self.node2)]))

    def contains_node(self, node: Node) -> bool:
        """Check if the edge contains a given node.

        Args:
            node: Node to check

        Returns:
            True if the edge contains the node
        """
        return self.node1 is node or self.node2 is node

    def get_other_node(self, node: Node) -> Node:
        """Get the other node in the edge.

        Args:
            node: One node in the edge

        Returns:
            The other node

        Raises:
            ValueError: If the given node is not in the edge
        """
        if self.node1 is node:
            return self.node2
        elif self.node2 is node:
            return self.node1
        else:
            raise ValueError("Node is not in the edge")

    def __repr__(self) -> str:
        """Return string representation of the edge."""
        return f"Edge(node1={self.node1}, node2={self.node2}, cost={self.cost})"


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
            cost: The cost of the edge
        """
        new_edge = Edge(node1, node2, cost)
        # Check if edge already exists (order-independent)
        if new_edge not in self.edges:
            self.edges.append(new_edge)

    def remove_edge(self, node1: Node, node2: Node) -> None:
        """Remove an edge from the graph (undirected).

        Args:
            node1: The first node
            node2: The second node
        """
        edge_to_remove = Edge(node1, node2)
        if edge_to_remove in self.edges:
            self.edges.remove(edge_to_remove)

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
