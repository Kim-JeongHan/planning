from typing import Optional

import numpy as np


class Node:
    """

    Attributes:
        state: The state of the node (n-dimensional vector)
        parent: Reference to the parent node
        cost: Cost from the start node
        children: List of child nodes
    """

    def __init__(
        self,
        state: tuple[float, ...] | np.ndarray | list[float],
        parent: Optional["Node"] = None,
        cost: float = 0.0,
    ) -> None:
        """Initialize a node.

        Args:
            state: The state of the node (n-dimensional)
            parent: The parent node (None for root node)
            cost: The cost from the start node (default: 0.0)
        """
        self.state = np.array(state, dtype=float)
        self.parent = parent
        self.cost = cost
        self.children: list[Node] = []

        # Automatically add this node as a child of the parent
        if parent is not None:
            parent.children.append(self)

    @property
    def dim(self) -> int:
        """Get the dimension of the node state."""
        return len(self.state)

    def __getitem__(self, index: int) -> float:
        """Get a specific dimension of the state.

        Args:
            index: The dimension index

        Returns:
            The value at that dimension
        """
        return float(self.state[index])

    def distance_to(self, other: "Node") -> float:
        """Calculate Euclidean distance to another node.

        Args:
            other: The other node

        Returns:
            The Euclidean distance
        """
        return float(np.linalg.norm(self.state - other.state))

    def distance_to_state(self, state: tuple[float, ...] | np.ndarray | list[float]) -> float:
        """Calculate Euclidean distance to a state.

        Args:
            state: The target state

        Returns:
            The Euclidean distance
        """
        return float(np.linalg.norm(self.state - np.array(state)))

    def get_path_to_root(self) -> list["Node"]:
        """Get the path from this node to the root node.

        Returns:
            List of nodes from this node to the root (inclusive)
        """
        path = [self]
        current = self
        while current.parent is not None:
            current = current.parent
            path.append(current)
        return path

    def get_path_from_root(self) -> list["Node"]:
        """Get the path from the root node to this node.

        Returns:
            List of nodes from the root to this node (inclusive)
        """
        return list(reversed(self.get_path_to_root()))

    def get_path_states(self) -> np.ndarray:
        """Get the states of nodes from root to this node.

        Returns:
            Array of shape (N, dim) containing states
        """
        path = self.get_path_from_root()
        return np.array([node.state for node in path])

    def update_cost(self, new_cost: float) -> None:
        """Update the cost of this node and all descendants.

        Args:
            new_cost: The new cost from the start node
        """
        cost_diff = new_cost - self.cost
        self.cost = new_cost

        # Recursively update children's costs
        for child in self.children:
            child.update_cost(child.cost + cost_diff)

    def change_parent(self, new_parent: "Node", new_cost: float) -> None:
        """Change the parent of this node (used in RRT* rewiring).

        Args:
            new_parent: The new parent node
            new_cost: The new cost from the start node
        """
        # Remove from old parent's children
        if self.parent is not None:
            self.parent.children.remove(self)

        # Set new parent
        self.parent = new_parent
        new_parent.children.append(self)

        # Update cost
        self.update_cost(new_cost)

    def is_root(self) -> bool:
        """Check if this node is the root node.

        Returns:
            True if this is the root node
        """
        return self.parent is None

    def is_leaf(self) -> bool:
        """Check if this node is a leaf node.

        Returns:
            True if this node has no children
        """
        return len(self.children) == 0

    def get_depth(self) -> int:
        """Get the depth of this node (distance from root).

        Returns:
            The depth (root has depth 0)
        """
        depth = 0
        current = self
        while current.parent is not None:
            depth += 1
            current = current.parent
        return depth

    def __repr__(self) -> str:
        """String representation of the node."""
        state_str = ", ".join([f"{x:.2f}" for x in self.state])
        parent_str = (
            "None"
            if self.parent is None
            else f'({", ".join([f"{x:.2f}" for x in self.parent.state])})'
        )
        return f"Node(state=({state_str}), parent={parent_str}, cost={self.cost:.2f})"

    def __eq__(self, other: object) -> bool:
        """Check equality based on state."""
        if not isinstance(other, Node):
            return False
        return np.allclose(self.state, other.state)

    def __hash__(self) -> int:
        """Hash based on state."""
        return hash(tuple(self.state))


def distance(node1: Node, node2: Node) -> float:
    """Calculate distance between two nodes.

    Args:
        node1: First node
        node2: Second node

    Returns:
        Euclidean distance
    """
    return node1.distance_to(node2)


def steer(from_node: Node, to_node: Node, max_distance: float) -> Node:
    """Steer from one node towards another with a maximum distance.

    This creates a new node that is at most max_distance away from from_node
    in the direction of to_node.

    Args:
        from_node: The starting node
        to_node: The target node
        max_distance: Maximum distance to steer

    Returns:
        A new node in the direction of to_node
    """
    direction = to_node.state - from_node.state
    dist = float(np.linalg.norm(direction))

    if dist <= max_distance:
        # Target is within max_distance, return target state
        new_state = to_node.state
        new_cost = from_node.cost + dist
    else:
        # Steer towards target with max_distance
        direction = direction / dist  # Normalize
        new_state = from_node.state + direction * max_distance
        new_cost = from_node.cost + max_distance

    return Node(state=new_state, parent=from_node, cost=new_cost)


def get_nearest_node(nodes: list[Node], target: Node) -> Node:
    """Find the nearest node to a target node from a list of nodes.

    Args:
        nodes: List of nodes to search
        target: Target node

    Returns:
        The nearest node
    """
    if not nodes:
        raise ValueError("Node list is empty")

    distances = [node.distance_to(target) for node in nodes]
    nearest_idx = np.argmin(distances)
    return nodes[nearest_idx]


def get_nodes_within_radius(nodes: list[Node], center: Node, radius: float) -> list[Node]:
    """Get all nodes within a certain radius of a center node.

    Args:
        nodes: List of nodes to search
        center: Center node
        radius: Search radius

    Returns:
        List of nodes within the radius
    """
    return [node for node in nodes if node.distance_to(center) <= radius]
