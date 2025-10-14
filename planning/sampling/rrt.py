"""RRT (Rapidly-exploring Random Tree) algorithm implementation."""

import numpy as np
from pydantic import BaseModel, field_validator

from ..collision import CollisionChecker
from ..graph import Graph, Node, get_nearest_node, get_nodes_within_radius, steer
from ..search import AStar
from .base import RRGBase, RRTBase
from .sampler import GoalBiasedSampler, Sampler, UniformSampler


class RRTConfig(BaseModel):
    """Configuration for RRT algorithm."""

    sampler: type[Sampler] = GoalBiasedSampler
    max_iterations: int = 5000
    step_size: float = 0.5
    goal_tolerance: float = 0.5
    goal_bias: float = 0.05
    seed: int | None = None

    @field_validator("sampler")
    @classmethod
    def validate_sampler(cls, v: type[Sampler]) -> type[Sampler]:
        if not isinstance(v, type):
            raise TypeError("sampler must be a class type")
        if not issubclass(v, Sampler):
            raise TypeError("sampler must inherit from Sampler")
        return v


class RRT(RRTBase):
    """RRT (Rapidly-exploring Random Tree) path planner."""

    def __init__(
        self,
        start_state: tuple[float, ...] | np.ndarray | list[float],
        goal_state: tuple[float, ...] | np.ndarray | list[float],
        bounds: list[tuple[float, float]],
        collision_checker: CollisionChecker | None = None,
        config: RRTConfig | None = None,
    ) -> None:
        """Initialize the RRT planner.

        Args:
            start_state: Starting state
            goal_state: Goal state
            bounds: List of (min, max) tuples for each dimension
            collision_checker: Collision checker instance (None for obstacle-free)
            config : RRTConfig
        """
        if config is None:
            config = RRTConfig()

        # Initialize base class
        super().__init__(
            start_state=start_state,
            goal_state=goal_state,
            bounds=bounds,
            collision_checker=collision_checker,
            max_iterations=config.max_iterations,
            step_size=config.step_size,
            goal_tolerance=config.goal_tolerance,
            seed=config.seed,
        )

        # Sampler
        if config.sampler is GoalBiasedSampler:
            self.sampler = config.sampler(  # type: ignore[call-arg]
                bounds=bounds,
                goal_state=self.goal_state,
                goal_bias=config.goal_bias,
                seed=config.seed,
            )
        else:
            self.sampler = config.sampler(bounds=bounds, seed=config.seed)

        # Tree
        self.nodes: list[Node] = []

    def plan(self) -> list[Node] | None:
        """Run the RRT algorithm to find a path.

        Returns:
            List of nodes from start to goal, or None if no path found
        """
        # Initialize tree with start node
        self.root = Node(state=self.start_state, parent=None, cost=0.0)
        self.nodes = [self.root]
        self.goal_node = None

        # Check if start and goal are collision-free
        if not self._check_start_goal_collision():
            return None

        # Main RRT loop
        for iteration in range(self.max_iterations):
            # Sample a random state
            random_state = self.sampler.sample()
            random_node = Node(state=random_state)

            # Find nearest node in the tree
            nearest_node = get_nearest_node(self.nodes, random_node)

            # Steer towards the random state
            new_node = steer(nearest_node, random_node, self.step_size)

            # Check if the path is collision-free
            if self.collision_checker.is_path_collision_free(nearest_node.state, new_node.state):
                # Add new node to the tree
                self.nodes.append(new_node)

                # Check if goal is reached
                if self._is_goal_reached(new_node):
                    self.goal_node = new_node
                    print(f"Goal reached in {iteration + 1} iterations!")
                    return self._extract_path()

        print(f"Failed to reach goal after {self.max_iterations} iterations")
        return None

    def _is_goal_reached(self, node: Node) -> bool:
        """Check if a node is close enough to the goal.

        Args:
            node: The node to check

        Returns:
            True if within goal tolerance
        """
        return bool(np.linalg.norm(node.state - self.goal_state) <= self.goal_tolerance)

    def _extract_path(self) -> list[Node]:
        """Extract the path from start to goal.

        Returns:
            List of nodes from start to goal
        """
        if self.goal_node is None:
            return []

        return self.goal_node.get_path_from_root()

    def get_tree_edges(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Get all edges in the tree for visualization.

        Returns:
            List of (parent_state, child_state) tuples
        """
        edges = []
        for node in self.nodes:
            if node.parent is not None:
                edges.append((node.parent.state, node.state))
        return edges

    def get_path_length(self) -> float:
        """Get the length of the path.

        Returns:
            Path length, or inf if no path found
        """
        if self.goal_node is None:
            return float("inf")
        return self.goal_node.cost

    def get_stats(self) -> dict:
        """Get statistics about the planning process.

        Returns:
            Dictionary with statistics
        """
        return {
            "num_nodes": len(self.nodes),
            "goal_reached": self.goal_node is not None,
            "path_length": self.get_path_length(),
            "path_nodes": len(self._extract_path()) if self.goal_node else 0,
        }

    def get_all_nodes(self) -> list[Node]:
        """Get all nodes in the tree for visualization.

        Returns:
            List of all nodes explored during planning
        """
        return self.nodes

    def get_goal_node(self) -> Node | None:
        """Get the goal node for visualization.

        Returns:
            The node that reached the goal, or None if planning failed
        """
        return self.goal_node


class RRTConnectConfig(BaseModel):
    """Configuration for RRT-Connect algorithm."""

    max_iterations: int = 5000
    step_size: float = 0.5
    goal_tolerance: float = 0.5
    seed: int | None = None


class RRTConnect(RRTBase):
    """RRT-Connect algorithm (bidirectional RRT)."""

    def __init__(
        self,
        start_state: tuple[float, ...] | np.ndarray | list[float],
        goal_state: tuple[float, ...] | np.ndarray | list[float],
        bounds: list[tuple[float, float]],
        collision_checker: CollisionChecker | None = None,
        config: RRTConnectConfig | None = None,
    ) -> None:
        """Initialize the RRT-Connect planner.

        Args:
            start_state: Starting state
            goal_state: Goal state
            bounds: List of (min, max) tuples for each dimension
            collision_checker: Collision checker instance
            config: RRTConnectConfig instance
        """
        if config is None:
            config = RRTConnectConfig()

        # Initialize base class
        super().__init__(
            start_state=start_state,
            goal_state=goal_state,
            bounds=bounds,
            collision_checker=collision_checker,
            max_iterations=config.max_iterations,
            step_size=config.step_size,
            goal_tolerance=config.goal_tolerance,
            seed=config.seed,
        )

        # Sampler
        self.sampler = UniformSampler(bounds=bounds, seed=config.seed)

        # Trees
        self.start_nodes: list[Node] = []
        self.goal_nodes: list[Node] = []
        self.start_root: Node | None = None
        self.goal_root: Node | None = None
        self.connection_point_start: Node | None = None
        self.connection_point_goal: Node | None = None
        self.swapped: bool = False  # Track if trees are swapped

        # Track failed extension attempts for visualization
        self.failed_nodes: list[Node] = []

    def plan(self) -> list[Node] | None:
        """Run the RRT-Connect algorithm.

        Returns:
            List of nodes from start to goal, or None if no path found
        """
        # Initialize trees
        self.start_root = Node(state=self.start_state, parent=None, cost=0.0)
        self.goal_root = Node(state=self.goal_state, parent=None, cost=0.0)
        self.start_nodes = [self.start_root]
        self.goal_nodes = [self.goal_root]
        self.failed_nodes = []  # Reset failed attempts

        # Check if start and goal are collision-free
        if not self._check_start_goal_collision():
            return None

        # Main loop
        for iteration in range(self.max_iterations):
            # Sample random state
            random_state = self.sampler.sample()
            random_node = Node(state=random_state)

            # Extend start tree
            new_node_start = self._extend_tree(self.start_nodes, random_node)
            if new_node_start is not None:
                # Try to connect to goal tree
                connection_node = self._connect_tree(self.goal_nodes, new_node_start)

                if connection_node is not None:
                    self.connection_point_start = new_node_start
                    self.connection_point_goal = connection_node
                    print(f"Trees connected in {iteration + 1} iterations!")
                    return self._extract_path()

            # Swap trees (alternate which tree extends)
            self.start_nodes, self.goal_nodes = self.goal_nodes, self.start_nodes
            self.swapped = not self.swapped

        print(f"Failed to connect trees after {self.max_iterations} iterations")
        return None

    def _extend_tree(self, tree: list[Node], target: Node) -> Node | None:
        """Extend a tree towards a target.

        Args:
            tree: The tree to extend
            target: The target node

        Returns:
            The new node if successful, None otherwise
        """
        nearest = get_nearest_node(tree, target)
        new_node = steer(nearest, target, self.step_size)

        if self.collision_checker.is_path_collision_free(nearest.state, new_node.state):
            tree.append(new_node)
            return new_node

        # Track failed extension attempt
        failed_node = Node(state=new_node.state, parent=nearest)
        self.failed_nodes.append(failed_node)
        return None

    def _connect_tree(self, tree: list[Node], target: Node) -> Node | None:
        """Try to connect a tree to a target node.

        Args:
            tree: The tree to connect
            target: The target node (from the other tree)

        Returns:
            The last node added to this tree that connects to target, None otherwise
        """
        nearest = get_nearest_node(tree, target)

        while True:
            dist = nearest.distance_to(target)

            if dist <= self.step_size:
                if self.collision_checker.is_path_collision_free(nearest.state, target.state):
                    target_node = Node(state=target.state, parent=nearest)
                    tree.append(target_node)
                    return target_node
                failed_node = Node(state=target.state, parent=nearest)
                self.failed_nodes.append(failed_node)
                return None

            new_node = steer(nearest, target, self.step_size)

            if not self.collision_checker.is_path_collision_free(nearest.state, new_node.state):
                failed_node = Node(state=new_node.state, parent=nearest)
                self.failed_nodes.append(failed_node)
                return None

            tree.append(new_node)
            prev = nearest
            nearest = new_node

            if np.allclose(prev.state, nearest.state):
                return None

    def _extract_path(self) -> list[Node]:
        """Extract the complete path from start to goal."""
        if self.connection_point_start is None or self.connection_point_goal is None:
            return []

        # If trees were swapped, swap connection points back to original order
        conn_start = self.connection_point_start
        conn_goal = self.connection_point_goal
        if self.swapped:
            conn_start, conn_goal = conn_goal, conn_start

        path_start = []
        current_node: Node | None = conn_start
        while current_node is not None:
            path_start.append(current_node)
            current_node = current_node.parent
        path_start.reverse()  # root → connection

        path_goal = []
        current_node = conn_goal
        while current_node is not None:
            path_goal.append(current_node)
            current_node = current_node.parent

        full_path = path_start + path_goal
        return full_path

    def get_stats(self) -> dict:
        """Get statistics about the planning process.

        Returns:
            Dictionary with statistics
        """
        path = self._extract_path()
        return {
            "num_nodes_start": len(self.start_nodes),
            "num_nodes_goal": len(self.goal_nodes),
            "num_failed_attempts": len(self.failed_nodes),
            "total_nodes": len(self.start_nodes) + len(self.goal_nodes) + len(self.failed_nodes),
            "trees_connected": self.connection_point_start is not None,
            "path_nodes": len(path),
        }

    def get_path_length(self) -> float:
        """Get the length of the path.

        Returns:
            Path length, or inf if no path found
        """
        path = self._extract_path()
        if not path:
            return float("inf")

        # Calculate total path length
        total_length = 0.0
        for i in range(len(path) - 1):
            total_length += float(np.linalg.norm(path[i + 1].state - path[i].state))
        return total_length

    def get_all_nodes(self) -> list[Node]:
        """Get all nodes in both trees for visualization.

        Returns:
            List of all nodes explored during planning (including failed attempts)
        """
        return self.start_nodes + self.goal_nodes + self.failed_nodes

    def get_goal_node(self) -> Node | None:
        """Get the connection point for visualization.

        Returns:
            The connection point in the start tree, or None if trees didn't connect
        """
        # Return the connection point that's in the path to the goal
        if self.connection_point_start is None:
            return None

        # If trees were swapped, return the appropriate connection point
        if self.swapped:
            return self.connection_point_goal
        return self.connection_point_start


class RRTStarConfig(BaseModel):
    """Configuration for RRT* algorithm."""

    sampler: type[Sampler] = GoalBiasedSampler
    max_iterations: int = 5000
    step_size: float = 0.5
    goal_tolerance: float = 0.5
    goal_bias: float = 0.05
    radius_gain: float = 1.0
    seed: int | None = None


class RRTStar(RRGBase):
    """RRT* (RRT-Star) path planner."""

    def __init__(
        self,
        start_state: tuple[float, ...] | np.ndarray | list[float],
        goal_state: tuple[float, ...] | np.ndarray | list[float],
        bounds: list[tuple[float, float]],
        collision_checker: CollisionChecker | None = None,
        config: RRTStarConfig | None = None,
    ) -> None:
        """Initialize the RRT* planner.

        Args:
            start_state: Starting state
            goal_state: Goal state
            bounds: List of (min, max) tuples for each dimension
            collision_checker: Collision checker instance (None for obstacle-free)
            config : RRTStarConfig
        """
        if config is None:
            config = RRTStarConfig()

        # Initialize base class
        super().__init__(
            start_state=start_state,
            goal_state=goal_state,
            bounds=bounds,
            collision_checker=collision_checker,
            max_iterations=config.max_iterations,
            step_size=config.step_size,
            goal_tolerance=config.goal_tolerance,
            seed=config.seed,
        )

        self.radius_gain = config.radius_gain

        # Sampler
        if config.sampler is GoalBiasedSampler:
            self.sampler = config.sampler(  # type: ignore[call-arg]
                bounds=bounds,
                goal_state=self.goal_state,
                goal_bias=config.goal_bias,
                seed=config.seed,
            )
        else:
            self.sampler = config.sampler(bounds=bounds, seed=config.seed)

        # Graph
        self.graph = Graph()

        # A*
        self.astar = AStar(self.graph)

        self.path: list[Node] | None = None

    def plan(self) -> list[Node] | None:
        """Run the RRG algorithm."""

        self.root = Node(state=self.start_state)
        self.path = None
        self.goal_node = None

        # graph initialization
        self.graph.reset()
        self.graph.add_node(self.root)

        if not self._check_start_goal_collision():
            return None

        # Main RRG loop
        for iteration in range(self.max_iterations):
            # Sample a random state
            random_state = self.sampler.sample()
            random_node = Node(state=random_state)

            # Find nearest node in the graph
            nearest_node = get_nearest_node(self.graph.nodes, random_node)
            new_node = steer(nearest_node, random_node, self.step_size)

            # Check if the path is collision-free
            if self.collision_checker.is_path_collision_free(nearest_node.state, new_node.state):
                neighbor_nodes = self.get_near_node(new_node)
                self.graph.add_node(new_node)

                if neighbor_nodes is None:
                    continue

                min_cost_node = self.get_min_cost_node([*neighbor_nodes, nearest_node], new_node)

                if min_cost_node is not None:
                    best_cost = min_cost_node.cost + min_cost_node.distance_to(new_node)
                    new_node.change_parent(min_cost_node, best_cost)
                    self.graph.add_edge(
                        min_cost_node, new_node, min_cost_node.distance_to(new_node)
                    )

                else:
                    continue

                for neighbor_node in neighbor_nodes:
                    if not self.collision_checker.is_path_collision_free(
                        neighbor_node.state, new_node.state
                    ):
                        continue

                    if new_node.cost + neighbor_node.distance_to(new_node) < neighbor_node.cost:
                        old_parent = neighbor_node.parent
                        if old_parent:
                            self.graph.remove_edge(neighbor_node, old_parent)
                        self.graph.add_edge(
                            new_node, neighbor_node, neighbor_node.distance_to(new_node)
                        )
                        neighbor_node.change_parent(
                            new_node, new_node.cost + neighbor_node.distance_to(new_node)
                        )

                if self._is_goal_reached(new_node):
                    self.goal_node = new_node
                    print(f"Goal reached in {iteration + 1} iterations!")
                    self.path = self.astar.search(self.root, self.goal_node)
                    return self.path

        return None

    def get_near_node(self, target: Node) -> list[Node]:
        """Get the near nodes of the target node."""
        num_nodes = len(self.graph.nodes)

        if num_nodes <= 1:
            return []

        radius = min(
            self.step_size, self.radius_gain * np.power(np.log(num_nodes) / num_nodes, 1 / self.dim)
        )
        # radius = self.radius_gain * np.power(np.log(num_nodes) / num_nodes, 1 / self.dim)

        return get_nodes_within_radius(self.graph.nodes, target, radius)

    def get_min_cost_node(self, nodes: list[Node], target: Node) -> Node | None:
        min_cost = float("inf")
        min_cost_node = None

        for node in nodes:
            cost = node.cost + node.distance_to(target)
            if cost < min_cost:
                min_cost = cost
                min_cost_node = node

        return min_cost_node

    def _is_goal_reached(self, node: Node) -> bool:
        """Check if a node is close enough to the goal.

        Args:
            node: The node to check

        Returns:
            True if within goal tolerance
        """
        return bool(np.linalg.norm(node.state - self.goal_state) <= self.goal_tolerance)

    def get_stats(self) -> dict[str, float | int | bool | None]:
        """Get statistics about the planning process.

        Returns:
            Dictionary with number of nodes and edges
        """
        path_length = self.get_path_length()
        return {
            "num_nodes": len(self.graph.nodes),
            "goal_reached": self.goal_node is not None,
            "num_edges": len(self.graph.edges),
            "path_length": path_length if path_length > 0 else None,
            "path_nodes": len(self.path) if self.path else None,
        }

    def get_path_length(self) -> float:
        """Get the total length of the current path."""
        if not self.path:
            return 0.0

        distances = [n.distance_to(n.parent) for n in self.path if n.parent]
        return float(sum(distances)) if distances else 0.0

    def get_all_nodes(self) -> list[Node]:
        return self.graph.nodes

    def get_goal_node(self) -> Node | None:
        return self.goal_node
