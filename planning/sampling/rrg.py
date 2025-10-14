"""RRG (Rapidly-exploring Random Graph) algorithm implementation."""

import numpy as np
from pydantic import BaseModel, field_validator

from ..collision import CollisionChecker
from ..graph import Graph, Node, get_nearest_node, get_nodes_within_radius
from ..search import AStar
from .rrt import RRTBase
from .sampler import GoalBiasedSampler, Sampler


class RRGConfig(BaseModel):
    """Configuration for RRG algorithm."""

    sampler: type[Sampler] = GoalBiasedSampler
    max_iterations: int = 1000
    radius_gain: float = 0.5  # gamma_RRG (scaling factor for connection radius)

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


class RRGBase(RRTBase):
    """Base class for RRG algorithm."""

    def __init__(
        self,
        start_state: tuple[float, ...] | np.ndarray | list[float],
        goal_state: tuple[float, ...] | np.ndarray | list[float],
        bounds: list[tuple[float, float]],
        collision_checker: CollisionChecker | None = None,
        max_iterations: int = 1000,
        step_size: float = 0.5,
        goal_tolerance: float = 0.5,
        seed: int | None = None,
    ) -> None:
        """Initialize the RRG base class.

        Args:
            start_state: Starting state
            goal_state: Goal state
            bounds: List of (min, max) tuples for each dimension
            collision_checker: Collision checker instance
            max_iterations: Maximum number of iterations to run
            step_size: Maximum distance to extend tree at each iteration
            goal_tolerance: Distance threshold to consider goal reached
            seed: Random seed for reproducibility
        """
        super().__init__(
            start_state=start_state,
            goal_state=goal_state,
            bounds=bounds,
            collision_checker=collision_checker,
            max_iterations=max_iterations,
            step_size=step_size,
            goal_tolerance=goal_tolerance,
            seed=seed,
        )

        # Graph
        self.graph = Graph()
        self.path: list[Node] | None = None

        # A*
        self.astar = AStar(self.graph)


class RRG(RRGBase):
    """RRG (Rapidly-exploring Random Graph) algorithm."""

    def __init__(
        self,
        start_state: tuple[float, ...] | np.ndarray | list[float],
        goal_state: tuple[float, ...] | np.ndarray | list[float],
        bounds: list[tuple[float, float]],
        collision_checker: CollisionChecker | None = None,
        config: RRGConfig | None = None,
    ) -> None:
        """Initialize the RRG planner.

        Args:
            start_state: Starting state
            goal_state: Goal state
            bounds: List of (min, max) tuples for each dimension
            collision_checker: Collision checker instance
            config: RRGConfig instance
        """
        if config is None:
            config = RRGConfig()

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
            new_node, new_cost = self.graph.steer(nearest_node, random_node, self.step_size)

            # Check if the path is collision-free
            if self.collision_checker.is_path_collision_free(nearest_node.state, new_node.state):
                self.graph.add_node(new_node)
                self.graph.add_edge(nearest_node, new_node, new_cost)

                neighbor_nodes = self.get_near_node(new_node)
                if neighbor_nodes is None:
                    continue

                for neighbor_node in neighbor_nodes:
                    if self.collision_checker.is_path_collision_free(
                        neighbor_node.state, new_node.state
                    ):
                        cost = neighbor_node.distance_to(new_node)
                        self.graph.add_edge(neighbor_node, new_node, cost)

                        # Check if goal is reached
                        if self._is_goal_reached(new_node):
                            self.goal_node = new_node
                            print(f"Goal reached in {iteration + 1} iterations!")
                            self.path = self.astar.search(self.root, self.goal_node)
                            return self.path

        return None

    # def _extend_tree(self, tree: list[Node], target: Node) -> Node | None:
    def get_near_node(self, target: Node) -> list[Node] | None:
        """Get the near nodes of the target node."""
        num_nodes = len(self.graph.nodes)

        if num_nodes <= 1:
            return []

        radius = min(
            self.step_size, self.radius_gain * np.power(np.log(num_nodes) / num_nodes, 1 / self.dim)
        )

        return get_nodes_within_radius(self.graph.nodes, target, radius)

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
        if self.path is None:
            return float("inf")

        distances = [node.distance_to(node.parent) for node in self.path if node.parent is not None]

        if not distances:
            return float("inf")

        sum_dis = float(np.sum(distances))
        return float("inf") if np.isinf(sum_dis) else sum_dis

    def get_all_nodes(self) -> list[Node]:
        return self.graph.nodes

    def get_goal_node(self) -> Node | None:
        return self.goal_node
