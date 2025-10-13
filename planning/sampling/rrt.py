"""RRT (Rapidly-exploring Random Tree) algorithm implementation."""

import numpy as np
from typing import List, Tuple, Optional, Union
from ..graph.node import Node, steer, get_nearest_node
from .sampler import UniformSampler, GoalBiasedSampler
from .collision_checker import CollisionChecker, EmptyCollisionChecker
from dataclasses import dataclass




@dataclass
class RRTConfig:
    max_iterations: int = 5000,
    step_size: float = 0.5,
    goal_tolerance: float = 0.5,
    goal_bias: float = 0.05,
    seed: Optional[int] = None

class RRT:
    """RRT (Rapidly-exploring Random Tree) path planner."""
    
    def __init__(
        self,
        start_state: Union[Tuple[float, ...], np.ndarray, List[float]],
        goal_state: Union[Tuple[float, ...], np.ndarray, List[float]],
        bounds: List[Tuple[float, float]],
        collision_checker: Optional[CollisionChecker] = None,
        config: RRTConfig = RRTConfig()
    ):
        """Initialize the RRT planner.
        
        Args:
            start_state: Starting state
            goal_state: Goal state
            bounds: List of (min, max) tuples for each dimension
            collision_checker: Collision checker instance (None for obstacle-free)
            max_iterations: Maximum number of iterations
            step_size: Maximum distance to extend in each iteration
            goal_tolerance: Distance threshold to consider goal reached
            goal_bias: Probability of sampling the goal state
            seed: Random seed for reproducibility
        """
        self.start_state = np.array(start_state)
        self.goal_state = np.array(goal_state)
        self.bounds = bounds
        self.max_iterations = config.max_iterations
        self.step_size = config.step_size
        self.goal_tolerance = config.goal_tolerance
        self.seed = config.seed
        
        # Collision checker
        if collision_checker is None:
            self.collision_checker = EmptyCollisionChecker()
        else:
            self.collision_checker = collision_checker
        
        # Sampler
        self.sampler = GoalBiasedSampler(
            bounds=bounds,
            goal_state=self.goal_state,
            goal_bias=config.goal_bias,
            seed=config.seed
        )
        
        # Tree
        self.nodes: List[Node] = []
        self.root: Optional[Node] = None
        self.goal_node: Optional[Node] = None
        
    def plan(self) -> Optional[List[Node]]:
        """Run the RRT algorithm to find a path.
        
        Returns:
            List of nodes from start to goal, or None if no path found
        """
        # Initialize tree with start node
        self.root = Node(state=self.start_state, parent=None, cost=0.0)
        self.nodes = [self.root]
        self.goal_node = None
        
        # Check if start is collision-free
        if not self.collision_checker.is_collision_free(self.start_state):
            print("Start state is in collision!")
            return None
        
        # Check if goal is collision-free
        if not self.collision_checker.is_collision_free(self.goal_state):
            print("Goal state is in collision!")
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
            if self.collision_checker.is_path_collision_free(
                nearest_node.state,
                new_node.state
            ):
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
        return np.linalg.norm(node.state - self.goal_state) <= self.goal_tolerance
    
    def _extract_path(self) -> List[Node]:
        """Extract the path from start to goal.
        
        Returns:
            List of nodes from start to goal
        """
        if self.goal_node is None:
            return []
        
        return self.goal_node.get_path_from_root()
    
    def get_tree_edges(self) -> List[Tuple[np.ndarray, np.ndarray]]:
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
            return float('inf')
        return self.goal_node.cost
    
    def get_stats(self) -> dict:
        """Get statistics about the planning process.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'num_nodes': len(self.nodes),
            'goal_reached': self.goal_node is not None,
            'path_length': self.get_path_length(),
            'path_nodes': len(self._extract_path()) if self.goal_node else 0
        }


@dataclass
class RRTConnectConfig:
    max_iterations: int = 5000,
    step_size: float = 0.5,
    seed: Optional[int] = None

class RRTConnect:
    """RRT-Connect algorithm (bidirectional RRT)."""
    
    def __init__(
        self,
        start_state: Union[Tuple[float, ...], np.ndarray, List[float]],
        goal_state: Union[Tuple[float, ...], np.ndarray, List[float]],
        bounds: List[Tuple[float, float]],
        collision_checker: Optional[CollisionChecker] = None,
        config: RRTConnectConfig = RRTConnectConfig()
    ):
        """Initialize the RRT-Connect planner.
        
        Args:
            start_state: Starting state
            goal_state: Goal state
            bounds: List of (min, max) tuples for each dimension
            collision_checker: Collision checker instance
            max_iterations: Maximum number of iterations
            step_size: Maximum distance to extend in each iteration
            seed: Random seed for reproducibility
        """
        self.start_state = np.array(start_state)
        self.goal_state = np.array(goal_state)
        self.bounds = bounds
        self.max_iterations = config.max_iterations
        self.step_size = config.step_size
        self.seed = config.seed
        
        # Collision checker
        if collision_checker is None:
            self.collision_checker = EmptyCollisionChecker()
        else:
            self.collision_checker = collision_checker
        
        # Sampler
        self.sampler = UniformSampler(bounds=bounds, seed=config.seed)
        
        # Trees
        self.start_nodes: List[Node] = []
        self.goal_nodes: List[Node] = []
        self.start_root: Optional[Node] = None
        self.goal_root: Optional[Node] = None
        self.connection_point_start: Optional[Node] = None
        self.connection_point_goal: Optional[Node] = None
    
    def plan(self) -> Optional[List[Node]]:
        """Run the RRT-Connect algorithm.
        
        Returns:
            List of nodes from start to goal, or None if no path found
        """
        # Initialize trees
        self.start_root = Node(state=self.start_state, parent=None, cost=0.0)
        self.goal_root = Node(state=self.goal_state, parent=None, cost=0.0)
        self.start_nodes = [self.start_root]
        self.goal_nodes = [self.goal_root]
        
        # Check if start and goal are collision-free
        if not self.collision_checker.is_collision_free(self.start_state):
            print("Start state is in collision!")
            return None
        
        if not self.collision_checker.is_collision_free(self.goal_state):
            print("Goal state is in collision!")
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
        
        print(f"Failed to connect trees after {self.max_iterations} iterations")
        return None
    
    def _extend_tree(self, tree: List[Node], target: Node) -> Optional[Node]:
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
        
        return None
    
    def _connect_tree(self, tree: List[Node], target: Node) -> Optional[Node]:
        """Try to connect a tree to a target node.
        
        Args:
            tree: The tree to connect
            target: The target node
            
        Returns:
            The connection node if successful, None otherwise
        """
        current_target = target
        
        while True:
            nearest = get_nearest_node(tree, current_target)
            distance = nearest.distance_to(current_target)
            
            if distance <= self.step_size:
                # Check if we can connect directly
                if self.collision_checker.is_path_collision_free(nearest.state, current_target.state):
                    # Create connection node
                    connection_node = Node(state=current_target.state, parent=nearest)
                    tree.append(connection_node)
                    return connection_node
                else:
                    return None
            
            # Extend towards target
            new_node = steer(nearest, current_target, self.step_size)
            
            if not self.collision_checker.is_path_collision_free(nearest.state, new_node.state):
                return None
            
            tree.append(new_node)
            current_target = new_node
    
    def _extract_path(self) -> List[Node]:
        """Extract the complete path from start to goal.
        
        Returns:
            List of nodes from start to goal
        """
        if self.connection_point_start is None or self.connection_point_goal is None:
            return []
        
        # Path from start to connection point
        start_path = self.connection_point_start.get_path_from_root()
        
        # Path from connection point to goal (reversed)
        goal_path = self.connection_point_goal.get_path_to_root()
        
        # Combine paths
        return start_path + goal_path
    
    def get_stats(self) -> dict:
        """Get statistics about the planning process.
        
        Returns:
            Dictionary with statistics
        """
        path = self._extract_path()
        return {
            'num_nodes_start': len(self.start_nodes),
            'num_nodes_goal': len(self.goal_nodes),
            'total_nodes': len(self.start_nodes) + len(self.goal_nodes),
            'trees_connected': self.connection_point_start is not None,
            'path_nodes': len(path)
        }

