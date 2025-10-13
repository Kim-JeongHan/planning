"""Collision checking utilities."""

import numpy as np
from typing import List, Tuple, Optional
from ..map.obstacles import Obstacle


class CollisionChecker:
    """Check for collisions with obstacles."""
    
    def __init__(self, obstacles: List[Obstacle]):
        """Initialize the collision checker.
        
        Args:
            obstacles: List of obstacles to check against
        """
        self.obstacles = obstacles
    
    def is_collision_free(self, state: np.ndarray) -> bool:
        """Check if a state is collision-free.
        
        Args:
            state: The state to check (first 3 dimensions are treated as position)
            
        Returns:
            True if collision-free, False otherwise
        """
        # Extract position from state (first 2 or 3 dimensions)
        if len(state) >= 3:
            position = tuple(state[:3])
        elif len(state) == 2:
            position = (state[0], state[1], 0.0)
        else:
            raise ValueError(f"State must have at least 2 dimensions, got {len(state)}")
        
        # Check against all obstacles
        for obstacle in self.obstacles:
            if obstacle.contains_point(position):
                return False
        
        return True
    
    def is_path_collision_free(
        self,
        from_state: np.ndarray,
        to_state: np.ndarray,
        resolution: float = 0.1
    ) -> bool:
        """Check if the straight-line path between two states is collision-free.
        
        Args:
            from_state: Starting state
            to_state: Ending state
            resolution: Step size for checking intermediate points
            
        Returns:
            True if the entire path is collision-free, False otherwise
        """
        # Calculate distance and number of steps
        distance = np.linalg.norm(to_state - from_state)
        num_steps = int(np.ceil(distance / resolution))
        
        if num_steps == 0:
            return self.is_collision_free(from_state)
        
        # Check intermediate points
        for i in range(num_steps + 1):
            t = i / num_steps
            intermediate_state = from_state + t * (to_state - from_state)
            
            if not self.is_collision_free(intermediate_state):
                return False
        
        return True


class EmptyCollisionChecker:
    """Collision checker for obstacle-free environments."""
    
    def is_collision_free(self, state: np.ndarray) -> bool:
        """Always returns True (no obstacles).
        
        Args:
            state: The state to check
            
        Returns:
            Always True
        """
        return True
    
    def is_path_collision_free(
        self,
        from_state: np.ndarray,
        to_state: np.ndarray,
        resolution: float = 0.1
    ) -> bool:
        """Always returns True (no obstacles).
        
        Args:
            from_state: Starting state
            to_state: Ending state
            resolution: Step size (unused)
            
        Returns:
            Always True
        """
        return True

