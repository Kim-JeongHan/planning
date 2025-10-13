"""Obstacle generation and management module."""

import numpy as np
import viser
from typing import List, Tuple


class Obstacle:
    """Obstacle class."""
    
    def __init__(self, position: Tuple[float, float, float], size: Tuple[float, float, float]):
        """
        Args:
            position: Center position of the obstacle (x, y, z)
            size: Size of the obstacle (width, depth, height)
        """
        self.position = position
        self.size = size
    
    def get_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """Returns the bounds of the obstacle.
        
        Returns:
            ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        """
        x, y, z = self.position
        w, d, h = self.size
        
        return (
            (x - w/2, x + w/2),
            (y - d/2, y + d/2),
            (z - h/2, z + h/2)
        )
    
    def contains_point(self, point: Tuple[float, float, float]) -> bool:
        """Check if a point is inside the obstacle."""
        x, y, z = point
        bounds = self.get_bounds()
        
        return (bounds[0][0] <= x <= bounds[0][1] and
                bounds[1][0] <= y <= bounds[1][1] and
                bounds[2][0] <= z <= bounds[2][1])
    
    def intersects(self, other: 'Obstacle') -> bool:
        """Check if this obstacle intersects with another obstacle."""
        self_bounds = self.get_bounds()
        other_bounds = other.get_bounds()
        
        # AABB (Axis-Aligned Bounding Box) collision detection
        for i in range(3):
            if self_bounds[i][1] < other_bounds[i][0] or self_bounds[i][0] > other_bounds[i][1]:
                return False
        return True


def generate_random_obstacles(
    server: viser.ViserServer,
    map_size: int,
    num_obstacles: int,
    min_size: float = 0.5,
    max_size: float = 3.0,
    max_height: float = 2.0,
    min_height: float = 0.5,
    seed: int = None,
    color: Tuple[int, int, int] = (200, 100, 50),
    check_overlap: bool = True
) -> List[Obstacle]:
    """Generate box obstacles with random sizes in an n*n map.
    
    Args:
        server: Viser server instance
        map_size: Size of the map (n x n)
        num_obstacles: Number of obstacles to generate
        min_size: Minimum size of obstacles (width/depth)
        max_size: Maximum size of obstacles (width/depth)
        max_height: Maximum height of obstacles
        min_height: Minimum height of obstacles
        seed: Random seed for reproducibility
        color: Obstacle color (R, G, B)
        check_overlap: Whether to prevent obstacle overlap
    
    Returns:
        List of generated obstacles
    """
    if seed is not None:
        np.random.seed(seed)
    
    obstacles = []
    attempts = 0
    max_attempts = num_obstacles * 10  # Prevent infinite loop
    
    while len(obstacles) < num_obstacles and attempts < max_attempts:
        attempts += 1
        
        # Generate random size
        width = np.random.uniform(min_size, max_size)
        depth = np.random.uniform(min_size, max_size)
        height = np.random.uniform(min_height, max_height)
        
        # Generate random position (within map boundaries)
        # Ensure obstacles don't extend outside the map
        x = np.random.uniform(-map_size/2 + width/2, map_size/2 - width/2)
        y = np.random.uniform(-map_size/2 + depth/2, map_size/2 - depth/2)
        z = height / 2  # Start from ground
        
        new_obstacle = Obstacle(
            position=(x, y, z),
            size=(width, depth, height)
        )
        
        # Check overlap
        if check_overlap:
            overlap = False
            for existing_obstacle in obstacles:
                if new_obstacle.intersects(existing_obstacle):
                    overlap = True
                    break
            
            if overlap:
                continue
        
        # Add obstacle
        obstacles.append(new_obstacle)
        
        # Add box to Viser
        server.scene.add_box(
            f"/obstacles/box_{len(obstacles)}",
            dimensions=(width, depth, height),
            position=(x, y, z),
            color=color,
        )
    
    if len(obstacles) < num_obstacles:
        print(f"Warning: Only {len(obstacles)} out of {num_obstacles} obstacles were generated.")
    
    return obstacles