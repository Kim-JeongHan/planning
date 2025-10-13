"""Map class for managing planning environment."""

import numpy as np
import viser
from typing import List, Tuple, Optional
from .obstacles import Obstacle, generate_random_obstacles


class Map:
    """Planning map with obstacles and boundaries."""
    
    def __init__(
        self,
        size: float,
        z_range: Tuple[float, float] = (0.5, 2.5)
    ):
        """Initialize a map.
        
        Args:
            size: Size of the square map (n x n)
            z_range: (min_z, max_z) height range
        """
        self.size = size
        self.z_min, self.z_max = z_range
        self.obstacles: List[Obstacle] = []
    
    def get_bounds(self) -> List[Tuple[float, float]]:
        """Get the bounds of the map for sampling.
        
        Returns:
            List of (min, max) tuples for each dimension [x, y, z]
        """
        half_size = self.size / 2
        return [
            (-half_size, half_size),  # x bounds
            (-half_size, half_size),  # y bounds
            (self.z_min, self.z_max)  # z bounds
        ]
    
    def get_bounds_2d(self) -> List[Tuple[float, float]]:
        """Get 2D bounds (x, y only).
        
        Returns:
            List of (min, max) tuples for x and y
        """
        half_size = self.size / 2
        return [
            (-half_size, half_size),  # x bounds
            (-half_size, half_size),  # y bounds
        ]
    
    def add_obstacle(self, obstacle: Obstacle):
        """Add an obstacle to the map.
        
        Args:
            obstacle: Obstacle to add
        """
        self.obstacles.append(obstacle)
    
    def add_obstacles(self, obstacles: List[Obstacle]):
        """Add multiple obstacles to the map.
        
        Args:
            obstacles: List of obstacles to add
        """
        self.obstacles.extend(obstacles)
    
    def generate_random_obstacles(
        self,
        server: viser.ViserServer,
        num_obstacles: int,
        min_size: float = 0.5,
        max_size: float = 3.0,
        min_height: Optional[float] = None,
        max_height: Optional[float] = None,
        seed: Optional[int] = None,
        color: Tuple[int, int, int] = (200, 100, 50),
        check_overlap: bool = True
    ) -> List[Obstacle]:
        """Generate random obstacles in the map.
        
        Args:
            server: Viser server instance for visualization
            num_obstacles: Number of obstacles to generate
            min_size: Minimum obstacle size
            max_size: Maximum obstacle size
            min_height: Minimum obstacle height (defaults to z_min)
            max_height: Maximum obstacle height (defaults to z_max)
            seed: Random seed
            color: Obstacle color
            check_overlap: Whether to prevent overlap
            
        Returns:
            List of generated obstacles
        """
        if min_height is None:
            min_height = self.z_min
        if max_height is None:
            max_height = self.z_max
        
        obstacles = generate_random_obstacles(
            server=server,
            map_size=self.size,
            num_obstacles=num_obstacles,
            min_size=min_size,
            max_size=max_size,
            min_height=min_height,
            max_height=max_height,
            seed=seed,
            color=color,
            check_overlap=check_overlap
        )
        
        self.obstacles.extend(obstacles)
        return obstacles
    
    def visualize_bounds(
        self,
        server: viser.ViserServer,
        ground_color: Tuple[int, int, int] = (150, 150, 150),
        boundary_color: Tuple[int, int, int] = (100, 100, 100)
    ):
        """Visualize map boundaries.
        
        Args:
            server: Viser server instance
            ground_color: Color of the ground plane
            boundary_color: Color of the boundary walls
        """
        # Ground plane
        server.scene.add_box(
            "/map/ground",
            dimensions=(self.size, self.size, 0.01),
            position=(0, 0, -0.005),
            color=ground_color,
        )
        
        # Boundary lines (4 walls)
        boundary_height = 0.5
        boundary_thickness = 0.05
        
        # Top boundary
        server.scene.add_box(
            "/map/boundary_top",
            dimensions=(self.size, boundary_thickness, boundary_height),
            position=(0, self.size/2, boundary_height/2),
            color=boundary_color,
        )
        
        # Bottom boundary
        server.scene.add_box(
            "/map/boundary_bottom",
            dimensions=(self.size, boundary_thickness, boundary_height),
            position=(0, -self.size/2, boundary_height/2),
            color=boundary_color,
        )
        
        # Left boundary
        server.scene.add_box(
            "/map/boundary_left",
            dimensions=(boundary_thickness, self.size, boundary_height),
            position=(-self.size/2, 0, boundary_height/2),
            color=boundary_color,
        )
        
        # Right boundary
        server.scene.add_box(
            "/map/boundary_right",
            dimensions=(boundary_thickness, self.size, boundary_height),
            position=(self.size/2, 0, boundary_height/2),
            color=boundary_color,
        )
    
    def is_valid_state(self, state: np.ndarray) -> bool:
        """Check if a state is within map bounds.
        
        Args:
            state: State to check (at least 2D or 3D)
            
        Returns:
            True if state is within bounds
        """
        half_size = self.size / 2
        
        # Check x, y bounds
        if not (-half_size <= state[0] <= half_size):
            return False
        if len(state) > 1 and not (-half_size <= state[1] <= half_size):
            return False
        
        # Check z bounds if 3D
        if len(state) > 2 and not (self.z_min <= state[2] <= self.z_max):
            return False
        
        return True
    
    def clear_obstacles(self):
        """Clear all obstacles from the map."""
        self.obstacles.clear()
    
    def __repr__(self) -> str:
        """String representation of the map."""
        return f"Map(size={self.size}, z_range=({self.z_min}, {self.z_max}), obstacles={len(self.obstacles)})"

