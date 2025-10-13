"""Map class for managing planning environment."""

from typing import Literal

import numpy as np
import viser

from .obstacles import (
    Obstacle,
    _generate_random_box_obstacle,
    _generate_random_sphere_obstacle,
    _visualize_obstacle,
)


class Map:
    """Planning map with obstacles and boundaries."""

    def __init__(self, size: float, z_range: tuple[float, float] = (0.5, 2.5)) -> None:
        """Initialize a map.

        Args:
            size: Size of the square map (n x n)
            z_range: (min_z, max_z) height range
        """
        self.size = size
        self.z_min, self.z_max = z_range
        self.obstacles: list[Obstacle] = []

    def get_bounds(self) -> list[tuple[float, float]]:
        """Get the bounds of the map for sampling.

        Returns:
            List of (min, max) tuples for each dimension [x, y, z]
        """
        half_size = self.size / 2
        return [
            (-half_size, half_size),  # x bounds
            (-half_size, half_size),  # y bounds
            (self.z_min, self.z_max),  # z bounds
        ]

    def get_bounds_2d(self) -> list[tuple[float, float]]:
        """Get 2D bounds (x, y only).

        Returns:
            List of (min, max) tuples for x and y
        """
        half_size = self.size / 2
        return [
            (-half_size, half_size),  # x bounds
            (-half_size, half_size),  # y bounds
        ]

    def add_obstacle(self, obstacle: Obstacle) -> None:
        """Add an obstacle to the map.

        Args:
            obstacle: Obstacle to add
        """
        self.obstacles.append(obstacle)

    def add_obstacles(self, obstacles: list[Obstacle]) -> None:
        """Add multiple obstacles to the map.

        Args:
            obstacles: List of obstacles to add
        """
        self.obstacles.extend(obstacles)

    def generate_obstacles(
        self,
        server: viser.ViserServer,
        num_obstacles: int,
        min_size: float = 0.5,
        max_size: float = 3.0,
        min_height: float | None = None,
        max_height: float | None = None,
        seed: int | None = None,
        color: tuple[int, int, int] = (200, 100, 50),
        check_overlap: bool = True,
        obstacle_type: Literal["box", "sphere", "mixed"] = "box",
    ) -> list[Obstacle]:
        """Generate random obstacles in the map.

        Args:
            server: Viser server instance for visualization
            num_obstacles: Number of obstacles to generate
            min_size: Minimum obstacle size (box: width/depth, sphere: radius)
            max_size: Maximum obstacle size (box: width/depth, sphere: radius)
            min_height: Minimum obstacle height (defaults to z_min)
            max_height: Maximum obstacle height (defaults to z_max)
            seed: Random seed
            color: Obstacle color
            check_overlap: Whether to prevent overlap
            obstacle_type: Type of obstacles ("box", "sphere", or "mixed")

        Returns:
            List of generated obstacles

        Example:
            >>> map_env = Map(size=20, z_range=(0.5, 2.5))
            >>> # Generate only box obstacles
            >>> obstacles = map_env.generate_random_obstacles(server, num_obstacles=10, obstacle_type="box")
            >>> # Generate only sphere obstacles
            >>> obstacles = map_env.generate_random_obstacles(server, num_obstacles=10, obstacle_type="sphere")
            >>> # Generate mixed obstacles (50/50 chance)
            >>> obstacles = map_env.generate_random_obstacles(server, num_obstacles=10, obstacle_type="mixed")
        """
        if min_height is None:
            min_height = self.z_min
        if max_height is None:
            max_height = self.z_max

        if seed is not None:
            np.random.seed(seed)

        obstacles: list[Obstacle] = []
        attempts = 0
        max_attempts = num_obstacles * 10  # Prevent infinite loop

        while len(obstacles) < num_obstacles and attempts < max_attempts:
            attempts += 1

            # Decide obstacle type for this iteration
            new_obstacle: Obstacle
            if obstacle_type == "mixed":
                current_type = np.random.choice(["box", "sphere"])
            else:
                current_type = obstacle_type

            if current_type == "box":
                new_obstacle = _generate_random_box_obstacle(
                    self.size, min_size, max_size, min_height, max_height
                )
            else:  # sphere
                new_obstacle = _generate_random_sphere_obstacle(
                    self.size, min_size, max_size, min_height, max_height
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

            # Visualize in Viser
            _visualize_obstacle(server, new_obstacle, len(obstacles), color)

        if len(obstacles) < num_obstacles:
            print(
                f"Warning: Only {len(obstacles)} out of {num_obstacles} obstacles were generated."
            )

        self.obstacles.extend(obstacles)
        return obstacles

    def visualize_bounds(
        self,
        server: viser.ViserServer,
        ground_color: tuple[int, int, int] = (150, 150, 150),
        boundary_color: tuple[int, int, int] = (100, 100, 100),
    ) -> None:
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
            position=(0, self.size / 2, boundary_height / 2),
            color=boundary_color,
        )

        # Bottom boundary
        server.scene.add_box(
            "/map/boundary_bottom",
            dimensions=(self.size, boundary_thickness, boundary_height),
            position=(0, -self.size / 2, boundary_height / 2),
            color=boundary_color,
        )

        # Left boundary
        server.scene.add_box(
            "/map/boundary_left",
            dimensions=(boundary_thickness, self.size, boundary_height),
            position=(-self.size / 2, 0, boundary_height / 2),
            color=boundary_color,
        )

        # Right boundary
        server.scene.add_box(
            "/map/boundary_right",
            dimensions=(boundary_thickness, self.size, boundary_height),
            position=(self.size / 2, 0, boundary_height / 2),
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
        if len(state) > 2:
            return bool(self.z_min <= state[2] <= self.z_max)

        return True

    def clear_obstacles(self) -> None:
        """Clear all obstacles from the map."""
        self.obstacles.clear()

    def __repr__(self) -> str:
        """String representation of the map."""
        return f"Map(size={self.size}, z_range=({self.z_min}, {self.z_max}), obstacles={len(self.obstacles)})"
