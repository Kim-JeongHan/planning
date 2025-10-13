"""Map-related module."""

from .map import Map
from .obstacles import generate_random_obstacles, Obstacle

__all__ = ["Map", "generate_random_obstacles", "Obstacle"]
