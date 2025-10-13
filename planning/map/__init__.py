"""Map-related module."""

from .map import Map
from .obstacles import Obstacle, generate_random_obstacles

__all__ = ["Map", "Obstacle", "generate_random_obstacles"]
