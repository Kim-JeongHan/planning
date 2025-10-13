"""Tests for collision detection."""

import numpy as np

from planning.map import Obstacle
from planning.sampling import CollisionChecker, EmptyCollisionChecker


def test_obstacle_contains_point():
    """Test point-in-obstacle detection."""
    obstacle = Obstacle(position=(5, 5, 5), size=(2, 2, 2))

    # Inside
    assert obstacle.contains_point((5, 5, 5)) is True
    assert obstacle.contains_point((4.5, 5, 5)) is True

    # Outside
    assert obstacle.contains_point((10, 10, 10)) is False
    assert obstacle.contains_point((3, 5, 5)) is False


def test_obstacle_intersection():
    """Test obstacle-obstacle intersection."""
    obs1 = Obstacle(position=(0, 0, 0), size=(2, 2, 2))
    obs2 = Obstacle(position=(1, 0, 0), size=(2, 2, 2))
    obs3 = Obstacle(position=(10, 0, 0), size=(2, 2, 2))

    assert obs1.intersects(obs2) is True
    assert obs1.intersects(obs3) is False


def test_collision_checker_point():
    """Test collision checking for a point."""
    obstacle = Obstacle(position=(5, 5, 5), size=(2, 2, 2))
    checker = CollisionChecker([obstacle])

    free_state = np.array([0, 0, 0])
    collision_state = np.array([5, 5, 5])

    assert checker.is_collision_free(free_state) is True
    assert checker.is_collision_free(collision_state) is False


def test_collision_checker_path():
    """Test collision checking for a path."""
    obstacle = Obstacle(position=(5, 5, 5), size=(2, 2, 2))
    checker = CollisionChecker([obstacle])

    # Path that goes through obstacle
    start = np.array([0, 5, 5])
    goal = np.array([10, 5, 5])

    assert checker.is_path_collision_free(start, goal) is False


def test_collision_checker_path_free():
    """Test collision-free path."""
    obstacle = Obstacle(position=(5, 5, 5), size=(2, 2, 2))
    checker = CollisionChecker([obstacle])

    # Path that avoids obstacle
    start = np.array([0, 0, 0])
    goal = np.array([1, 1, 1])

    assert checker.is_path_collision_free(start, goal) is True


def test_empty_collision_checker():
    """Test empty collision checker (no obstacles)."""
    checker = EmptyCollisionChecker()

    assert checker.is_collision_free(np.array([0, 0, 0])) is True
    assert checker.is_path_collision_free(np.array([0, 0, 0]), np.array([10, 10, 10])) is True
