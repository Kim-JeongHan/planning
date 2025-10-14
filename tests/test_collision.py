"""Tests for collision detection."""

import numpy as np

from planning.collision import EmptyCollisionChecker, ObstacleCollisionChecker
from planning.map import BoxObstacle, SphereObstacle


def test_obstacle_contains_point():
    """Test point-in-obstacle detection."""
    obstacle = BoxObstacle(position=(5, 5, 5), size=(2, 2, 2))

    # Inside
    assert obstacle.contains_point((5, 5, 5)) is True
    assert obstacle.contains_point((4.5, 5, 5)) is True

    # Outside
    assert obstacle.contains_point((10, 10, 10)) is False
    assert obstacle.contains_point((3, 5, 5)) is False


def test_obstacle_intersection():
    """Test obstacle-obstacle intersection."""
    obs1 = BoxObstacle(position=(0, 0, 0), size=(2, 2, 2))
    obs2 = BoxObstacle(position=(1, 0, 0), size=(2, 2, 2))
    obs3 = BoxObstacle(position=(10, 0, 0), size=(2, 2, 2))

    assert obs1.intersects(obs2) is True
    assert obs1.intersects(obs3) is False


def test_sphere_obstacle_contains_point():
    """Test point-in-sphere detection."""
    obstacle = SphereObstacle(center=(5, 5, 5), radius=2.0)

    # Inside
    assert obstacle.contains_point((5, 5, 5)) is True
    assert obstacle.contains_point((5.5, 5, 5)) is True
    assert obstacle.contains_point((6, 5, 5)) is True

    # Outside
    assert obstacle.contains_point((10, 10, 10)) is False
    assert obstacle.contains_point((8, 5, 5)) is False


def test_sphere_sphere_intersection():
    """Test sphere-to-sphere intersection."""
    obs1 = SphereObstacle(center=(0, 0, 0), radius=2.0)
    obs2 = SphereObstacle(center=(3, 0, 0), radius=2.0)  # Should intersect
    obs3 = SphereObstacle(center=(10, 0, 0), radius=2.0)  # Should not intersect

    assert obs1.intersects(obs2) is True
    assert obs1.intersects(obs3) is False


def test_sphere_box_intersection():
    """Test sphere-to-box intersection."""
    sphere = SphereObstacle(center=(0, 0, 0), radius=2.0)
    box1 = BoxObstacle(position=(2, 0, 0), size=(1, 1, 1))  # Should intersect
    box2 = BoxObstacle(position=(10, 0, 0), size=(1, 1, 1))  # Should not intersect

    assert sphere.intersects(box1) is True
    assert sphere.intersects(box2) is False


def test_collision_checker_point():
    """Test collision checking for a point."""
    obstacle = BoxObstacle(position=(5, 5, 5), size=(2, 2, 2))
    checker = ObstacleCollisionChecker([obstacle])

    free_state = np.array([0, 0, 0])
    collision_state = np.array([5, 5, 5])

    assert checker.is_collision_free(free_state) is True
    assert checker.is_collision_free(collision_state) is False


def test_collision_checker_sphere():
    """Test collision checking with sphere obstacle."""
    obstacle = SphereObstacle(center=(5, 5, 5), radius=2.0)
    checker = ObstacleCollisionChecker([obstacle])

    free_state = np.array([0, 0, 0])
    collision_state = np.array([5, 5, 5])
    edge_state = np.array([7.5, 5, 5])  # Just outside (distance = 2.5 > radius 2.0)

    assert checker.is_collision_free(free_state) is True
    assert checker.is_collision_free(collision_state) is False
    assert checker.is_collision_free(edge_state) is True


def test_collision_checker_path():
    """Test collision checking for a path."""
    obstacle = BoxObstacle(position=(5, 5, 5), size=(2, 2, 2))
    checker = ObstacleCollisionChecker([obstacle])

    # Path that goes through obstacle
    start = np.array([0, 5, 5])
    goal = np.array([10, 5, 5])

    assert checker.is_path_collision_free(start, goal) is False


def test_collision_checker_path_free():
    """Test collision-free path."""
    obstacle = BoxObstacle(position=(5, 5, 5), size=(2, 2, 2))
    checker = ObstacleCollisionChecker([obstacle])

    # Path that avoids obstacle
    start = np.array([0, 0, 0])
    goal = np.array([1, 1, 1])

    assert checker.is_path_collision_free(start, goal) is True


def test_empty_collision_checker():
    """Test empty collision checker (no obstacles)."""
    checker = EmptyCollisionChecker()

    assert checker.is_collision_free(np.array([0, 0, 0])) is True
    assert checker.is_path_collision_free(np.array([0, 0, 0]), np.array([10, 10, 10])) is True
