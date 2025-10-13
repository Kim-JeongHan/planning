"""Tests for Map class."""

import numpy as np
import pytest
from planning.map import Map


def test_map_creation():
    """Test creating a map."""
    map_env = Map(size=10, z_range=(0, 5))
    
    assert map_env.size == 10
    assert map_env.z_min == 0
    assert map_env.z_max == 5
    assert len(map_env.obstacles) == 0


def test_get_bounds():
    """Test getting map bounds."""
    map_env = Map(size=20, z_range=(1, 3))
    
    bounds = map_env.get_bounds()
    
    assert bounds == [(-10, 10), (-10, 10), (1, 3)]


def test_get_bounds_2d():
    """Test getting 2D bounds."""
    map_env = Map(size=10)
    
    bounds_2d = map_env.get_bounds_2d()
    
    assert bounds_2d == [(-5, 5), (-5, 5)]


def test_is_valid_state_within_bounds():
    """Test state validation within bounds."""
    map_env = Map(size=10, z_range=(0, 5))
    
    valid_state = np.array([0, 0, 2])
    
    assert map_env.is_valid_state(valid_state) is True


def test_is_valid_state_outside_bounds():
    """Test state validation outside bounds."""
    map_env = Map(size=10, z_range=(0, 5))
    
    # Outside x bounds
    assert map_env.is_valid_state(np.array([20, 0, 2])) is False
    
    # Outside z bounds
    assert map_env.is_valid_state(np.array([0, 0, 10])) is False


def test_add_obstacle():
    """Test adding obstacles to map."""
    from planning.map import Obstacle
    
    map_env = Map(size=10)
    obstacle = Obstacle(position=(1, 1, 1), size=(1, 1, 1))
    
    map_env.add_obstacle(obstacle)
    
    assert len(map_env.obstacles) == 1
    assert map_env.obstacles[0] == obstacle


def test_clear_obstacles():
    """Test clearing obstacles."""
    from planning.map import Obstacle
    
    map_env = Map(size=10)
    map_env.add_obstacle(Obstacle((0, 0, 0), (1, 1, 1)))
    map_env.add_obstacle(Obstacle((2, 2, 2), (1, 1, 1)))
    
    map_env.clear_obstacles()
    
    assert len(map_env.obstacles) == 0


def test_map_repr():
    """Test map string representation."""
    map_env = Map(size=20, z_range=(0.5, 2.5))
    
    repr_str = repr(map_env)
    
    assert "size=20" in repr_str
    assert "0.5" in repr_str
    assert "obstacles=0" in repr_str

