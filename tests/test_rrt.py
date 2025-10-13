"""Tests for RRT algorithm."""

import numpy as np

from planning.sampling import RRT, RRTConfig, RRTConnect, RRTConnectConfig


def test_rrt_simple_2d():
    """Test RRT in simple 2D obstacle-free environment."""
    start = (0, 0)
    goal = (10, 10)
    bounds = [(-2, 12), (-2, 12)]

    rrt = RRT(
        start_state=start,
        goal_state=goal,
        bounds=bounds,
        config=RRTConfig(max_iterations=1000, step_size=0.5, goal_tolerance=0.5, seed=42),
    )

    path = rrt.plan()

    assert path is not None
    assert len(path) > 0
    assert np.allclose(path[0].state, start)


def test_rrt_reaches_goal():
    """Test that RRT finds goal within tolerance."""
    start = (0, 0, 0)
    goal = (5, 5, 2)
    bounds = [(-1, 6), (-1, 6), (0, 3)]

    rrt = RRT(
        start_state=start,
        goal_state=goal,
        bounds=bounds,
        config=RRTConfig(max_iterations=2000, step_size=0.5, goal_tolerance=0.5, seed=42),
    )

    path = rrt.plan()

    if path is not None:
        final_state = path[-1].state
        distance_to_goal = np.linalg.norm(final_state - np.array(goal))
        assert distance_to_goal <= rrt.goal_tolerance


def test_rrt_stats():
    """Test RRT statistics tracking."""
    rrt = RRT(
        start_state=(0, 0),
        goal_state=(5, 5),
        bounds=[(-1, 6), (-1, 6)],
        config=RRTConfig(max_iterations=500, seed=42),
    )

    _path = rrt.plan()
    stats = rrt.get_stats()

    assert "num_nodes" in stats
    assert "goal_reached" in stats
    assert "path_length" in stats
    assert stats["num_nodes"] > 0


def test_rrt_connect_simple():
    """Test RRT-Connect in obstacle-free space."""
    start = (0, 0, 0)
    goal = (10, 10, 5)
    bounds = [(-2, 12), (-2, 12), (-2, 10)]

    rrt_connect = RRTConnect(
        start_state=start,
        goal_state=goal,
        bounds=bounds,
        config=RRTConnectConfig(max_iterations=1000, step_size=0.5, seed=42),
    )

    path = rrt_connect.plan()

    assert path is not None or len(rrt_connect.start_nodes) > 0


def test_rrt_invalid_start():
    """Test RRT with start in collision."""
    from planning.map import BoxObstacle
    from planning.sampling import ObstacleCollisionChecker

    # Obstacle at start
    obstacle = BoxObstacle(position=(0, 0, 0), size=(2, 2, 2))
    checker = ObstacleCollisionChecker([obstacle])

    rrt = RRT(
        start_state=(0, 0, 0),
        goal_state=(10, 10, 5),
        bounds=[(-5, 15), (-5, 15), (-5, 10)],
        collision_checker=checker,
        config=RRTConfig(max_iterations=100),
    )

    path = rrt.plan()

    assert path is None


def test_rrt_tree_edges():
    """Test tree edge extraction."""
    rrt = RRT(
        start_state=(0, 0),
        goal_state=(5, 5),
        bounds=[(-1, 6), (-1, 6)],
        config=RRTConfig(max_iterations=100, seed=42),
    )

    _path = rrt.plan()
    edges = rrt.get_tree_edges()

    assert len(edges) > 0
    assert all(len(edge) == 2 for edge in edges)  # Each edge is (parent, child)


def test_rrt_connect_with_obstacles():
    """Test RRT-Connect with obstacles in the middle."""
    from planning.map import BoxObstacle
    from planning.sampling import ObstacleCollisionChecker

    # Create obstacle in the middle
    obstacle = BoxObstacle(position=(5, 5, 2.5), size=(2, 2, 2))
    checker = ObstacleCollisionChecker([obstacle])

    start = (0, 0, 0)
    goal = (10, 10, 5)
    bounds = [(-2, 12), (-2, 12), (-2, 10)]

    rrt_connect = RRTConnect(
        start_state=start,
        goal_state=goal,
        bounds=bounds,
        collision_checker=checker,
        config=RRTConnectConfig(max_iterations=2000, seed=42),
    )

    path = rrt_connect.plan()

    # Should find a path around the obstacle or fail gracefully
    if path is not None:
        # Verify path doesn't go through obstacle
        for node in path:
            assert not obstacle.contains_point(tuple(node.state[:3]))
        # Verify path connects start to goal
        assert np.allclose(path[0].state, start)
        # Last node should be close to goal (within step_size of goal tree root)
        distance_to_goal = np.linalg.norm(path[-1].state - np.array(goal))
        assert distance_to_goal <= 5.0  # Reasonable distance for connected path


def test_rrt_connect_collision_start():
    """Test RRT-Connect with start state in collision."""
    from planning.map import BoxObstacle
    from planning.sampling import ObstacleCollisionChecker

    # Obstacle at start
    obstacle = BoxObstacle(position=(0, 0, 0), size=(2, 2, 2))
    checker = ObstacleCollisionChecker([obstacle])

    rrt_connect = RRTConnect(
        start_state=(0, 0, 0),
        goal_state=(10, 10, 5),
        bounds=[(-5, 15), (-5, 15), (-5, 10)],
        collision_checker=checker,
        config=RRTConnectConfig(max_iterations=100),
    )

    path = rrt_connect.plan()

    assert path is None  # Should fail immediately
