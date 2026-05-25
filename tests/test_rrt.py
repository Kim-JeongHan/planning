"""Tests for RRT algorithm."""

from typing import cast

import numpy as np
import pytest

from planning.collision import CollisionChecker
from planning.graph import Node
from planning.sampling import (
    RRT,
    GoalBiasedSampler,
    InformedRRTStar,
    RRTConfig,
    RRTConnect,
    RRTConnectConfig,
    RRTStar,
    RRTStarConfig,
)
from planning.sampling.sampler import InformedSampler, Sampler


class DummySampler(Sampler):
    """Sampler used to verify kwargs forwarding."""

    def __init__(
        self,
        bounds: list[tuple[float, float]],
        *,
        marker: str = "default",
        **kwargs: object,
    ) -> None:
        super().__init__(bounds)
        self.marker = marker
        self.extra_kwargs = kwargs

    def sample(self) -> np.ndarray:
        """Return deterministic sample."""
        return np.zeros(self.dim)


class SequenceSampler(Sampler):
    """Sampler that returns a fixed sequence, then repeats the last sample."""

    def __init__(
        self,
        bounds: list[tuple[float, float]],
        *,
        samples: list[tuple[float, ...]],
        **kwargs: object,
    ) -> None:
        super().__init__(bounds)
        self.samples = [np.asarray(sample, dtype=float) for sample in samples]
        self.index = 0

    def sample(self) -> np.ndarray:
        """Return the next configured sample."""
        if self.index >= len(self.samples):
            return self.samples[-1].copy()
        sample = self.samples[self.index]
        self.index += 1
        return sample.copy()


class RejectAllPaths(CollisionChecker):
    """Collision checker that allows states but rejects all edges."""

    def is_collision_free(self, state: np.ndarray) -> bool:
        return True

    def is_path_collision_free(
        self, from_state: np.ndarray, to_state: np.ndarray, resolution: float = 0.1
    ) -> bool:
        return False


@pytest.mark.parametrize("config_cls", [RRTConfig, RRTConnectConfig, RRTStarConfig])
def test_rrt_configs_reject_non_positive_step_size(config_cls: type) -> None:
    """Invalid step sizes should fail before planning can loop indefinitely."""
    with pytest.raises(ValueError):
        config_cls(step_size=0.0)

    with pytest.raises(ValueError):
        config_cls(step_size=-0.1)


def test_rrt_simple_2d():
    """Test RRT in simple 2D obstacle-free environment."""
    start = (0, 0)
    goal = (10, 10)
    bounds = [(-2.0, 12.0), (-2.0, 12.0)]

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
    bounds = [(-1.0, 6.0), (-1.0, 6.0), (0.0, 3.0)]

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
        bounds=[(-1.0, 6.0), (-1.0, 6.0)],
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
    bounds = [(-2.0, 12.0), (-2.0, 12.0), (-2.0, 10.0)]

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
        bounds=[(-5.0, 15.0), (-5.0, 15.0), (-5.0, 10.0)],
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
        bounds=[(-1.0, 6.0), (-1.0, 6.0)],
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
    bounds = [(-2.0, 12.0), (-2.0, 12.0), (-2.0, 10.0)]

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
        bounds=[(-5.0, 15.0), (-5.0, 15.0), (-5.0, 10.0)],
        collision_checker=checker,
        config=RRTConnectConfig(max_iterations=100),
    )

    path = rrt_connect.plan()

    assert path is None  # Should fail immediately


def test_rrt_sampler_kwargs_are_passed_to_custom_sampler() -> None:
    """Custom sampler should receive provided sampler_kwargs."""
    rrt = RRT(
        start_state=(0, 0, 0),
        goal_state=(4, 4, 1),
        bounds=[(-1.0, 6.0), (-1.0, 6.0), (0.0, 3.0)],
        config=RRTConfig(
            sampler=DummySampler,
            sampler_kwargs={"marker": "custom", "alpha": 0.75},
        ),
    )

    assert isinstance(rrt.sampler, DummySampler)
    assert rrt.sampler.marker == "custom"
    assert rrt.sampler.extra_kwargs["alpha"] == 0.75


def test_rrt_goal_biased_sampler_uses_config_values() -> None:
    """Goal bias from config should override any conflicting sampler_kwargs."""
    rrt = RRT(
        start_state=(0, 0, 0),
        goal_state=(4, 4, 1),
        bounds=[(-1.0, 6.0), (-1.0, 6.0), (0.0, 3.0)],
        config=RRTConfig(
            sampler=GoalBiasedSampler,
            goal_bias=0.2,
            seed=5,
            sampler_kwargs={
                "goal_bias": 0.9,
                "goal_state": np.array([999.0, 999.0, 999.0]),
            },
        ),
    )

    sampler = cast(GoalBiasedSampler, rrt.sampler)
    assert sampler.goal_bias == 0.2
    assert np.allclose(sampler.goal_state, np.array([4.0, 4.0, 1.0]))


def test_rrt_star_sampler_kwargs_are_passed_to_custom_sampler() -> None:
    """RRT* should also forward sampler_kwargs to custom sampler."""
    rrt_star = RRTStar(
        start_state=(0, 0, 0),
        goal_state=(3, 3, 1),
        bounds=[(-1.0, 6.0), (-1.0, 6.0), (0.0, 3.0)],
        config=RRTStarConfig(
            sampler=DummySampler,
            sampler_kwargs={"marker": "rrtstar", "beta": 2},
        ),
    )

    assert isinstance(rrt_star.sampler, DummySampler)
    assert rrt_star.sampler.marker == "rrtstar"
    assert rrt_star.sampler.extra_kwargs["beta"] == 2


def test_rrt_star_goal_biased_sampler_uses_config_values() -> None:
    """Goal bias from config should override any conflicting sampler_kwargs."""
    rrt_star = RRTStar(
        start_state=(0, 0, 0),
        goal_state=(4, 4, 1),
        bounds=[(-1.0, 6.0), (-1.0, 6.0), (0.0, 3.0)],
        config=RRTStarConfig(
            sampler=GoalBiasedSampler,
            goal_bias=0.15,
            sampler_kwargs={
                "goal_bias": 0.7,
                "goal_state": np.array([999.0, 999.0, 999.0]),
            },
        ),
    )

    sampler = cast(GoalBiasedSampler, rrt_star.sampler)
    assert sampler.goal_bias == 0.15
    assert np.allclose(sampler.goal_state, np.array([4.0, 4.0, 1.0]))


def test_rrt_rejected_extension_does_not_attach_failed_child() -> None:
    """Rejected candidate edges should not mutate the accepted tree structure."""
    rrt = RRT(
        start_state=(0.0, 0.0),
        goal_state=(1.0, 0.0),
        bounds=[(-1.0, 2.0), (-1.0, 1.0)],
        collision_checker=RejectAllPaths(),
        config=RRTConfig(max_iterations=1, seed=1),
    )

    assert rrt.plan() is None
    assert rrt.root is not None
    assert rrt.nodes == [rrt.root]
    assert rrt.root.children == []


def test_rrt_close_start_connects_exact_goal() -> None:
    """A close start should still return an exact-goal waypoint when reachable."""
    rrt = RRT(
        start_state=(0.0, 0.0),
        goal_state=(0.1, 0.0),
        bounds=[(-1.0, 1.0), (-1.0, 1.0)],
        config=RRTConfig(max_iterations=0, goal_tolerance=0.2),
    )

    path = rrt.plan()

    assert path is not None
    assert len(path) == 2
    assert np.allclose(path[0].state, np.array([0.0, 0.0]))
    assert np.allclose(path[-1].state, np.array([0.1, 0.0]))
    assert rrt.goal_node is path[-1]
    assert rrt.goal_node.cost == pytest.approx(0.1)


def test_rrt_close_start_requires_collision_free_goal_connection() -> None:
    """A close start should not succeed if the exact-goal segment is blocked."""
    rrt = RRT(
        start_state=(0.0, 0.0),
        goal_state=(0.1, 0.0),
        bounds=[(-1.0, 1.0), (-1.0, 1.0)],
        collision_checker=RejectAllPaths(),
        config=RRTConfig(max_iterations=0, goal_tolerance=0.2),
    )

    assert rrt.plan() is None
    assert rrt.root is not None
    assert rrt.nodes == [rrt.root]


def test_rrt_connect_connection_node_records_cumulative_cost() -> None:
    """Final connection nodes should carry the cost from their own tree root."""
    rrt_connect = RRTConnect(
        start_state=(0.0, 0.0),
        goal_state=(1.0, 0.0),
        bounds=[(-1.0, 2.0), (-1.0, 1.0)],
        config=RRTConnectConfig(step_size=0.75),
    )
    goal_root = Node((1.0, 0.0), cost=0.0)
    target = Node((0.5, 0.0), cost=0.5)

    connection = rrt_connect._connect_tree([goal_root], target)

    assert connection is not None
    assert connection.parent is not None
    expected_cost = connection.parent.cost + connection.parent.distance_to(connection)
    assert connection.cost == pytest.approx(expected_cost)


def test_informed_sampler_handles_best_cost_below_straight_line_distance() -> None:
    """Invalid best costs should not create NaNs or hang the informed sampler."""
    sampler = InformedSampler(
        bounds=[(-1.0, 2.0), (-1.0, 1.0)],
        start_state=np.array([0.0, 0.0]),
        goal_state=np.array([1.0, 0.0]),
    )

    sample = sampler.sample(0.9)

    assert sample.shape == (2,)
    assert np.all(np.isfinite(sample))


def test_informed_rrt_star_tracks_exact_goal_cost_for_informed_sampling() -> None:
    """Near-goal candidates should include the final segment to the exact goal."""
    rrt_star = InformedRRTStar(
        start_state=(0.0, 0.0),
        goal_state=(1.0, 0.0),
        bounds=[(-1.0, 2.0), (-1.0, 1.0)],
        config=RRTStarConfig(
            sampler=SequenceSampler,
            sampler_kwargs={"samples": [(0.9, 0.0)]},
            max_iterations=2,
            step_size=1.0,
            goal_tolerance=0.2,
        ),
    )

    path = rrt_star.plan()

    assert path is not None
    assert np.allclose(path[-1].state, np.array([1.0, 0.0]))
    assert rrt_star.goal_node is not None
    assert rrt_star.goal_node.cost == pytest.approx(1.0)
    assert min(node.cost for node in rrt_star.goal_nodes) >= rrt_star.informed_sampler.c_min
