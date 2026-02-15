"""Tests for one-shot trajectory generation helpers."""

from pathlib import Path

import numpy as np
import pytest

from examples import diffusion_trajectory_one_shot_example as helper
from planning.collision import CollisionChecker


class _BlockedChecker(CollisionChecker):
    """Collision checker that blocks x coordinates above a threshold."""

    def __init__(self, blocked_x: float) -> None:
        self.blocked_x = blocked_x

    def is_collision_free(self, state: np.ndarray) -> bool:
        return state[0] < self.blocked_x

    def is_path_collision_free(
        self,
        from_state: np.ndarray,
        to_state: np.ndarray,
        resolution: float = 0.1,
    ) -> bool:
        steps = max(2, int(np.ceil(np.linalg.norm(to_state - from_state) / resolution)) + 1)
        direction = to_state - from_state
        for ratio in np.linspace(0.0, 1.0, steps):
            point = from_state + ratio * direction
            if point[0] >= self.blocked_x:
                return False
        return True


def test_extract_observations_from_mapping_like() -> None:
    """Helper should extract trajectory array from object attributes or raw arrays."""

    class _Wrapped:
        def __init__(self, observations: np.ndarray) -> None:
            self.observations = observations

    wrapped = _Wrapped(
        np.array(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            ],
            dtype=float,
        )
    )
    assert np.array_equal(
        helper._extract_observations(wrapped),
        np.array(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            ],
            dtype=float,
        ),
    )

    raw = helper._extract_observations((None, wrapped.observations))
    assert raw.shape == (1, 2, 3)


def test_ensure_vector_and_vector2_validation() -> None:
    """Scalar loaders and vector validators should coerce and validate inputs."""
    assert np.array_equal(
        helper._ensure_vector((1, 2, 3), 3, "vector"),
        np.array([1.0, 2.0, 3.0], dtype=float),
    )
    assert helper._ensure_vector2((7, 8), "point") == (7.0, 8.0)

    with pytest.raises(ValueError, match="must have shape"):
        helper._ensure_vector((1, 2), 3, "bad-vector")
    with pytest.raises(ValueError, match="must have shape"):
        helper._ensure_vector2((1, 2, 3), "bad-point")


def test_load_run_config_reads_yaml(tmp_path: Path) -> None:
    """Load config from YAML and validate mapping conversion."""
    pytest.importorskip("yaml")
    config_path = tmp_path / "one_shot.yaml"
    config_path.write_text("seed: 7\nstart_state: [1, 2, 3]\n", encoding="utf-8")

    cfg = helper._load_run_config(str(config_path))

    assert cfg["seed"] == 7
    assert cfg["start_state"] == [1, 2, 3]


def test_select_collision_free_trajectory_prefers_first_valid() -> None:
    """Selection should return the first trajectory that passes checks."""
    trajectories = np.array(
        [
            [[-1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0]],
            [[-1.0, 0.0, 1.0], [-1.1, 0.0, 1.0], [-1.2, 0.0, 1.0]],
            [[-0.5, 0.0, 1.0], [-0.6, 0.0, 1.0], [-0.7, 0.0, 1.0]],
        ],
        dtype=float,
    )
    checker = _BlockedChecker(blocked_x=0.5)
    bounds = np.array([[-3.0, 3.0], [-3.0, 3.0], [0.0, 3.0]])
    start = np.array([-1.0, 0.0, 1.0], dtype=float)
    goal = np.array([-1.2, 0.0, 1.0], dtype=float)

    result = helper._select_collision_free_trajectory(
        trajectories=trajectories,
        collision_checker=checker,
        bounds=bounds,
        start_state=start,
        goal_state=goal,
        endpoint_tolerance=0.6,
    )

    assert result is not None
    assert np.array_equal(
        result,
        np.array([[-1.0, 0.0, 1.0], [-1.1, 0.0, 1.0], [-1.2, 0.0, 1.0]], dtype=float),
    )


def test_select_collision_free_trajectory_returns_none_if_all_invalid() -> None:
    """No valid trajectory should return None."""
    trajectories = np.array(
        [
            [[-1.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
            [[-2.0, 0.0, 1.0], [2.0, 0.0, 1.0]],
        ],
        dtype=float,
    )
    checker = _BlockedChecker(blocked_x=0.0)
    bounds = np.array([[-3.0, 3.0], [-3.0, 3.0], [0.0, 3.0]])

    result = helper._select_collision_free_trajectory(
        trajectories=trajectories,
        collision_checker=checker,
        bounds=bounds,
    )

    assert result is None
