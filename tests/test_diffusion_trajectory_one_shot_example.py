"""Tests for one-shot trajectory generation helpers."""

from pathlib import Path

import numpy as np
import pytest

from examples import diffusion_trajectory_one_shot_example as helper
from planning.collision import CollisionChecker
from planning.constraint import select_collision_free_trajectory
from planning.diffusion.inference import extract_trajectory_observations


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


def test_extract_trajectory_observations_from_mapping_like() -> None:
    """extract_trajectory_observations should handle mapping-like or raw array outputs."""

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
        extract_trajectory_observations(wrapped),
        np.array(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            ],
            dtype=float,
        ),
    )

    raw = extract_trajectory_observations((None, wrapped.observations))
    assert raw.shape == (1, 2, 3)


def test_load_run_config_reads_yaml(tmp_path: Path) -> None:
    """Load config from YAML and validate with DiffusionOneShotConfig."""
    pytest.importorskip("yaml")
    config_path = tmp_path / "one_shot.yaml"
    config_path.write_text(
        "diffusion:\n"
        "  dataset: test_dataset\n"
        "  loadbase: logs/pretrained\n"
        "environment:\n"
        "  start_state: [1.0, 2.0, 1.0]\n"
        "  goal_state: [2.0, 3.0, 1.0]\n"
        "rollout: {}\n",
        encoding="utf-8",
    )

    cfg = helper.DiffusionOneShotConfig.load(str(config_path))

    assert cfg.environment.start_state == (1.0, 2.0, 1.0)
    assert cfg.environment.goal_state == (2.0, 3.0, 1.0)
    assert cfg.diffusion.dataset == "test_dataset"


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
    start = np.array([-1.0, 0.0, 1.0], dtype=float)
    goal = np.array([-1.2, 0.0, 1.0], dtype=float)

    result = select_collision_free_trajectory(
        trajectories=trajectories,
        collision_checker=checker,
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

    result = select_collision_free_trajectory(
        trajectories=trajectories,
        collision_checker=checker,
    )

    assert result is None
