"""Trajectory-level constraints and filtering utilities."""

from __future__ import annotations

import numpy as np

from planning.collision import CollisionChecker


def _within_tolerance(
    point: np.ndarray,
    target_state: np.ndarray | None,
    tolerance: float,
) -> bool:
    """Check whether a point is within Euclidean tolerance of a target state."""
    return target_state is None or np.linalg.norm(point - target_state[:3], ord=2) <= tolerance


def _is_collision_free_path(
    points: np.ndarray,
    collision_checker: CollisionChecker,
    segment_resolution: float,
) -> bool:
    """Return whether every consecutive edge in the path is collision free."""
    return all(
        collision_checker.is_path_collision_free(
            start,
            end,
            resolution=segment_resolution,
        )
        for start, end in zip(points[:-1], points[1:])
    )


def select_collision_free_trajectory(
    trajectories: np.ndarray,
    collision_checker: CollisionChecker,
    *,
    start_state: np.ndarray | None = None,
    goal_state: np.ndarray | None = None,
    endpoint_tolerance: float = 0.5,
    segment_resolution: float = 0.1,
) -> np.ndarray | None:
    """Return the first trajectory that satisfies start/goal and collision constraints."""
    if endpoint_tolerance < 0.0:
        raise ValueError("endpoint_tolerance must be non-negative.")
    if segment_resolution <= 0.0:
        raise ValueError("segment_resolution must be positive.")
    if trajectories.ndim != 3:
        raise ValueError("Trajectories must be a 3D array [batch, horizon, state_dim].")

    for trajectory in trajectories:
        if trajectory.ndim != 2 or trajectory.shape[0] < 2 or trajectory.shape[1] < 3:
            continue

        points = trajectory[:, :3]
        if not _within_tolerance(points[0], start_state, endpoint_tolerance):
            continue
        if not _within_tolerance(points[-1], goal_state, endpoint_tolerance):
            continue
        if _is_collision_free_path(points, collision_checker, segment_resolution):
            return np.asarray(points, dtype=float)

    return None
