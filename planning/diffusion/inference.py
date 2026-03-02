"""Inference helpers for diffusion trajectory sampling."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class PolicyCallable(Protocol):
    def __call__(
        self,
        conditions: dict[int | str, np.ndarray],
        *,
        batch_size: int,
    ) -> object: ...


def extract_trajectory_observations(result: object) -> np.ndarray:
    """Extract trajectory observations from policy outputs.

    Args:
        result: Raw policy output object.

    Returns:
        A NumPy array of shape [batch, horizon, state_dim].
    """
    if hasattr(result, "observations"):
        return np.asarray(result.observations, dtype=float)
    if isinstance(result, tuple):
        for item in result:
            observations = extract_trajectory_observations(item)
            if observations.ndim == 3:
                return observations
        raise ValueError("Policy result tuple did not contain trajectory observations.")
    return np.asarray(result, dtype=float)


def sample_trajectory_batch(
    policy: PolicyCallable,
    condition: np.ndarray,
    sample_batch_size: int,
    condition_key: int | str = 0,
) -> np.ndarray:
    """Sample a batch of trajectories from a policy.

    Args:
        policy: Diffusion policy instance implementing a call interface.
        condition: Conditioning vector.
        sample_batch_size: Batch size.
        condition_key: Condition mapping key used by the sampler.

    Returns:
        Observations with shape [batch, horizon, state_dim].
    """
    result = policy(
        {condition_key: condition},
        batch_size=sample_batch_size,
    )
    observations = extract_trajectory_observations(result)
    if observations.ndim != 3:
        raise ValueError("Policy output must be [batch, horizon, state_dim].")
    return observations
