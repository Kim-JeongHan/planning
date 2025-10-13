"""Sampling strategies for RRT algorithms."""

import numpy as np


class Sampler:
    """Base class for state space samplers."""

    def __init__(self, bounds: list[tuple[float, float]], seed: int | None = None) -> None:
        """Initialize the sampler.

        Args:
            bounds: List of (min, max) tuples for each dimension
            seed: Random seed for reproducibility
        """
        self.bounds = np.array(bounds)
        self.dim = len(bounds)

        if seed is not None:
            np.random.seed(seed)

    def sample(self) -> np.ndarray:
        """Sample a random state from the state space.

        Returns:
            A random state vector
        """
        raise NotImplementedError


class UniformSampler(Sampler):
    """Uniform random sampler."""

    def sample(self) -> np.ndarray:
        """Sample uniformly from the state space.

        Returns:
            A uniformly sampled state vector
        """
        result: np.ndarray = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
        return result


class GoalBiasedSampler(Sampler):
    """Goal-biased sampler that samples the goal with some probability."""

    def __init__(
        self,
        bounds: list[tuple[float, float]],
        goal_state: np.ndarray,
        goal_bias: float = 0.05,
        seed: int | None = None,
    ) -> None:
        """Initialize the goal-biased sampler.

        Args:
            bounds: List of (min, max) tuples for each dimension
            goal_state: The goal state to bias towards
            goal_bias: Probability of sampling the goal (0.0 to 1.0)
            seed: Random seed for reproducibility
        """
        super().__init__(bounds, seed)
        self.goal_state = np.array(goal_state)
        self.goal_bias = goal_bias

        if len(self.goal_state) != self.dim:
            raise ValueError(
                f"Goal state dimension {len(self.goal_state)} does not match bounds dimension {self.dim}"
            )

    def sample(self) -> np.ndarray:
        """Sample with goal bias.

        Returns:
            Either the goal state or a uniformly sampled state
        """
        if np.random.random() < self.goal_bias:
            return self.goal_state.copy()
        return np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])  # type: ignore[no-any-return]
