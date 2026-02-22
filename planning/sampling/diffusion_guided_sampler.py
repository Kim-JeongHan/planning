"""Diffusion-guided sampler for planner integration."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np

from ..diffusion import utils as _diffuser_utils
from .sampler import Sampler


class DiffusionGuidedSampler(Sampler):
    """Diffusion-guided sampler that proposes states from a trajectory model."""

    def __init__(
        self,
        bounds: list[tuple[float, float]],
        *,
        loadbase: str = "logs/pretrained",
        dataset: str,
        config: str | None = None,
        diffusion_loadpath: str = "f:diffusion/defaults_H{horizon}_T{n_diffusion_steps}",
        value_loadpath: str = "f:values/defaults_H{horizon}_T{n_diffusion_steps}_d{discount}",
        diffusion_epoch: str | int = "latest",
        value_epoch: str | int = "latest",
        guide: str = "sampling.ValueGuide",
        policy: str = "sampling.GuidedPolicy",
        n_guide_steps: int = 2,
        scale: float = 0.1,
        t_stopgrad: int = 2,
        scale_grad_by_std: bool = True,
        preprocess_fns: list[Callable[[np.ndarray], np.ndarray]] | None = None,
        sample_fn: Callable[..., object] | None = None,
        condition_key: int | str = 0,
        condition: np.ndarray | list[float] | None = None,
        state_indices: tuple[int, ...] | None = None,
        state_projection: Callable[[np.ndarray], np.ndarray] | None = None,
        sample_batch_size: int = 4,
        max_projection_retries: int = 128,
        verbose: bool = False,
        seed: int | None = None,
        policy_kwargs: dict[str, object] | None = None,
        **_: object,
    ) -> None:
        """Initialize diffusion-guided sampler.

        Args:
            bounds: List of (min, max) bounds for planner states.
            loadbase: Base directory containing `dataset` checkpoints.
            dataset: Environment/dataset identifier.
            config: Optional diffuser templating config module/file path.
            diffusion_loadpath: Diffusion checkpoint path under `loadbase`.
            value_loadpath: Value function checkpoint path under `loadbase`.
            diffusion_epoch: Diffusion checkpoint epoch.
            value_epoch: Value checkpoint epoch.
            guide: Guide class path for diffusion policy.
            policy: Policy class path for trajectory sampling.
            n_guide_steps: Number of guide steps per reverse step.
            scale: Guidance scale.
            t_stopgrad: Diffusion stopping index for gradients.
            scale_grad_by_std: Whether to scale gradients by std.
            preprocess_fns: Preprocess callbacks for observations.
            sample_fn: Optional sampling function override.
            condition_key: Condition mapping key.
            condition: Optional initial condition vector passed to the policy.
            state_indices: Optional indices to extract planner state from raw state vector.
            state_projection: Optional projection function for planner state.
            sample_batch_size: Number of trajectories sampled per policy call.
            max_projection_retries: Retry count before hard failure.
            verbose: Verbose output.
            seed: RNG seed.
            policy_kwargs: Optional extra kwargs for policy constructor.
            **_: Reserved for compatibility.
        """
        super().__init__(bounds, seed=seed)

        if self.dim != 3:
            raise ValueError("DiffusionGuidedSampler currently supports 3D state spaces only.")
        if sample_batch_size <= 0:
            raise ValueError("sample_batch_size must be a positive integer.")
        if max_projection_retries <= 0:
            raise ValueError("max_projection_retries must be a positive integer.")

        self.loadbase = loadbase
        self.dataset = dataset
        self.config = config
        self.diffusion_loadpath = diffusion_loadpath
        self.value_loadpath = value_loadpath
        self.diffusion_epoch = diffusion_epoch
        self.value_epoch = value_epoch
        self.guide = guide
        self.policy = policy
        self.n_guide_steps = n_guide_steps
        self.scale = scale
        self.t_stopgrad = t_stopgrad
        self.scale_grad_by_std = scale_grad_by_std
        self.preprocess_fns = preprocess_fns or []
        self.sample_fn = sample_fn
        self.condition_key = condition_key
        self.condition = np.asarray(condition, dtype=float) if condition is not None else None
        self.state_indices = state_indices
        self.state_projection = state_projection
        self.sample_batch_size = sample_batch_size
        self.max_projection_retries = max_projection_retries
        self.verbose = verbose
        self.seed = seed
        self.policy_kwargs = policy_kwargs or {}

        self._policy = None
        self._cached_states: list[np.ndarray] = []

    def sample(self) -> np.ndarray:
        """Sample a valid planner state from diffusion trajectories."""
        for _ in range(self.max_projection_retries):
            if not self._cached_states:
                self._cached_states.extend(self._sample_from_diffusion())

            if not self._cached_states:
                continue

            candidate = self._cached_states.pop()
            if self._in_bounds(candidate):
                return candidate

        raise ValueError(
            "No valid 3D sample could be produced from diffusion proposals "
            f"after {self.max_projection_retries} attempts."
        )

    def _sample_from_diffusion(self) -> list[np.ndarray]:
        policy = self._get_policy()
        conditions = self._format_conditions()
        result = policy(
            conditions,
            batch_size=self.sample_batch_size,
            verbose=self.verbose,
        )
        observations = self._extract_policy_observations(result)
        if observations.ndim != 3:
            raise ValueError(
                "Diffuser policy must return observations with shape [batch, horizon, state_dim]."
            )

        flattened = observations.reshape((-1, observations.shape[-1]))
        candidates = [self._project_state(state) for state in flattened]
        return candidates

    def _extract_policy_observations(self, result: object) -> np.ndarray:
        if not isinstance(result, tuple):
            return self._extract_observations(result)

        for sample in result:
            observations = self._extract_observations(sample)
            if observations.ndim == 3:
                return observations

        # Fallback to the first element for tuple-like wrappers with metadata.
        return self._extract_observations(result[0])

    def _extract_observations(self, samples: object) -> np.ndarray:
        if hasattr(samples, "observations"):
            return np.asarray(samples.observations, dtype=float)
        return np.asarray(samples, dtype=float)

    def _project_state(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=float)
        if state_projection := self.state_projection:
            projected = np.asarray(state_projection(state), dtype=float)
        elif self.state_indices is not None:
            projected = np.asarray(state[list(self.state_indices)], dtype=float)
        else:
            projected = state[:3]

        if projected.shape != (3,):
            raise ValueError("Projected state must be a 3D vector.")
        return projected

    def _in_bounds(self, state: np.ndarray) -> bool:
        if not np.isfinite(state).all():
            return False
        return bool(np.all(state >= self.bounds[:, 0]) and np.all(state <= self.bounds[:, 1]))

    def _get_policy(self) -> object:
        if self._policy is None:
            self._policy = self._build_policy()
        return self._policy

    def _build_policy(self) -> object:
        diffusion_experiment = self._load_diffusion_checkpoint(
            self.diffusion_loadpath, self.diffusion_epoch
        )
        value_experiment = self._load_diffusion_checkpoint(self.value_loadpath, self.value_epoch)
        _diffuser_utils.check_compatibility(diffusion_experiment, value_experiment)

        guide_config = _diffuser_utils.Config(
            self.guide,
            model=value_experiment.ema,
            verbose=self.verbose,
        )
        guide = guide_config()

        policy_config = _diffuser_utils.Config(
            self.policy,
            guide=guide,
            scale=self.scale,
            diffusion_model=diffusion_experiment.ema,
            normalizer=diffusion_experiment.dataset.normalizer,
            preprocess_fns=self.preprocess_fns,
            sample_fn=self.sample_fn,
            n_guide_steps=self.n_guide_steps,
            t_stopgrad=self.t_stopgrad,
            scale_grad_by_std=self.scale_grad_by_std,
            verbose=self.verbose,
            **self.policy_kwargs,
        )
        return policy_config()

    def _load_diffusion_checkpoint(self, loadpath: str, epoch: str | int) -> object:
        catalog = _diffuser_utils.CheckpointCatalog(
            self.loadbase,
            self.dataset,
            loadpath,
            config=self.config,
        )
        return _diffuser_utils.DiffusionArtifactLoader(
            catalog,
            seed=self.seed,
        ).load(epoch)

    def _format_conditions(self) -> Mapping[int | str, np.ndarray]:
        if self.condition is None:
            return {}
        return {self.condition_key: self.condition}
