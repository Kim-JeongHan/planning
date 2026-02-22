"""Sampling helpers for the local diffuser compatibility layer.

Implements the guided diffusion planning algorithm from Janner et al. (2022)
"Planning with Diffusion for Flexible Behavior Synthesis":

  * Algorithm 1 (Guided Diffusion Planning): reverse diffusion with a value-
    function guide applied to the predicted mean μ, before adding noise.
  * Inpainting-based conditioning: at each denoising step, constrained
    timestep values (e.g. start/goal states) are written back into the
    trajectory to enforce boundary conditions without any special network head.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import SimpleNamespace

import numpy as np
import torch

from .core import PlannerStateNormalizer
from .training.noise import DiffusionSchedule

# ---------------------------------------------------------------------------
# Condition helpers
# ---------------------------------------------------------------------------

# Inpainting conditions map ``{timestep_index: state_array}``.
InpaintingConditions = Mapping[int, np.ndarray]


class ConditionAdapter:
    """Convert various condition formats to inpainting dicts or flat vectors."""

    @staticmethod
    def to_inpainting(
        conditions: dict[object, object] | None,
        state_dim: int,
    ) -> dict[int, np.ndarray]:
        """Normalise a raw conditions dict to ``{int_timestep: ndarray}`` format.

        Accepts either:
          - Already-inpainting format: ``{0: array, H-1: array}``
          - Legacy flat-vector format: ``{0: flat_vector}`` (vector interpreted
            as the start state; goal is not constrained)
        """
        if not conditions:
            return {}
        result: dict[int, np.ndarray] = {}
        for key, value in conditions.items():
            if not isinstance(key, (int, np.integer)):
                continue
            arr = np.asarray(value, dtype=float).ravel()
            result[int(key)] = arr[:state_dim]
        return result

    @staticmethod
    def to_vector(conditions: dict[object, object] | None, state_dim: int) -> np.ndarray:
        """Extract a single goal vector from a conditions dict (legacy helper)."""
        if not conditions:
            return np.zeros((state_dim,), dtype=float)
        for key in ("goal", "target", "end", "final", "pose"):
            candidate = conditions.get(key)
            if isinstance(candidate, (list, tuple, np.ndarray)):
                arr = np.asarray(candidate, dtype=float)
                return arr[:state_dim]
        last = list(conditions.values())[-1]
        if isinstance(last, (list, tuple, np.ndarray)):
            arr = np.asarray(last, dtype=float)
            return arr[:state_dim]
        return np.zeros((state_dim,), dtype=float)

    @staticmethod
    def to_matrix(
        condition: object,
        batch_size: int | None,
        fallback_dim: int | None = None,
    ) -> np.ndarray | None:
        if condition is None:
            return None
        if isinstance(condition, dict):
            if fallback_dim is None:
                raise ValueError("fallback_dim is required when condition is a mapping")
            condition = ConditionAdapter.to_vector(condition, fallback_dim)
        condition_array = np.asarray(condition, dtype=float)
        if condition_array.ndim == 1:
            condition_array = condition_array.reshape(1, -1)
        elif condition_array.ndim != 2:
            condition_array = condition_array.reshape(condition_array.shape[0], -1)
        if batch_size is not None:
            if condition_array.shape[0] == 1:
                condition_array = np.repeat(condition_array, batch_size, axis=0)
            elif condition_array.shape[0] != batch_size:
                condition_array = condition_array[:batch_size]
        return condition_array


# ---------------------------------------------------------------------------
# Model predictor (wraps the torch model for numpy-based sampling loops)
# ---------------------------------------------------------------------------

class ModelPredictor:
    """Tiny adapter for model inference during sampling."""

    def __init__(self, *, model: object | None = None) -> None:
        self.model = model

    def predict(
        self,
        x: np.ndarray,
        t: np.ndarray,
        condition: dict[object, object] | None = None,
    ) -> np.ndarray:
        """Predict noise ε for trajectory ``x`` at timesteps ``t``."""
        if self.model is None:
            return np.zeros_like(x)

        if hasattr(self.model, "predict_numpy"):
            return np.asarray(self.model.predict_numpy(x, t), dtype=float)

        if hasattr(self.model, "forward"):
            with torch.no_grad():
                x_t = torch.as_tensor(x, dtype=torch.float32)
                t_t = torch.as_tensor(t, dtype=torch.long)
                eps = self.model(x_t, t_t)
                return np.asarray(eps.detach().cpu().numpy(), dtype=float)

        return np.zeros_like(x, dtype=float)


# ---------------------------------------------------------------------------
# Guidance policy (computes ∇J for a value model)
# ---------------------------------------------------------------------------

class GuidancePolicy:
    """Compute gradient-based guidance from a differentiable value model.

    The gradient ∇J(μ) is used in Algorithm 1 to shift the reverse-process mean:
        τ^{i-1} ~ N(μ + a·Σ·∇J(μ), Σ^i)  (a = alpha_t)
    """

    def __init__(
        self,
        model: object | None = None,
        verbose: bool = False,
        *,
        fallback_scale: float = 1.0,
    ) -> None:
        self.model = model
        self.verbose = verbose
        self._fallback_scale = float(fallback_scale)

    def __call__(
        self,
        x: np.ndarray,
        t: np.ndarray,
        condition: dict[object, object] | None = None,
    ) -> np.ndarray:
        """Return ∇J(x) or a heuristic fallback gradient."""
        state_dim = x.shape[-1]
        if self.model is not None:
            grad = self._value_gradient(x)
            if grad is not None:
                return grad
        return self._goal_gradient(x, condition, state_dim)

    def _goal_gradient(
        self,
        x: np.ndarray,
        condition: dict[object, object] | None,
        state_dim: int,
    ) -> np.ndarray:
        """Fallback gradient: pull trajectory toward goal state."""
        goal = ConditionAdapter.to_vector(condition, state_dim)
        if goal.size == 0:
            return np.zeros_like(x, dtype=float)
        return np.clip((goal.reshape(1, 1, -1) - x) * self._fallback_scale, -1.0, 1.0)

    def _value_gradient(self, x: np.ndarray) -> np.ndarray | None:
        """Compute ∇_x J(x) via autograd through the value model."""
        if not callable(self.model):
            return None
        if x.ndim != 3:
            return None
        x_t = torch.as_tensor(x, dtype=torch.float32).requires_grad_(True)
        try:
            with torch.enable_grad():
                value = self.model(x_t)
                if not torch.is_tensor(value):
                    return None
                value.mean().backward()
            if x_t.grad is None:
                return None
            grad = x_t.grad.detach().cpu().numpy()
            if not np.isfinite(grad).all():
                return None
            return np.clip(grad, -1.0, 1.0)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Core reverse-diffusion loop (Algorithm 1)
# ---------------------------------------------------------------------------

class DiffusionSamplingEngine:
    """Reverse diffusion with optional guidance and inpainting conditioning.

    Implements Algorithm 1 from Janner et al. 2022:

    1.  Initialise τ^N ~ N(0, I)
    2.  For i = N, …, 1:
        a.  Compute reverse-process mean  μ = μ_θ(τ^i, i)
        b.  (If guide) compute gradient   g = ∇J(μ)
        c.  Shift mean:                   μ ← μ + a·Σ·g  (a = alpha_t)
        d.  Sample:                       τ^{i-1} ~ N(μ, Σ^i)
        e.  Inpaint constrained timesteps in τ^{i-1}
    3.  Return τ^0
    """

    def __init__(self, schedule: DiffusionSchedule, *, seed: int | None = None) -> None:
        self.schedule = schedule
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _apply_inpainting(
        trajectory: np.ndarray,
        inpaint: dict[int, np.ndarray],
    ) -> np.ndarray:
        """Overwrite constrained timesteps with the known conditioning values."""
        if not inpaint:
            return trajectory
        traj = trajectory.copy()
        for t_idx, value in inpaint.items():
            state_dim = min(value.shape[0], traj.shape[-1])
            traj[:, t_idx, :state_dim] = value[:state_dim]
        return traj

    def sample(
        self,
        model: object,
        *,
        sample_shape: tuple[int, int, int],
        guide: object | None = None,
        schedule: DiffusionSchedule | None = None,
        condition: dict[object, object] | None = None,
        n_guide_steps: int = 2,
        verbose: bool = False,
        t_stopgrad: int = 2,
        scale_grad_by_std: bool = True,
        scale: float = 0.1,
        seed: int | None = None,
        **_: object,
    ) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        active_schedule = self.schedule if schedule is None else schedule
        batch, _, state_dim = sample_shape
        predictor = ModelPredictor(model=model)
        guide_policy: GuidancePolicy | None = guide if guide is not None else None

        # Build inpainting dict from conditions.
        inpaint = ConditionAdapter.to_inpainting(condition, state_dim)

        # Initialise from standard Gaussian; immediately inpaint known states.
        trajectory = self.rng.normal(size=sample_shape)
        trajectory = self._apply_inpainting(trajectory, inpaint)

        for step in range(active_schedule.n_diffusion_steps - 1, -1, -1):
            t_index = np.full((batch,), step, dtype=int)

            # --- ε-prediction and reverse-process mean ---
            eps = predictor.predict(trajectory, t_index)
            if eps.shape != trajectory.shape:
                eps = np.zeros_like(trajectory)

            alpha = float(active_schedule.alpha[step])
            alpha_bar = float(active_schedule.alpha_bar[step])
            beta = float(active_schedule.beta(step))
            sigma = float(np.sqrt(max(1e-12, active_schedule.posterior_variance[step])))

            den = float(np.sqrt(max(1e-12, 1.0 - alpha_bar)))
            # μ = (τ^i - β/sqrt(1-alpha_bar) * ε_θ) / sqrt(alpha)  (DDPM reverse mean)
            mu = (trajectory - beta / den * eps) / float(np.sqrt(max(1e-12, alpha)))

            # --- Guided sampling (Algorithm 1, lines 7-8) ---
            if guide_policy is not None and step >= t_stopgrad:
                guidance_sum = np.zeros_like(mu)
                for _ in range(max(1, n_guide_steps)):
                    g = guide_policy(mu, t_index, condition)
                    if np.isfinite(g).all():
                        guidance_sum = guidance_sum + g
                grad = guidance_sum / max(1.0, float(n_guide_steps))

                # Scale: alpha * sigma^2 * grad  (Equation 3 in paper)
                guidance_scale = scale * (sigma**2 if scale_grad_by_std else sigma)
                mu = mu + guidance_scale * grad

            # --- Sample τ^{i-1} ~ N(μ, Σ^i) ---
            noise = self.rng.normal(size=sample_shape) if step > 0 else 0.0
            trajectory = mu + sigma * noise

            # --- Inpainting: restore constrained timesteps ---
            trajectory = self._apply_inpainting(trajectory, inpaint)

        return np.asarray(trajectory, dtype=float)


# ---------------------------------------------------------------------------
# Value guide wrapper
# ---------------------------------------------------------------------------

class ValueGuide:
    """Thin wrapper around GuidancePolicy for use with GuidedPolicy."""

    def __init__(self, model: object | None = None, verbose: bool = False, **_: object) -> None:
        self.model = model
        self.verbose = verbose
        self._policy = GuidancePolicy(model=model, verbose=verbose)

    def __call__(
        self,
        x: np.ndarray,
        t: np.ndarray,
        condition: dict[object, object] | None = None,
    ) -> np.ndarray:
        return self._policy(x, t, condition)


# ---------------------------------------------------------------------------
# High-level policy
# ---------------------------------------------------------------------------

class GuidedPolicy:
    """Produces full trajectory batches via guided reverse diffusion.

    Accepts conditions as an inpainting dict ``{timestep_index: state_array}``.
    The start state (timestep 0) and goal state (timestep H-1) are fixed via
    inpainting, not via a condition vector passed to the denoising network.
    """

    def __init__(
        self,
        *,
        guide: ValueGuide | None,
        scale: float,
        diffusion_model: object,
        normalizer: PlannerStateNormalizer,
        preprocess_fns: list[Callable[[dict[object, object] | None], dict[object, object]]]
        | None = None,
        sample_fn: Callable[..., np.ndarray] | None = None,
        n_guide_steps: int = 2,
        t_stopgrad: int = 2,
        scale_grad_by_std: bool = True,
        verbose: bool = False,
        **_: object,
    ) -> None:
        self.guide = guide
        self.scale = scale
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.preprocess_fns = preprocess_fns or []
        self.n_guide_steps = n_guide_steps
        self.t_stopgrad = t_stopgrad
        self.scale_grad_by_std = scale_grad_by_std
        self.verbose = verbose

        self.horizon = max(1, int(getattr(diffusion_model, "horizon", normalizer.mean.size)))
        self.state_dim = max(1, int(getattr(diffusion_model, "state_dim", normalizer.mean.size)))
        self.schedule = DiffusionSchedule.cosine(
            n_diffusion_steps=int(getattr(diffusion_model, "n_diffusion_steps", 100))
        )
        self._engine = DiffusionSamplingEngine(self.schedule)
        self.sample_fn = sample_fn if sample_fn is not None else self._engine.sample

    def _prepare_conditions(
        self, conditions: dict[object, object] | None
    ) -> dict[object, object]:
        prepared = conditions or {}
        for fn in self.preprocess_fns:
            prepared = fn(prepared)  # type: ignore[assignment]
        return prepared

    def _sample(
        self,
        conditions: dict[object, object] | None,
        batch_size: int,
        verbose: bool,
    ) -> np.ndarray:
        prepared = self._prepare_conditions(conditions)
        # Normalize inpainting conditions from raw coordinates to model space.
        # Integer keys are timestep indices; their values must be in the same
        # normalized space the diffusion model was trained on.
        norm_cond: dict[object, object] = {}
        for k, v in prepared.items():
            if isinstance(k, (int, np.integer)):
                arr = np.asarray(v, dtype=float).reshape(1, -1)
                norm_cond[k] = self.normalizer.normalize(arr)[0]
            else:
                norm_cond[k] = v
        return self.sample_fn(
            self.diffusion_model,
            sample_shape=(batch_size, self.horizon, self.state_dim),
            schedule=self.schedule,
            guide=self.guide,
            condition=norm_cond,
            n_guide_steps=self.n_guide_steps,
            t_stopgrad=self.t_stopgrad,
            scale_grad_by_std=self.scale_grad_by_std,
            scale=self.scale,
            verbose=verbose or self.verbose,
        )

    def __call__(
        self,
        conditions: dict[object, object],
        batch_size: int,
        verbose: bool = False,
    ) -> SimpleNamespace:
        raw = self._sample(conditions, batch_size=batch_size, verbose=verbose)
        if raw.shape[-1] != self.state_dim:
            raw = raw[..., : self.state_dim]
        observations = self.normalizer.denormalize(raw)
        return SimpleNamespace(observations=observations)


__all__ = [
    "ConditionAdapter",
    "DiffusionSamplingEngine",
    "GuidancePolicy",
    "GuidedPolicy",
    "ModelPredictor",
    "ValueGuide",
]
