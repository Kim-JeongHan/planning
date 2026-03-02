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
from typing import cast

import numpy as np
import torch

from .core import PlannerStateNormalizer
from .training.noise import DiffusionSchedule

# ---------------------------------------------------------------------------
# Condition helpers
# ---------------------------------------------------------------------------

# Inpainting conditions map ``{timestep_index: state_vector}``.
InpaintingConditions = Mapping[int, torch.Tensor]
_GOAL_KEYS = ("goal", "target", "end", "final", "pose")


class ConditionAdapter:
    """Convert various condition formats to inpainting dicts or flat vectors."""

    @staticmethod
    def _to_tensor_vector(
        value: object,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if torch.is_tensor(value):
            tensor = cast(torch.Tensor, value)
            return tensor.to(device=device, dtype=dtype).reshape(-1)
        return torch.as_tensor(value, dtype=dtype, device=device).reshape(-1)

    @staticmethod
    def to_inpainting(
        conditions: dict[object, object] | None,
        state_dim: int,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> dict[int, torch.Tensor]:
        """Normalise a raw conditions dict to ``{int_timestep: tensor}`` format.

        Accepts either:
          - Already-inpainting format: ``{0: array, H-1: array}``
          - Legacy flat-vector format: ``{0: flat_vector}`` (vector interpreted
            as the start state; goal is not constrained)
        """
        if not conditions:
            return {}
        result: dict[int, torch.Tensor] = {}
        for key, value in conditions.items():
            if not isinstance(key, int | np.integer):
                continue
            try:
                arr = ConditionAdapter._to_tensor_vector(value, device=device, dtype=dtype)
            except (TypeError, ValueError, RuntimeError):
                continue
            if arr.numel() == 0:
                continue
            result[int(key)] = arr[:state_dim]
        return result

    @staticmethod
    def to_vector(
        conditions: dict[object, object] | None,
        state_dim: int,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Extract a single goal vector from a conditions dict (legacy helper)."""
        goal = torch.zeros((state_dim,), dtype=dtype, device=device)
        if not conditions:
            return goal
        for key in _GOAL_KEYS:
            candidate = conditions.get(key)
            if candidate is None:
                continue
            try:
                arr = ConditionAdapter._to_tensor_vector(candidate, device=device, dtype=dtype)
            except (TypeError, ValueError, RuntimeError):
                continue
            if arr.numel() == 0:
                continue
            dim = min(int(arr.numel()), state_dim)
            goal[:dim] = arr[:dim]
            return goal
        try:
            arr = ConditionAdapter._to_tensor_vector(
                list(conditions.values())[-1],
                device=device,
                dtype=dtype,
            )
        except (TypeError, ValueError, RuntimeError):
            return goal
        if arr.numel() == 0:
            return goal
        dim = min(int(arr.numel()), state_dim)
        goal[:dim] = arr[:dim]
        return goal


# ---------------------------------------------------------------------------
# Model predictor (wraps the torch model for torch-based sampling loops)
# ---------------------------------------------------------------------------


class ModelPredictor:
    """Tiny adapter for model inference during sampling."""

    def __init__(self, *, model: object, device: torch.device) -> None:
        self.model = model
        self.device = device

    def predict(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: dict[object, object] | None = None,
    ) -> torch.Tensor:
        """Predict noise ε for trajectory ``x`` at timesteps ``t``."""
        if self.model is None:
            return torch.zeros_like(x)

        if hasattr(self.model, "forward"):
            with torch.no_grad():
                x_t = x.to(device=self.device, dtype=torch.float32)
                t_t = t.to(device=self.device, dtype=torch.long)
                model_fn = cast(Callable[[torch.Tensor, torch.Tensor], torch.Tensor], self.model)
                eps = model_fn(x_t, t_t)
                if not torch.is_tensor(eps):
                    return torch.zeros_like(x)
                return eps.to(device=x.device, dtype=x.dtype)

        return torch.zeros_like(x)


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
        *,
        fallback_scale: float = 1.0,
        device: torch.device,
    ) -> None:
        self.model = model
        self._fallback_scale = float(fallback_scale)
        self._device = device

    def __call__(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: dict[object, object] | None = None,
    ) -> torch.Tensor:
        """Return ∇J(x) or a heuristic fallback gradient."""
        x_t = x
        state_dim = x_t.shape[-1]
        if self.model is not None:
            grad = self._value_gradient(x_t)
            if grad is not None:
                return grad
        return self._goal_gradient(x_t, condition, state_dim)

    def _goal_gradient(
        self,
        x: torch.Tensor,
        condition: dict[object, object] | None,
        state_dim: int,
    ) -> torch.Tensor:
        """Fallback gradient: pull trajectory toward goal state."""
        goal_t = ConditionAdapter.to_vector(
            condition,
            state_dim,
            device=x.device,
            dtype=x.dtype,
        ).reshape(1, 1, -1)
        return torch.clamp((goal_t - x) * self._fallback_scale, -1.0, 1.0)

    def _value_gradient(self, x: torch.Tensor) -> torch.Tensor | None:
        """Compute ∇_x J(x) via autograd through the value model."""
        if not callable(self.model):
            return None
        if x.ndim != 3:
            return None
        x_t = x.to(device=self._device, dtype=torch.float32).detach().clone().requires_grad_(True)
        try:
            with torch.enable_grad():
                value = self.model(x_t)
                if not torch.is_tensor(value):
                    return None
                value.mean().backward()
            if x_t.grad is None:
                return None
            grad = x_t.grad.detach()
            if not torch.isfinite(grad).all():
                return None
            return torch.clamp(grad, -1.0, 1.0).to(device=x.device, dtype=x.dtype)
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
        self.seed = seed
        self._generators: dict[str, torch.Generator] = {}

    def _get_generator(self, device: torch.device) -> torch.Generator | None:
        if self.seed is None:
            return None
        if device.type not in {"cpu", "cuda"}:
            return None
        device_key = f"{device.type}:{device.index if device.index is not None else -1}"
        generator = self._generators.get(device_key)
        if generator is None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(self.seed))
            self._generators[device_key] = generator
        return generator

    @staticmethod
    def _apply_inpainting(
        trajectory: torch.Tensor,
        inpaint: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Overwrite constrained timesteps with the known conditioning values."""
        if not inpaint:
            return trajectory
        traj = trajectory.clone()
        for t_idx, value in inpaint.items():
            if t_idx < 0 or t_idx >= traj.shape[1]:
                continue
            dim = min(int(value.shape[0]), int(traj.shape[-1]))
            fill = value[:dim].to(device=traj.device, dtype=traj.dtype).unsqueeze(0)
            traj[:, t_idx, :dim] = fill.expand(traj.shape[0], -1)
        return traj

    def sample(
        self,
        model: object,
        *,
        device: torch.device,
        sample_shape: tuple[int, int, int],
        guide: GuidancePolicy | None = None,
        schedule: DiffusionSchedule | None = None,
        condition: dict[object, object] | None = None,
        n_guide_steps: int = 2,
        t_stopgrad: int = 2,
        scale_grad_by_std: bool = True,
        scale: float = 0.1,
        seed: int | None = None,
        **_kwargs: object,
    ) -> np.ndarray:
        if seed is not None:
            self.seed = int(seed)
            self._generators.clear()

        active_schedule = self.schedule if schedule is None else schedule
        batch, _horizon, state_dim = sample_shape
        generator = self._get_generator(device)
        predictor = ModelPredictor(model=model, device=device)
        guide_policy = guide

        # Build inpainting dict from conditions.
        inpaint = ConditionAdapter.to_inpainting(
            condition,
            state_dim,
            device=device,
            dtype=torch.float32,
        )

        # Initialise from standard Gaussian; immediately inpaint known states.
        trajectory = torch.randn(
            sample_shape,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        trajectory = self._apply_inpainting(trajectory, inpaint)

        for step in range(active_schedule.n_diffusion_steps - 1, -1, -1):
            t_index = torch.full((batch,), step, dtype=torch.long, device=device)

            # --- ε-prediction and reverse-process mean ---
            eps = predictor.predict(trajectory, t_index)
            if eps.shape != trajectory.shape:
                eps = torch.zeros_like(trajectory)

            alpha = float(active_schedule.alpha[step])
            alpha_bar = float(active_schedule.alpha_bar[step])
            beta = float(active_schedule.beta(step))
            sigma = float(np.sqrt(max(1e-12, active_schedule.posterior_variance[step])))

            den = float(np.sqrt(max(1e-12, 1.0 - alpha_bar)))
            # μ = (τ^i - β/sqrt(1-alpha_bar) * ε_θ) / sqrt(alpha)  (DDPM reverse mean)
            mu = (trajectory - beta / den * eps) / float(np.sqrt(max(1e-12, alpha)))

            # --- Guided sampling (Algorithm 1, lines 7-8) ---
            if guide_policy is not None and step >= t_stopgrad:
                guidance_scale = scale * (sigma**2 if scale_grad_by_std else sigma)
                for _guide_step in range(max(1, n_guide_steps)):
                    g = guide_policy(mu, t_index, condition)
                    if not torch.is_tensor(g):
                        g = torch.as_tensor(g, dtype=mu.dtype, device=mu.device)
                    g = g.to(device=mu.device, dtype=mu.dtype)
                    if g.shape != mu.shape:
                        continue
                    if torch.isfinite(g).all():
                        # Apply guidance update at each guide step so n_guide_steps
                        # materially changes the reverse-process trajectory.
                        mu = mu + guidance_scale * g

            # --- Sample τ^{i-1} ~ N(μ, Σ^i) ---
            noise = (
                torch.randn(
                    sample_shape,
                    dtype=trajectory.dtype,
                    device=trajectory.device,
                    generator=generator,
                )
                if step > 0
                else 0.0
            )
            trajectory = mu + sigma * noise

            # --- Inpainting: restore constrained timesteps ---
            trajectory = self._apply_inpainting(trajectory, inpaint)

        return np.asarray(trajectory.detach().cpu().numpy(), dtype=float)


# ---------------------------------------------------------------------------
# Value guide wrapper
# ---------------------------------------------------------------------------


class ValueGuide(GuidancePolicy):
    """GuidancePolicy subclass used as the default guide for GuidedPolicy.

    Accepts and discards unknown kwargs so ``Config("sampling.ValueGuide", ...)``
    calls with extra keyword arguments do not raise.
    """

    def __init__(
        self,
        model: object | None = None,
        *,
        device: torch.device,
        **_: object,
    ) -> None:
        super().__init__(model=model, device=device)


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
        preprocess_fns: (
            list[Callable[[dict[object, object] | None], dict[object, object]]] | None
        ) = None,
        sample_fn: Callable[..., np.ndarray] | None = None,
        n_guide_steps: int = 2,
        t_stopgrad: int = 2,
        scale_grad_by_std: bool = True,
        device: torch.device,
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
        self._device = device

        default_dim = int(normalizer.mean.shape[0])
        self.horizon = max(1, int(getattr(diffusion_model, "horizon", default_dim)))
        self.state_dim = max(1, int(getattr(diffusion_model, "state_dim", default_dim)))
        self.schedule = DiffusionSchedule.cosine(
            n_diffusion_steps=int(getattr(diffusion_model, "n_diffusion_steps", 100))
        )
        self._engine = DiffusionSamplingEngine(self.schedule)
        self.sample_fn = sample_fn if sample_fn is not None else self._engine.sample

    def _prepare_conditions(self, conditions: dict[object, object] | None) -> dict[object, object]:
        prepared = conditions or {}
        for fn in self.preprocess_fns:
            prepared = fn(prepared)  # type: ignore[assignment]
        return prepared

    def _sample(
        self,
        conditions: dict[object, object] | None,
        batch_size: int,
    ) -> np.ndarray:
        prepared = self._prepare_conditions(conditions)
        # Normalize inpainting conditions from raw coordinates to model space.
        # Integer keys are timestep indices; their values must be in the same
        # normalized space the diffusion model was trained on.
        norm_cond: dict[object, object] = {}
        for k, v in prepared.items():
            if isinstance(k, int | np.integer):
                timestep = int(k)
                inpaint_value = ConditionAdapter.to_inpainting(
                    {timestep: v},
                    self.state_dim,
                    device=self._device,
                    dtype=torch.float32,
                ).get(timestep)
                if inpaint_value is None:
                    continue
                norm_cond[timestep] = self.normalizer.normalize_tensor(
                    inpaint_value.unsqueeze(0)
                ).squeeze(0)
            elif isinstance(k, str) and k in _GOAL_KEYS:
                # Keep goal-style condition keys in the same normalized space
                # as inpainting conditions to avoid mixed-space guidance.
                try:
                    goal_value = ConditionAdapter._to_tensor_vector(
                        v,
                        device=self._device,
                        dtype=torch.float32,
                    )
                except (TypeError, ValueError, RuntimeError):
                    norm_cond[k] = v
                    continue
                if goal_value.numel() == 0:
                    continue
                norm_cond[k] = self.normalizer.normalize_tensor(
                    goal_value[: self.state_dim].unsqueeze(0)
                ).squeeze(0)
            else:
                norm_cond[k] = v
        return self.sample_fn(
            self.diffusion_model,
            device=self._device,
            sample_shape=(batch_size, self.horizon, self.state_dim),
            schedule=self.schedule,
            guide=self.guide,
            condition=norm_cond,
            n_guide_steps=self.n_guide_steps,
            t_stopgrad=self.t_stopgrad,
            scale_grad_by_std=self.scale_grad_by_std,
            scale=self.scale,
        )

    def __call__(
        self,
        conditions: dict[object, object],
        batch_size: int,
    ) -> SimpleNamespace:
        raw = self._sample(conditions, batch_size=batch_size)
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
