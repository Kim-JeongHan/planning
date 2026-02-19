"""Sampling helpers for the local diffuser compatibility layer."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import torch  # type: ignore

from .core import PlannerStateNormalizer
from .training.noise import DiffusionSchedule


class ConditionAdapter:
    """Normalize condition inputs used by guides and samplers."""

    @staticmethod
    def to_vector(conditions: dict[object, object] | None, state_dim: int) -> np.ndarray:
        if not conditions:
            return np.zeros((state_dim,), dtype=float)

        goal = conditions.get("goal", None)
        if isinstance(goal, (list, tuple, np.ndarray)):
            goal_array = np.asarray(goal, dtype=float)
            return goal_array[:state_dim]

        if isinstance(conditions, dict) and len(conditions):
            for key in ("goal", "target", "end", "final", "pose"):
                candidate = conditions.get(key, None)
                if isinstance(candidate, (list, tuple, np.ndarray)):
                    candidate_arr = np.asarray(candidate, dtype=float)
                    if candidate_arr.size >= state_dim:
                        if (
                            candidate_arr.size % state_dim == 0
                            and candidate_arr.size >= 2 * state_dim
                        ):
                            candidate_arr = candidate_arr.reshape(-1, state_dim)
                            return candidate_arr[-1]
                        return candidate_arr[:state_dim]

            last = list(conditions.values())[-1]
            if isinstance(last, (list, tuple, np.ndarray)):
                last_arr = np.asarray(last, dtype=float)
                if last_arr.size >= state_dim:
                    if (
                        last_arr.size % state_dim == 0
                        and last_arr.size >= 2 * state_dim
                    ):
                        last_arr = last_arr.reshape(-1, state_dim)
                        return last_arr[-1]
                    return last_arr[:state_dim]

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


class ModelPredictor:
    """Tiny adapter for model inference during sampling."""

    def __init__(self, *, model: object | None = None) -> None:
        self.model = model

    @staticmethod
    def _as_torch_tensor(value: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32)

    def _to_condition_matrix(
        self,
        condition: dict[object, object] | None,
        batch_size: int,
        fallback_dim: int,
    ) -> np.ndarray | None:
        return ConditionAdapter.to_matrix(
            condition,
            batch_size=batch_size,
            fallback_dim=fallback_dim,
        )

    def _model_condition_dim(self, state_dim: int) -> int:
        condition_dim = getattr(self.model, "condition_dim", None)
        if isinstance(condition_dim, (int, np.integer)) and condition_dim > 0:
            return int(condition_dim)
        return int(state_dim)

    def predict(
        self,
        x: np.ndarray,
        t: np.ndarray,
        condition: dict[object, object] | None,
    ) -> np.ndarray:
        if self.model is None:
            return np.zeros_like(x)

        if hasattr(self.model, "predict_numpy"):
            return np.asarray(self.model.predict_numpy(x, t, condition), dtype=float)

        if hasattr(self.model, "forward"):
            with torch.no_grad():  # type: ignore[operator]
                model_input = self._as_torch_tensor(x)
                t_tensor = torch.as_tensor(t, dtype=torch.long)
                condition_array = self._to_condition_matrix(
                    condition,
                    batch_size=x.shape[0],
                    fallback_dim=self._model_condition_dim(x.shape[-1]),
                )
                condition_tensor = None
                if condition_array is not None:
                    condition_tensor = self._as_torch_tensor(condition_array).to(
                        dtype=model_input.dtype,
                        device=model_input.device,
                    )
                outputs = self.model(model_input, t_tensor, condition_tensor)  # type: ignore[misc]
                return np.asarray(outputs.detach().cpu().numpy(), dtype=float)

        return np.zeros_like(x, dtype=float)


class GuidancePolicy:
    """Reusable guidance policy with optional differentiable model term."""

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
        state_dim = x.shape[-1]
        target = self._fallback_target(x, condition, state_dim)
        if self.model is None:
            return target

        model_guidance = self._model_guidance(x, t, condition)
        if model_guidance is not None:
            return model_guidance
        return target

    @staticmethod
    def _target_goal_vector(condition: dict[object, object] | None, state_dim: int) -> np.ndarray:
        return ConditionAdapter.to_vector(condition, state_dim)

    def _fallback_target(
        self,
        x: np.ndarray,
        condition: dict[object, object] | None,
        state_dim: int,
    ) -> np.ndarray:
        goal = self._target_goal_vector(condition, state_dim)
        if goal.size == 0:
            return np.zeros_like(x, dtype=float)
        goal_vector = goal.reshape((1, 1, -1))
        return np.clip((goal_vector - x) * self._fallback_scale, -1.0, 1.0)

    def _model_condition_dim(self, state_dim: int) -> int:
        condition_dim = getattr(self.model, "condition_dim", None)
        if isinstance(condition_dim, (int, np.integer)) and condition_dim > 0:
            return int(condition_dim)
        return int(state_dim)

    def _to_condition_matrix(
        self,
        condition: dict[object, object] | None,
        batch_size: int,
        state_dim: int,
    ) -> np.ndarray | None:
        if condition is None:
            return None
        fallback_dim = self._model_condition_dim(state_dim)
        return ConditionAdapter.to_matrix(
            condition,
            batch_size=batch_size,
            fallback_dim=fallback_dim,
        )

    def _model_guidance(
        self,
        x: np.ndarray,
        t: np.ndarray,
        condition: dict[object, object] | None,
    ) -> np.ndarray | None:
        if x.ndim != 3:
            return None
        if not callable(self.model):
            return None

        x_tensor = torch.as_tensor(x, dtype=torch.float32)
        x_tensor.requires_grad_(True)
        condition_matrix = self._to_condition_matrix(
            condition,
            batch_size=x.shape[0],
            state_dim=x.shape[-1],
        )
        condition_tensor: torch.Tensor | None = None
        if condition_matrix is not None:
            condition_tensor = torch.as_tensor(
                condition_matrix,
                dtype=x_tensor.dtype,
                device=x_tensor.device,
            )
            if condition_tensor.ndim == 1:
                condition_tensor = condition_tensor.reshape(1, -1)
            if condition_tensor.shape[0] == 1:
                condition_tensor = condition_tensor.expand(x.shape[0], -1)

        try:
            with torch.enable_grad():
                value = self.model(x_tensor, condition_tensor)  # type: ignore[misc]
                if not torch.is_tensor(value):
                    return None
                value = value.reshape(value.shape[0], -1).mean(dim=1)
                value.mean().backward()
                if x_tensor.grad is None:
                    return None
                grad = -x_tensor.grad.detach().cpu().numpy()
                if not np.isfinite(grad).all():
                    return None
                return np.clip(grad, -1.0, 1.0)
        except Exception:
            return None


class DiffusionSamplingEngine:
    """Core reverse-diffusion loop used by guided policy."""

    def __init__(self, schedule: DiffusionSchedule, *, seed: int | None = None) -> None:
        self.schedule = schedule
        self.rng = np.random.default_rng(seed)
        self.predictor = ModelPredictor()
        self.guidance_policy = GuidancePolicy()

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
        trajectory = self.rng.normal(size=sample_shape)
        active_schedule = self.schedule if schedule is None else schedule
        predictor = ModelPredictor(model=model)
        guide_policy = guide if guide is not None else GuidancePolicy(model=None)

        for step in range(active_schedule.n_diffusion_steps - 1, -1, -1):
            t_index = np.full((sample_shape[0],), step, dtype=int)

            model_eps = predictor.predict(trajectory, t_index, condition)
            if model_eps.shape != trajectory.shape:
                model_eps = np.zeros_like(trajectory)

            target = np.zeros_like(trajectory)
            if guide_policy is not None and step >= t_stopgrad:
                total = 0
                for _ in range(max(1, n_guide_steps)):
                    total += guide_policy(trajectory, t_index, condition)
                target = np.asarray(total / max(1.0, float(n_guide_steps)))

            if not np.isfinite(model_eps).all():
                model_eps = np.zeros_like(trajectory)
            if not np.isfinite(target).all():
                target = np.zeros_like(trajectory)

            alpha = float(active_schedule.alpha[step])
            alpha_bar = float(active_schedule.alpha_bar[step])
            beta = float(active_schedule.beta(step))
            posterior_std = float(np.sqrt(max(1e-12, active_schedule.posterior_variance[step])))

            den = np.sqrt(max(1e-12, 1.0 - alpha_bar))
            trajectory = (
                (trajectory - beta / den * model_eps) / np.sqrt(max(1e-12, alpha))
            )

            if guide_policy is not None and step >= t_stopgrad:
                effective_scale = scale / max(1.0, float(step + 1))
                if scale_grad_by_std:
                    effective_scale /= posterior_std
                trajectory = trajectory - effective_scale * target

            if step > 0:
                trajectory = trajectory + posterior_std * self.rng.normal(size=sample_shape)
        return np.asarray(trajectory, dtype=float)


class ValueGuide:
    """Small goal-directed guide used during sampling."""

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


class GuidedPolicy:
    """Policy wrapper that produces entire trajectory trajectories."""

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
        self.sample_fn = sample_fn
        self.n_guide_steps = n_guide_steps
        self.t_stopgrad = t_stopgrad
        self.scale_grad_by_std = scale_grad_by_std
        self.verbose = verbose

        horizon = int(getattr(diffusion_model, "horizon", normalizer.mean.size))
        state_dim = int(getattr(diffusion_model, "state_dim", normalizer.mean.size))
        self.horizon = max(1, horizon)
        self.state_dim = max(1, state_dim)
        self.schedule = DiffusionSchedule.linear(
            n_diffusion_steps=int(getattr(diffusion_model, "n_diffusion_steps", 100))
        )
        self._engine = DiffusionSamplingEngine(self.schedule)
        if self.sample_fn is None:
            self.sample_fn = self._engine.sample

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
        sample_kwargs = {
            "sample_shape": (batch_size, self.horizon, self.state_dim),
            "schedule": self.schedule,
            "guide": self.guide,
            "condition": prepared,
            "n_guide_steps": self.n_guide_steps,
            "t_stopgrad": self.t_stopgrad,
            "scale_grad_by_std": self.scale_grad_by_std,
            "scale": self.scale,
            "verbose": verbose or self.verbose,
        }
        return self.sample_fn(self.diffusion_model, **sample_kwargs)

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
