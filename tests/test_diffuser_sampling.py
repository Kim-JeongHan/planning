"""Unit tests for local diffuser sampling wrappers."""

from __future__ import annotations

import numpy as np
import torch

from planning.diffusion.core import PlannerStateNormalizer
from planning.diffusion.sampling import (
    DiffusionSamplingEngine,
    GuidancePolicy,
    GuidedPolicy,
    ValueGuide,
)
from planning.diffusion.training.noise import DiffusionSchedule

_CPU = torch.device("cpu")


class _DummyModel:
    horizon = 5
    state_dim = 3
    n_diffusion_steps = 4

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


def test_engine_output_shape_determinism() -> None:
    schedule = DiffusionSchedule.linear(n_diffusion_steps=6)
    dummy = _DummyModel()
    guide = ValueGuide(model=dummy, device=_CPU)

    engine_a = DiffusionSamplingEngine(schedule, seed=123)
    sample_a = engine_a.sample(
        model=dummy,
        device=_CPU,
        sample_shape=(2, 4, 3),
        guide=guide,
        condition=None,
        n_guide_steps=1,
    )

    engine_b = DiffusionSamplingEngine(schedule, seed=123)
    sample_b = engine_b.sample(
        model=dummy,
        device=_CPU,
        sample_shape=(2, 4, 3),
        guide=guide,
        condition=None,
        n_guide_steps=1,
    )

    assert sample_a.shape == (2, 4, 3)
    assert sample_b.shape == (2, 4, 3)
    assert np.allclose(sample_a, sample_b)
    assert np.isfinite(sample_a).all()


def test_sample_shape_and_type() -> None:
    schedule = DiffusionSchedule.linear(n_diffusion_steps=8)
    dummy = _DummyModel()

    raw = DiffusionSamplingEngine(schedule, seed=2024).sample(
        model=dummy,
        device=_CPU,
        sample_shape=(2, 5, 3),
        schedule=schedule,
        guide=ValueGuide(model=dummy, device=_CPU),
        condition=None,
        n_guide_steps=2,
    )

    assert raw.shape == (2, 5, 3)
    assert np.isfinite(raw).all()


def test_guided_policy_returns_observations() -> None:
    schedule = DiffusionSchedule.linear(n_diffusion_steps=4)
    normalizer = PlannerStateNormalizer.identity(3)
    dummy = _DummyModel()

    policy = GuidedPolicy(
        guide=ValueGuide(model=dummy, device=_CPU),
        scale=0.1,
        diffusion_model=dummy,
        normalizer=normalizer,
        preprocess_fns=[],
        sample_fn=DiffusionSamplingEngine(schedule, seed=0).sample,
        device=_CPU,
    )
    policy.schedule = schedule

    result = policy({}, batch_size=3)
    assert hasattr(result, "observations")
    assert result.observations.shape == (3, 5, 3)


def test_guided_policy_default_uses_engine_sampler() -> None:
    schedule = DiffusionSchedule.linear(n_diffusion_steps=4)

    class DummyModel:
        horizon = 7
        state_dim = 3
        n_diffusion_steps = 4

    dummy = DummyModel()
    policy = GuidedPolicy(
        guide=ValueGuide(model=dummy, device=_CPU),
        scale=0.1,
        diffusion_model=dummy,
        normalizer=PlannerStateNormalizer.identity(3),
        preprocess_fns=[],
        device=_CPU,
    )
    policy.schedule = schedule

    result = policy({}, batch_size=2)
    assert hasattr(result, "observations")
    assert result.observations.shape == (2, policy.horizon, policy.state_dim)


def test_engine_n_guide_steps_changes_guided_result() -> None:
    schedule = DiffusionSchedule.linear(n_diffusion_steps=6)
    dummy = _DummyModel()

    class _ConstantGuide(GuidancePolicy):
        def __init__(self) -> None:
            super().__init__(model=None, device=_CPU)

        def __call__(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            condition: dict[object, object] | None = None,
        ) -> torch.Tensor:
            del t, condition
            return torch.full_like(x, 0.25)

    sample_once = DiffusionSamplingEngine(schedule, seed=123).sample(
        model=dummy,
        device=_CPU,
        sample_shape=(2, 5, 3),
        schedule=schedule,
        guide=_ConstantGuide(),
        condition=None,
        n_guide_steps=1,
        t_stopgrad=0,
        scale=0.3,
    )
    sample_many = DiffusionSamplingEngine(schedule, seed=123).sample(
        model=dummy,
        device=_CPU,
        sample_shape=(2, 5, 3),
        schedule=schedule,
        guide=_ConstantGuide(),
        condition=None,
        n_guide_steps=5,
        t_stopgrad=0,
        scale=0.3,
    )

    assert sample_once.shape == sample_many.shape
    assert not np.allclose(sample_once, sample_many)


def test_guided_policy_normalizes_goal_alias_conditions() -> None:
    schedule = DiffusionSchedule.linear(n_diffusion_steps=4)
    normalizer = PlannerStateNormalizer(
        mean=torch.tensor([5.0, 5.0, 5.0]),
        std=torch.tensor([2.0, 2.0, 2.0]),
    )
    dummy = _DummyModel()
    captured: dict[str, object] = {}

    def _sample_fn(model: object, **kwargs: object) -> np.ndarray:
        del model
        captured["condition"] = kwargs.get("condition")
        return np.zeros((1, dummy.horizon, dummy.state_dim), dtype=float)

    policy = GuidedPolicy(
        guide=ValueGuide(model=dummy, device=_CPU),
        scale=0.1,
        diffusion_model=dummy,
        normalizer=normalizer,
        preprocess_fns=[],
        sample_fn=_sample_fn,
        device=_CPU,
    )
    policy.schedule = schedule

    policy({0: np.array([7.0, 7.0, 7.0]), "goal": np.array([9.0, 9.0, 9.0])}, batch_size=1)
    condition = captured["condition"]
    assert isinstance(condition, dict)
    start = condition[0]
    goal = condition["goal"]
    assert torch.is_tensor(start)
    assert torch.is_tensor(goal)
    assert torch.allclose(start, torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32))
    assert torch.allclose(goal, torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32))
