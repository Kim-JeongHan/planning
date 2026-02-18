"""Unit tests for local diffuser sampling wrappers."""

from __future__ import annotations

import numpy as np

from planning.diffusion.core import PlannerStateNormalizer
from planning.diffusion.sampling import (
    DiffusionSamplingEngine,
    GuidedPolicy,
    ValueGuide,
)
from planning.diffusion.training.noise import DiffusionSchedule


def test_engine_output_shape_determinism() -> None:
    schedule = DiffusionSchedule.linear(n_diffusion_steps=6)

    engine_a = DiffusionSamplingEngine(schedule, seed=123)
    sample_a = engine_a.sample(
        model=None,
        sample_shape=(2, 4, 3),
        guide=ValueGuide(),
        condition=None,
        n_guide_steps=1,
    )

    engine_b = DiffusionSamplingEngine(schedule, seed=123)
    sample_b = engine_b.sample(
        model=None,
        sample_shape=(2, 4, 3),
        guide=ValueGuide(),
        condition=None,
        n_guide_steps=1,
    )

    assert sample_a.shape == (2, 4, 3)
    assert sample_b.shape == (2, 4, 3)
    assert np.allclose(sample_a, sample_b)
    assert np.isfinite(sample_a).all()


def test_sample_shape_and_type() -> None:
    schedule = DiffusionSchedule.linear(n_diffusion_steps=8)
    PlannerStateNormalizer.identity(3)

    raw = DiffusionSamplingEngine(schedule, seed=2024).sample(
        model=None,
        sample_shape=(2, 5, 3),
        schedule=schedule,
        guide=ValueGuide(),
        condition=None,
        n_guide_steps=2,
    )

    assert raw.shape == (2, 5, 3)
    assert np.isfinite(raw).all()


def test_guided_policy_returns_observations() -> None:
    schedule = DiffusionSchedule.linear(n_diffusion_steps=4)
    normalizer = PlannerStateNormalizer.identity(3)

    class DummyModel:
        horizon = 5
        state_dim = 3
        n_diffusion_steps = 4

    policy = GuidedPolicy(
        guide=ValueGuide(),
        scale=0.1,
        diffusion_model=DummyModel(),
        normalizer=normalizer,
        preprocess_fns=[],
        sample_fn=DiffusionSamplingEngine(schedule, seed=0).sample,
    )
    policy.schedule = schedule

    result = policy({}, batch_size=3, verbose=False)
    assert hasattr(result, "observations")
    assert result.observations.shape == (3, 5, 3)


def test_guided_policy_default_uses_engine_sampler() -> None:
    schedule = DiffusionSchedule.linear(n_diffusion_steps=4)

    class DummyModel:
        horizon = 7
        state_dim = 3
        n_diffusion_steps = 4

    policy = GuidedPolicy(
        guide=ValueGuide(),
        scale=0.1,
        diffusion_model=DummyModel(),
        normalizer=PlannerStateNormalizer.identity(3),
        preprocess_fns=[],
    )
    policy.schedule = schedule

    result = policy({}, batch_size=2, verbose=False)
    assert hasattr(result, "observations")
    assert result.observations.shape == (2, policy.horizon, policy.state_dim)
