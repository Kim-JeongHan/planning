"""Unit tests for local diffuser sampling wrappers."""

from __future__ import annotations

import numpy as np

from planning.diffusion.core import PlannerStateNormalizer
from planning.diffusion.sampling import GuidedPolicy, ValueGuide, n_step_guided_p_sample
from planning.diffusion.training.noise import DiffusionSchedule


def test_sample_shape_and_type() -> None:
    schedule = DiffusionSchedule.linear(n_diffusion_steps=8)
    PlannerStateNormalizer.identity(3)

    raw = n_step_guided_p_sample(
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
        sample_fn=n_step_guided_p_sample,
    )
    policy.schedule = schedule

    result = policy({}, batch_size=3, verbose=False)
    assert hasattr(result, "observations")
    assert result.observations.shape == (3, 5, 3)
