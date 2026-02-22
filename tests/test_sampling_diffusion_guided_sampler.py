"""Tests for DiffusionGuidedSampler."""

import numpy as np
import pytest


def _import_diffusion_sampler():
    """Import DiffusionGuidedSampler when optional dependency is available."""
    pytest.importorskip("planning.diffusion")
    from planning.sampling.diffusion_guided_sampler import DiffusionGuidedSampler

    return DiffusionGuidedSampler


class _FakePolicy:
    """Deterministic fake diffusion policy."""

    def __init__(self, observations: np.ndarray) -> None:
        self.observations = observations

    def __call__(self, conditions: dict[str, np.ndarray], batch_size: int, verbose: bool = False):
        return None, self


def test_sampler_projects_default_xyz_components() -> None:
    """Default projection should extract the first three dimensions."""
    sampler_cls = _import_diffusion_sampler()

    sampler = sampler_cls(
        bounds=[(-1.0, 1.0), (-1.0, 1.0), (0.0, 1.0)],
        dataset="unit-test-dataset",
        sample_batch_size=1,
        max_projection_retries=4,
    )
    sampler._build_policy = lambda: _FakePolicy(
        np.array(
            [
                [
                    [2.0, 0.2, 0.3, 9.0],
                    [0.1, 0.2, 0.5, 9.0],
                ]
            ],
            dtype=float,
        )
    )

    sample = sampler.sample()

    assert np.allclose(sample, np.array([0.1, 0.2, 0.5]))


def test_sampler_supports_state_projection_callable() -> None:
    """Custom projection callback should control how raw state is mapped to 3D."""
    sampler_cls = _import_diffusion_sampler()

    sampler = sampler_cls(
        bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        dataset="unit-test-dataset",
        state_projection=lambda state: state[3:6],
        sample_batch_size=1,
        max_projection_retries=4,
    )
    sampler._build_policy = lambda: _FakePolicy(
        np.array(
            [
                [[1.0, 2.0, 3.0, 0.2, 0.3, 0.4]],
            ],
            dtype=float,
        )
    )

    sample = sampler.sample()

    assert np.allclose(sample, np.array([0.2, 0.3, 0.4]))


def test_sampler_raises_on_projection_failures() -> None:
    """Sampler should fail explicitly after retry budget is exhausted."""
    sampler_cls = _import_diffusion_sampler()

    sampler = sampler_cls(
        bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        dataset="unit-test-dataset",
        sample_batch_size=1,
        max_projection_retries=2,
    )
    sampler._build_policy = lambda: _FakePolicy(np.array([[[2.0, 2.0, 2.0]]], dtype=float))

    with pytest.raises(ValueError, match="No valid 3D sample"):
        sampler.sample()
