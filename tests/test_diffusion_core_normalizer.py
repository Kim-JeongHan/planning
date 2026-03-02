"""Tests for tensor-first PlannerStateNormalizer behavior."""

from __future__ import annotations

import numpy as np
import torch  # type: ignore

from planning.diffusion.core import PlannerStateNormalizer


def test_normalizer_tensor_input_returns_tensor() -> None:
    trajectories = torch.randn(8, 5, 3)
    normalizer = PlannerStateNormalizer.fit(trajectories)

    normalized = normalizer.normalize(trajectories)
    assert torch.is_tensor(normalized)
    assert normalized.shape == trajectories.shape

    restored = normalizer.denormalize(normalized)
    assert torch.is_tensor(restored)
    assert restored.shape == trajectories.shape


def test_normalizer_numpy_input_returns_numpy() -> None:
    trajectories = np.random.RandomState(0).randn(4, 5, 3).astype(np.float32)
    normalizer = PlannerStateNormalizer.fit(trajectories)

    normalized = normalizer.normalize(trajectories)
    assert isinstance(normalized, np.ndarray)
    assert normalized.shape == trajectories.shape

    restored = normalizer.denormalize(normalized)
    assert isinstance(restored, np.ndarray)
    assert restored.shape == trajectories.shape
