"""Tests for tensor-based trajectory loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch  # type: ignore

from planning.diffusion.config import TrajectoryConfig
from planning.diffusion.training.dataset import TrajectoryDataSetSource


def test_load_npz_returns_torch_tensors(tmp_path: Path) -> None:
    observations = np.random.RandomState(0).randn(4, 5, 3)
    dataset_path = tmp_path / "toy.npz"
    np.savez(dataset_path, observations=observations)

    source = TrajectoryDataSetSource(
        TrajectoryConfig(
            dataset=str(dataset_path),
            horizon=5,
            state_dim=3,
            device="cpu",
            seed=42,
        )
    )
    trajectories = source.load()

    assert trajectories
    assert all(torch.is_tensor(trajectory) for trajectory in trajectories)
    assert all(trajectory.dtype == torch.float32 for trajectory in trajectories)
    assert all(trajectory.device.type == "cpu" for trajectory in trajectories)
