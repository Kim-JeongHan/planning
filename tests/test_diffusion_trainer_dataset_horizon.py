"""Dataset horizon validation tests for diffusion trainer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _pipeline_kwargs(dataset_path: Path, output_root: Path) -> dict[str, object]:
    return {
        "dataset": str(dataset_path),
        "output_root": str(output_root),
        "state_dim": 3,
        "n_diffusion_steps": 16,
        "epochs": 1,
        "batch_size": 8,
        "train_diffusion": False,
        "train_value": False,
    }


def test_load_and_prepare_dataset_requires_horizon(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from planning.diffusion.training.trainer import DiffusionTrainingPipeline

    dataset = np.random.RandomState(0).randn(8, 5, 3)
    dataset_path = tmp_path / "toy.npz"
    np.savez(dataset_path, observations=dataset)

    pipeline = DiffusionTrainingPipeline(
        **_pipeline_kwargs(dataset_path, tmp_path / "logs"),
        horizon=None,
    )

    with pytest.raises(ValueError, match="horizon must be provided"):
        pipeline._load_and_prepare_dataset()
