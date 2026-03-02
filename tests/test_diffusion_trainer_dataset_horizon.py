"""Dataset horizon validation tests for diffusion trainer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError


def _pipeline_kwargs(dataset_path: Path, output_path: Path) -> dict[str, object]:
    return {
        "dataset": str(dataset_path),
        "output_path": str(output_path),
        "state_dim": 3,
        "n_diffusion_steps": 16,
        "epochs": 1,
        "batch_size": 8,
        "train_diffusion": False,
        "train_value": False,
    }


def test_training_pipeline_requires_integer_horizon(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from planning.diffusion.training.trainer import DiffusionTrainingPipeline

    dataset = np.random.RandomState(0).randn(8, 5, 3)
    dataset_path = tmp_path / "toy.npz"
    np.savez(dataset_path, observations=dataset)

    with pytest.raises(ValidationError):
        DiffusionTrainingPipeline(
            **_pipeline_kwargs(dataset_path, tmp_path / "logs"),
            horizon=None,
        )
