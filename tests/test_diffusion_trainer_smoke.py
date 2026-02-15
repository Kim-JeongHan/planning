"""Smoke test for local diffuser training CLI path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_training_smoke(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from planning.diffusion.training.trainer import train

    dataset = np.random.RandomState(0).randn(24, 5, 3)
    dataset_path = tmp_path / "toy.npz"
    np.savez(dataset_path, observations=dataset)

    checkpoints = train(
        dataset=str(dataset_path),
        output_root=str(tmp_path / "logs"),
        horizon=5,
        state_dim=3,
        n_diffusion_steps=16,
        epochs=1,
        batch_size=4,
        train_value=False,
    )

    assert checkpoints
    assert all((tmp_path / "logs").rglob("*.ckpt"))
