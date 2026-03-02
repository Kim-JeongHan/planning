"""Unit tests for diffuser checkpoint utility objects."""

from __future__ import annotations

from pathlib import Path

import pytest

from planning.diffusion.core import PlannerStateNormalizer
from planning.diffusion.training.checkpoint import CheckpointManager
from planning.diffusion.utils import DiffusionArtifactLoader


def test_checkpoint_manager_collects_epoch_files(tmp_path: Path) -> None:
    root = tmp_path / "logs" / "diffusion" / "defaults_H8_T16"
    root.mkdir(parents=True, exist_ok=True)
    (root / "epoch_0010.ckpt").write_text("stub")
    (root / "epoch_0002.ckpt").write_text("stub")

    manager = CheckpointManager.for_loading(str(root))
    candidates = manager.candidates()

    assert manager.root == root
    assert [path.name for _, path in candidates] == ["epoch_0002.ckpt", "epoch_0010.ckpt"]


def test_diffusion_artifact_loader_loads_epoch_checkpoint(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    from planning.diffusion.model import DiffusionModel

    root = tmp_path / "logs" / "diffusion" / "defaults_H4_T8"
    root.mkdir(parents=True, exist_ok=True)

    model = DiffusionModel(
        state_dim=3,
        horizon=4,
        n_diffusion_steps=8,
        dim=32,
    )
    normalizer = PlannerStateNormalizer.identity(3)
    payload = {
        "model_class_path": "planning.diffusion.model.DiffusionModel",
        "model_kwargs": {
            "state_dim": 3,
            "horizon": 4,
            "n_diffusion_steps": 8,
            "dim": 32,
        },
        "normalizer": normalizer.to_dict(),
        "meta": {
            "horizon": 4,
            "state_dim": 3,
            "n_diffusion_steps": 8,
            "dataset": "unit-dataset",
            "name": "unit-dataset",
            "epoch": 1,
            "loss": 0.123,
        },
        "model_state_dict": model.state_dict(),
        "ema_state_dict": model.state_dict(),
    }
    ckpt_path = root / "epoch_0001.ckpt"
    torch.save(payload, ckpt_path)

    manager = CheckpointManager.for_loading(str(root))
    artifact = DiffusionArtifactLoader(manager).load("1")

    assert artifact.dataset.name == "unit-dataset"
    assert artifact.dataset.horizon == 4
    assert artifact.dataset.state_dim == 3
    assert "path" in artifact.meta


def test_checkpoint_manager_accepts_absolute_checkpoint_path(tmp_path: Path) -> None:
    root = tmp_path / "standalone"
    root.mkdir(parents=True, exist_ok=True)

    manager = CheckpointManager.for_loading(str(root))
    assert manager.root == root
