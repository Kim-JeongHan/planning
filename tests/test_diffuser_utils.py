"""Unit tests for diffuser checkpoint utility objects."""

from __future__ import annotations

from pathlib import Path

import pytest

from planning.diffusion.core import PlannerStateNormalizer
from planning.diffusion.utils import (
    CheckpointCatalog,
    DiffusionArtifactLoader,
    TemplatingContextResolver,
)


def test_templating_context_resolver_fallback() -> None:
    resolver = TemplatingContextResolver()
    resolved = resolver.resolve(
        "f:diffusion/defaults_H{horizon}_T{n_diffusion_steps}",
        dataset="unit-dataset",
    )
    assert resolved == "diffusion/defaults_H64_T100"


def test_checkpoint_catalog_collects_epoch_files(tmp_path: Path) -> None:
    loadbase = tmp_path / "logs"
    dataset = "unit-dataset"
    runpath = "f:diffusion/defaults_H{horizon}_T{n_diffusion_steps}"
    root = loadbase / dataset / "diffusion" / "defaults_H8_T16"
    root.mkdir(parents=True, exist_ok=True)
    (root / "epoch_0010.ckpt").write_text("stub")
    (root / "epoch_0002.ckpt").write_text("stub")

    catalog = CheckpointCatalog(str(loadbase), dataset=dataset, loadpath=runpath)
    candidates = catalog.candidates()

    assert catalog.root == root
    assert [path.name for _, path in candidates] == ["epoch_0002.ckpt", "epoch_0010.ckpt"]


def test_diffusion_artifact_loader_loads_epoch_checkpoint(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    from planning.diffusion.model import SimpleDiffusionModel

    dataset = "unit-dataset"
    loadbase = tmp_path / "logs"
    root = loadbase / dataset / "diffusion" / "defaults_H4_T8"
    root.mkdir(parents=True, exist_ok=True)

    model = SimpleDiffusionModel.create(
        state_dim=3,
        horizon=4,
        n_diffusion_steps=8,
        n_hidden=32,
        n_layers=1,
        condition_dim=6,
    )
    normalizer = PlannerStateNormalizer.identity(3)
    payload = {
        "model_class_path": "planning.diffusion.model.DiffusionModel",
        "model_kwargs": {
            "state_dim": 3,
            "horizon": 4,
            "n_diffusion_steps": 8,
            "n_hidden": 32,
            "n_layers": 1,
            "condition_dim": 6,
        },
        "normalizer": normalizer.to_dict(),
        "meta": {
            "horizon": 4,
            "state_dim": 3,
            "n_diffusion_steps": 8,
            "dataset": dataset,
            "name": dataset,
            "epoch": 1,
            "loss": 0.123,
        },
        "model_state_dict": model.state_dict(),
        "ema_state_dict": model.state_dict(),
    }
    ckpt_path = root / "epoch_0001.ckpt"
    torch.save(payload, ckpt_path)

    catalog = CheckpointCatalog(
        str(loadbase),
        dataset=dataset,
        loadpath="f:diffusion/defaults_H{horizon}_T{n_diffusion_steps}",
    )
    artifact = DiffusionArtifactLoader(catalog).load("1")

    assert artifact.dataset.name == dataset
    assert artifact.dataset.horizon == 4
    assert artifact.dataset.state_dim == 3
    assert "path" in artifact.meta
