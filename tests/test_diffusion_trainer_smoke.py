"""Smoke test for local diffuser training CLI path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_training_smoke(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from planning.diffusion.training.trainer import DiffusionTrainingPipeline

    dataset = np.random.RandomState(0).randn(24, 5, 3)
    dataset_path = tmp_path / "toy.npz"
    np.savez(dataset_path, observations=dataset)

    checkpoints = DiffusionTrainingPipeline(
        dataset=str(dataset_path),
        output_root=str(tmp_path / "logs"),
        horizon=5,
        state_dim=3,
        n_diffusion_steps=16,
        epochs=1,
        batch_size=4,
        train_value=False,
    ).run()

    assert checkpoints
    assert all((tmp_path / "logs").rglob("*.ckpt"))


def test_training_package_exports_pipeline() -> None:
    from planning.diffusion.training import DiffusionTrainingPipeline as PackagePipeline
    from planning.diffusion.training.trainer import DiffusionTrainingPipeline as ModulePipeline

    assert PackagePipeline is ModulePipeline


def test_train_arg_resolver_parses_advanced_flags() -> None:
    from planning.diffusion.training.cli import TrainArgResolver

    values = TrainArgResolver(
        [
            "--dataset",
            "dummy.npz",
            "--diffusion-max-epochs",
            "2",
            "--value-patience",
            "3",
            "--diffusion-min-delta",
            "0.05",
            "--tensorboard-log-dir",
            "logs/tensorboard",
            "--checkpoint-every",
            "3",
            "--keep-last-checkpoints",
            "2",
            "--best-top-k",
            "2",
        ]
    ).resolve()

    assert values["diffusion_max_epochs"] == 2
    assert values["value_patience"] == 3
    assert values["diffusion_min_delta"] == 0.05
    assert values["tensorboard_log_dir"] == "logs/tensorboard"
    assert values["checkpoint_every"] == 3
    assert values["keep_last_checkpoints"] == 2
    assert values["best_top_k"] == 2
