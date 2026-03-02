"""Config validation tests for diffusion training."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from planning.diffusion.config import DiffusionTrainingConfig


def _base_training_args(output_path: Path) -> dict[str, object]:
    return {
        "dataset": "dummy.npz",
        "output_path": str(output_path),
        "state_dim": 3,
        "horizon": 5,
        "epochs": 1,
        "batch_size": 8,
        "n_diffusion_steps": 16,
    }


def test_lr_schedule_is_normalized_in_config(tmp_path: Path) -> None:
    cfg = DiffusionTrainingConfig(
        **_base_training_args(tmp_path / "logs"),
        lr_schedule="  StEp ",
        lr_step_size=100,
        lr_gamma=0.5,
    )
    assert cfg.lr_schedule == "step"


def test_lr_schedule_rejects_invalid_schedule(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="Unsupported lr-schedule"):
        DiffusionTrainingConfig(
            **_base_training_args(tmp_path / "logs"),
            lr_schedule="triangular",
        )


def test_lr_schedule_rejects_step_misconfiguration(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="lr_step_size must be positive"):
        DiffusionTrainingConfig(
            **_base_training_args(tmp_path / "logs"),
            lr_schedule="step",
            lr_step_size=0,
        )

    with pytest.raises(ValidationError, match="lr_gamma must be in the interval"):
        DiffusionTrainingConfig(
            **_base_training_args(tmp_path / "logs"),
            lr_schedule="step",
            lr_step_size=1,
            lr_gamma=1.0,
        )


def test_lr_min_is_non_negative(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="lr_min must be non-negative"):
        DiffusionTrainingConfig(
            **_base_training_args(tmp_path / "logs"),
            lr_min=-1e-3,
        )


def test_patience_and_delta_are_validated_in_config(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="diffusion_patience must be >= 1"):
        DiffusionTrainingConfig(
            **_base_training_args(tmp_path / "logs"),
            diffusion_patience=0,
        )
    with pytest.raises(ValidationError, match="value_patience must be >= 1"):
        DiffusionTrainingConfig(
            **_base_training_args(tmp_path / "logs"),
            value_patience=0,
        )
    with pytest.raises(ValidationError, match="diffusion_min_delta must be >= 0"):
        DiffusionTrainingConfig(
            **_base_training_args(tmp_path / "logs"),
            diffusion_min_delta=-0.1,
        )
    with pytest.raises(ValidationError, match="value_min_delta must be >= 0"):
        DiffusionTrainingConfig(
            **_base_training_args(tmp_path / "logs"),
            value_min_delta=-0.1,
        )


def test_checkpoint_policy_and_validation_split_are_validated_in_config(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="checkpoint_every must be >= 0"):
        DiffusionTrainingConfig(
            **_base_training_args(tmp_path / "logs"),
            checkpoint_every=-1,
        )
    with pytest.raises(ValidationError, match="keep_last_checkpoints must be >= 0"):
        DiffusionTrainingConfig(
            **_base_training_args(tmp_path / "logs"),
            keep_last_checkpoints=-1,
        )
    with pytest.raises(ValidationError, match="latest_checkpoint_every must be >= 0"):
        DiffusionTrainingConfig(
            **_base_training_args(tmp_path / "logs"),
            latest_checkpoint_every=-1,
        )
    with pytest.raises(ValidationError, match=r"validation_split must be in \[0.0, 1.0\)"):
        DiffusionTrainingConfig(
            **_base_training_args(tmp_path / "logs"),
            validation_split=1.0,
        )


def test_dataset_key_is_derived_from_dataset_path(tmp_path: Path) -> None:
    args = _base_training_args(tmp_path / "logs")
    args["dataset"] = str(tmp_path / "data" / "my_dataset.npz")
    cfg = DiffusionTrainingConfig(
        **args,
    )
    assert cfg.dataset_key == "my_dataset"


def test_dataset_key_falls_back_when_dataset_stem_is_empty(tmp_path: Path) -> None:
    args = _base_training_args(tmp_path / "logs")
    args["dataset"] = "."
    cfg = DiffusionTrainingConfig(
        **args,
    )
    assert cfg.dataset_key == "dataset"
