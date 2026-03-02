"""Shared configuration models for diffusion inference and training."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_CONFIG_DIR = PROJECT_ROOT / "config"


class DiffusionBaseConfig(BaseModel):
    """Arguments shared by diffusion inference and training configs."""

    dataset: str
    n_diffusion_steps: int = 100
    discount: float = 1.0
    seed: int | None = None


class DiffusionConfig(DiffusionBaseConfig):
    """Diffusion artifact loading and sampling configuration."""

    seed: int = 42
    loadbase: str = "logs/pretrained"
    config: str | None = None
    diffusion_loadpath: str = "f:diffusion/defaults_H{horizon}_T{n_diffusion_steps}"
    value_loadpath: str = "f:values/defaults_H{horizon}_T{n_diffusion_steps}_d{discount}"
    diffusion_epoch: int | str = "latest"
    value_epoch: int | str = "latest"
    n_guide_steps: int = 2
    scale: float = 0.1
    sample_batch_size: int = 4
    condition_key: int | str = 0


class TrajectoryConfig(BaseModel):
    """Dataset loading configuration for trajectory preprocessing."""

    dataset: str
    horizon: int | None
    state_dim: int
    device: str = "cpu"
    seed: int | None = None

    @property
    def dataset_key(self) -> str:
        """Dataset identifier derived from ``dataset`` path."""
        return Path(self.dataset).stem or "dataset"


class CheckpointConfig(BaseModel):
    """Resolved checkpoint path configuration."""

    dataset: str
    horizon: int
    n_diffusion_steps: int
    root: str | Path = "logs"
    discount: float | None = None


class DiffusionTrainingPipelineConfig(DiffusionBaseConfig):
    """Structured configuration for local diffusion training pipeline orchestration."""

    n_diffusion_steps: int
    output_root: str
    horizon: int | None
    state_dim: int
    epochs: int
    batch_size: int
    device: str = "cpu"
    n_hidden: int = 256
    learning_rate: float = 1e-3
    lr_schedule: str = "constant"
    lr_step_size: int = 100
    lr_gamma: float = 0.5
    lr_min: float = 1e-5
    train_diffusion: bool = True
    train_value: bool = True
    log_every: int | None = None
    diffusion_max_epochs: int | None = None
    value_max_epochs: int | None = None
    diffusion_patience: int | None = None
    value_patience: int | None = None
    diffusion_min_delta: float = 0.0
    value_min_delta: float = 0.0
    checkpoint_every: int = 0
    keep_last_checkpoints: int = 0
    validation_split: float = 0.0
    latest_checkpoint_every: int = 0

    @property
    def dataset_key(self) -> str:
        """Dataset identifier derived from ``dataset`` path."""
        return Path(self.dataset).stem or "dataset"

    def to_trajectory_config(self) -> TrajectoryConfig:
        """Build trajectory loading config from pipeline config."""
        return TrajectoryConfig(
            dataset=self.dataset,
            horizon=self.horizon,
            state_dim=self.state_dim,
            device=self.device,
            seed=self.seed,
        )

    def to_checkpoint_config(self) -> CheckpointConfig:
        """Build checkpoint config from pipeline config and resolved dataset key."""
        if self.horizon is None:
            raise ValueError("horizon must be provided before creating checkpoint config.")
        return CheckpointConfig(
            dataset=self.dataset_key,
            horizon=self.horizon,
            n_diffusion_steps=self.n_diffusion_steps,
            root=self.output_root,
            discount=self.discount,
        )

    @field_validator("lr_schedule")
    @classmethod
    def _coerce_lr_schedule(cls, schedule: str) -> str:
        schedule = schedule.strip().lower()
        if schedule not in {"constant", "step", "cosine"}:
            raise ValueError("Unsupported lr-schedule. Choose one of: constant, step, cosine.")
        return schedule

    @model_validator(mode="after")
    def _validate_training_parameters(self) -> DiffusionTrainingPipelineConfig:
        if self.lr_schedule == "step" and self.lr_step_size <= 0:
            raise ValueError("lr_step_size must be positive when lr_schedule='step'.")
        if self.lr_schedule == "step" and not 0.0 < self.lr_gamma < 1.0:
            raise ValueError("lr_gamma must be in the interval (0, 1) for lr_schedule='step'.")
        if self.lr_min < 0:
            raise ValueError("lr_min must be non-negative.")
        if self.diffusion_patience is not None and self.diffusion_patience < 1:
            raise ValueError("diffusion_patience must be >= 1 when enabled.")
        if self.value_patience is not None and self.value_patience < 1:
            raise ValueError("value_patience must be >= 1 when enabled.")
        if self.diffusion_min_delta < 0.0:
            raise ValueError("diffusion_min_delta must be >= 0.")
        if self.value_min_delta < 0.0:
            raise ValueError("value_min_delta must be >= 0.")
        if self.checkpoint_every < 0:
            raise ValueError("checkpoint_every must be >= 0.")
        if self.keep_last_checkpoints < 0:
            raise ValueError("keep_last_checkpoints must be >= 0.")
        if self.latest_checkpoint_every < 0:
            raise ValueError("latest_checkpoint_every must be >= 0.")
        if not 0.0 <= float(self.validation_split) < 1.0:
            raise ValueError("validation_split must be in [0.0, 1.0).")
        return self


__all__ = [
    "PROJECT_CONFIG_DIR",
    "PROJECT_ROOT",
    "CheckpointConfig",
    "DiffusionBaseConfig",
    "DiffusionConfig",
    "DiffusionTrainingPipelineConfig",
    "TrajectoryConfig",
]
