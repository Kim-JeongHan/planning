"""Shared configuration models for diffusion inference and training."""

from __future__ import annotations

from pydantic import BaseModel


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


class DiffusionTrainingPipelineConfig(DiffusionBaseConfig):
    """Structured configuration for local diffusion training pipeline orchestration."""

    n_diffusion_steps: int
    output_root: str
    horizon: int | None
    state_dim: int
    epochs: int
    batch_size: int
    learning_rate: float = 1e-3
    lr_schedule: str = "constant"
    lr_step_size: int = 100
    lr_gamma: float = 0.5
    lr_min: float = 1e-5
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
    best_top_k: int = 1
    tensorboard_log_dir: str | None = None
    validation_split: float = 0.0
    latest_checkpoint_every: int = 0


__all__ = [
    "DiffusionBaseConfig",
    "DiffusionConfig",
    "DiffusionTrainingPipelineConfig",
]
