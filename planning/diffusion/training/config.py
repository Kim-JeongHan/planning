"""Configuration models for local diffuser training."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from ..core import PlannerStateNormalizer


class DiffusionTrainingConfig(BaseModel):
    """Configuration used by the local trainer."""

    dataset: str
    dataset_name: str = "dataset"
    horizon: int = 16
    state_dim: int = 3
    n_diffusion_steps: int = 100
    n_hidden: int = 256
    n_layers: int = 2
    epochs: int = 1
    batch_size: int = 16
    learning_rate: float = 1e-3
    lr_schedule: str = "constant"
    lr_step_size: int = 100
    lr_gamma: float = 0.5
    lr_min: float = 1e-5
    discount: float = 1.0
    train_value: bool = True
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

    @staticmethod
    def _derive_normalizer(trajectories: np.ndarray) -> PlannerStateNormalizer:
        return PlannerStateNormalizer.fit(trajectories)

    class Config:
        arbitrary_types_allowed = True
