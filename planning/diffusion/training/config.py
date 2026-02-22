"""Configuration models for local diffuser training."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..config import DiffusionTrainingPipelineConfig
from ..core import PlannerStateNormalizer


class DiffusionTrainingConfig(DiffusionTrainingPipelineConfig):
    """Configuration used by the local trainer."""

    dataset_name: str = "dataset"
    n_hidden: int = 256

    @staticmethod
    def _derive_normalizer(trajectories: np.ndarray) -> PlannerStateNormalizer:
        return PlannerStateNormalizer.fit(trajectories)

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_pipeline_config(
        cls,
        pipeline_config: DiffusionTrainingPipelineConfig,
        *,
        horizon: int,
        dataset_name: str,
        coerce_lr_schedule: Callable[..., str],
    ) -> DiffusionTrainingConfig:
        """Build runtime training config from pipeline args."""
        values = pipeline_config.model_dump()
        values["dataset_name"] = dataset_name
        values["horizon"] = horizon
        values["lr_schedule"] = coerce_lr_schedule(
                pipeline_config.lr_schedule,
                step_size=pipeline_config.lr_step_size,
                gamma=pipeline_config.lr_gamma,
                lr_min=pipeline_config.lr_min,
            )
        return cls(**values)
