"""Local minimal diffuser compatibility layer used by planning.

This package intentionally provides a minimal local diffuser-compatible surface
for checkpoint loading, sampling, and training workflows.
"""

from __future__ import annotations

from .inference import extract_trajectory_observations, sample_trajectory_batch
from .sampling import (
    ConditionAdapter,
    DiffusionSamplingEngine,
    GuidancePolicy,
    GuidedPolicy,
    ModelPredictor,
    ValueGuide,
)
from .training.checkpoint import (
    CheckpointConfig,
    CheckpointLoader,
    CheckpointManager,
    CheckpointWriter,
)
from .training.trainer import DiffusionTrainingPipeline
from .utils import (
    Config,
    DiffusionArtifactLoader,
    check_compatibility,
)

__all__ = [
    "CheckpointConfig",
    "CheckpointLoader",
    "CheckpointManager",
    "CheckpointWriter",
    "ConditionAdapter",
    "Config",
    "DiffusionArtifactLoader",
    "DiffusionSamplingEngine",
    "DiffusionTrainingPipeline",
    "GuidancePolicy",
    "GuidedPolicy",
    "ModelPredictor",
    "ValueGuide",
    "check_compatibility",
    "extract_trajectory_observations",
    "sample_trajectory_batch",
]
