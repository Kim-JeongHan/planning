"""Local minimal diffuser compatibility layer used by planning.

This package intentionally implements only the interfaces required by
``DiffusionGuidedSampler`` while keeping the public surface similar to the
external ``diffuser`` dependency originally referenced by the project.
"""

from __future__ import annotations

from .sampling import (
    ConditionAdapter,
    DiffusionSamplingEngine,
    GuidancePolicy,
    GuidedPolicy,
    ModelPredictor,
    ValueGuide,
)
from .training.checkpoint import CheckpointConfig, CheckpointPathManager, CheckpointWriter
from .training.trainer import DiffusionTrainingPipeline
from .utils import (
    CheckpointCatalog,
    Config,
    DiffusionArtifactLoader,
    TemplatingContextResolver,
    check_compatibility,
)

__all__ = [
    "CheckpointCatalog",
    "CheckpointConfig",
    "CheckpointPathManager",
    "CheckpointWriter",
    "Config",
    "ConditionAdapter",
    "DiffusionArtifactLoader",
    "DiffusionSamplingEngine",
    "DiffusionTrainingPipeline",
    "GuidedPolicy",
    "GuidancePolicy",
    "ModelPredictor",
    "check_compatibility",
    "TemplatingContextResolver",
    "ValueGuide",
]
