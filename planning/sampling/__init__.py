"""Sampling-based path planning algorithms."""

from __future__ import annotations

from ..collision import ObstacleCollisionChecker
from .prm import PRM, PRMConfig
from .rrt import (
    RRT,
    InformedRRTStar,
    RRTConfig,
    RRTConnect,
    RRTConnectConfig,
    RRTStar,
    RRTStarConfig,
)
from .sampler import GoalBiasedSampler, InformedSampler, Sampler, UniformSampler

try:
    from .diffusion_guided_sampler import DiffusionGuidedSampler
except ImportError as _diffusion_error:
    _DIFFUSER_IMPORT_ERROR = _diffusion_error
else:
    _DIFFUSER_IMPORT_ERROR = None


def __getattr__(name: str) -> object:
    if name == "DiffusionGuidedSampler" and _DIFFUSER_IMPORT_ERROR is not None:
        raise _DIFFUSER_IMPORT_ERROR
    raise AttributeError(f"module {__name__} has no attribute {name!r}")


__all__ = [
    "PRM",
    "RRT",
    "GoalBiasedSampler",
    "InformedRRTStar",
    "InformedSampler",
    "ObstacleCollisionChecker",
    "PRMConfig",
    "RRTConfig",
    "RRTConnect",
    "RRTConnectConfig",
    "RRTStar",
    "RRTStarConfig",
    "Sampler",
    "UniformSampler",
]

if _DIFFUSER_IMPORT_ERROR is None:
    __all__.append("DiffusionGuidedSampler")
