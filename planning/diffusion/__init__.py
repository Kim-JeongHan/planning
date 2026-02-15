"""Local minimal diffuser compatibility layer used by planning.

This package intentionally implements only the interfaces required by
``DiffusionGuidedSampler`` while keeping the public surface similar to the
external ``diffuser`` dependency originally referenced by the project.
"""

from __future__ import annotations

from .sampling import GuidedPolicy, ValueGuide, n_step_guided_p_sample
from .utils import Config, check_compatibility, load_diffusion

__all__ = [
    "Config",
    "GuidedPolicy",
    "ValueGuide",
    "check_compatibility",
    "load_diffusion",
    "n_step_guided_p_sample",
]
