"""Optional diffusion planning components.

Install with the ``diffuser`` extra to use torch-based models, sampling,
training, and checkpoint utilities.
"""

from __future__ import annotations

from .inference import extract_trajectory_observations, sample_trajectory_batch

__all__ = [
    "extract_trajectory_observations",
    "sample_trajectory_batch",
]
