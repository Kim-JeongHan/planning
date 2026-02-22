"""Core data models shared across local diffuser utilities."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np


def _as_float_array(values: np.ndarray | list[float] | tuple[float, ...], *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D vector, got shape {array.shape!r}")
    return array


@dataclass(frozen=True)
class PlannerStateNormalizer:
    """Simple state normalizer with mean/std statistics."""

    mean: np.ndarray
    std: np.ndarray
    clip_min: float = -5.0
    clip_max: float = 5.0

    @classmethod
    def fit(cls, trajectories: np.ndarray) -> PlannerStateNormalizer:
        if trajectories.ndim != 3:
            raise ValueError(
                f"trajectories must be 3-D [N, T, D], got {trajectories.ndim}-D array"
            )
        mean = np.mean(trajectories.reshape(-1, trajectories.shape[-1]), axis=0)
        std = np.std(trajectories.reshape(-1, trajectories.shape[-1]), axis=0)
        std = np.where(std <= 1e-8, 1.0, std)
        return cls(mean=mean.astype(float), std=std.astype(float))

    @classmethod
    def identity(cls, dim: int) -> PlannerStateNormalizer:
        return cls(mean=np.zeros(int(dim), dtype=float), std=np.ones(int(dim), dtype=float))

    @classmethod
    def from_dict(cls, payload: dict[str, object] | None) -> PlannerStateNormalizer:
        if not payload:
            raise ValueError("normalizer payload must be a non-empty dict")
        mean = _as_float_array(payload["mean"], name="normalizer.mean")
        std = _as_float_array(payload["std"], name="normalizer.std")
        clip_min = float(payload.get("clip_min", -5.0))
        clip_max = float(payload.get("clip_max", 5.0))
        return cls(mean=mean, std=std, clip_min=clip_min, clip_max=clip_max)

    def to_dict(self) -> dict[str, object]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
        }

    def normalize(self, values: np.ndarray) -> np.ndarray:
        normalized = (np.asarray(values, dtype=float) - self.mean) / self.std
        return np.clip(normalized, self.clip_min, self.clip_max)

    def denormalize(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.std + self.mean


@dataclass(frozen=True)
class DiffusionDataset:
    """Minimal dataset metadata object for compatibility with loaded experiments."""

    name: str
    normalizer: PlannerStateNormalizer
    horizon: int
    state_dim: int


@dataclass(frozen=True)
class DiffusionExperiment:
    """Container matching the fields expected by planning sampler."""

    ema: object
    dataset: DiffusionDataset
    meta: dict[str, object]

    @property
    def path(self) -> str | None:
        return self.meta.get("path") if isinstance(self.meta, dict) else None


def build_namespace(**kwargs: object) -> SimpleNamespace:
    """Create a namespaced result object for compatibility with legacy style code."""

    return SimpleNamespace(**kwargs)
