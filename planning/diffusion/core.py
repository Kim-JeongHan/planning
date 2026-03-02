"""Core data models shared across local diffuser utilities."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch  # type: ignore


def _as_float_vector(
    values: torch.Tensor | np.ndarray | list[float] | tuple[float, ...],
    *,
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be a 1-D vector, got shape {tuple(tensor.shape)!r}")
    return tensor


@dataclass(frozen=True)
class PlannerStateNormalizer:
    """Simple state normalizer with mean/std statistics."""

    mean: torch.Tensor
    std: torch.Tensor
    clip_min: float = -5.0
    clip_max: float = 5.0

    @classmethod
    def fit(cls, trajectories: torch.Tensor | np.ndarray) -> PlannerStateNormalizer:
        trajectories_t = torch.as_tensor(trajectories, dtype=torch.float32)
        if trajectories_t.ndim != 3:
            raise ValueError(
                f"trajectories must be 3-D [N, T, D], got {trajectories_t.ndim}-D array"
            )
        flat = trajectories_t.reshape(-1, trajectories_t.shape[-1])
        mean = flat.mean(dim=0)
        std = flat.std(dim=0, unbiased=False)
        std = torch.where(std <= 1e-8, torch.ones_like(std), std)
        return cls(mean=mean.cpu(), std=std.cpu())

    @classmethod
    def identity(cls, dim: int) -> PlannerStateNormalizer:
        return cls(
            mean=torch.zeros(int(dim), dtype=torch.float32),
            std=torch.ones(int(dim), dtype=torch.float32),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, object] | None) -> PlannerStateNormalizer:
        if not payload:
            raise ValueError("normalizer payload must be a non-empty dict")
        mean = _as_float_vector(payload["mean"], name="normalizer.mean")
        std = _as_float_vector(payload["std"], name="normalizer.std")
        clip_min = float(payload.get("clip_min", -5.0))
        clip_max = float(payload.get("clip_max", 5.0))
        return cls(mean=mean, std=std, clip_min=clip_min, clip_max=clip_max)

    def to_dict(self) -> dict[str, object]:
        return {
            "mean": self.mean.cpu().tolist(),
            "std": self.std.cpu().tolist(),
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
        }

    def _stats_for(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean.to(device=values.device, dtype=values.dtype)
        std = self.std.to(device=values.device, dtype=values.dtype)
        return mean, std

    def normalize_tensor(self, values: torch.Tensor) -> torch.Tensor:
        values_t = values.to(torch.float32)
        mean, std = self._stats_for(values_t)
        normalized = (values_t - mean) / std
        return torch.clamp(normalized, min=self.clip_min, max=self.clip_max)

    def denormalize_tensor(self, values: torch.Tensor) -> torch.Tensor:
        values_t = values.to(torch.float32)
        mean, std = self._stats_for(values_t)
        return values_t * std + mean

    def normalize(self, values: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        if torch.is_tensor(values):
            return self.normalize_tensor(values)
        values_t = torch.as_tensor(values, dtype=torch.float32)
        return self.normalize_tensor(values_t).cpu().numpy()

    def denormalize(self, values: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        if torch.is_tensor(values):
            return self.denormalize_tensor(values)
        values_t = torch.as_tensor(values, dtype=torch.float32)
        return self.denormalize_tensor(values_t).cpu().numpy()


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
