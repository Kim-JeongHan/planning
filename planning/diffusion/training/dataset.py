"""Dataset utilities for diffusion training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch  # type: ignore

from ..config import TrajectoryConfig
from ..core import PlannerStateNormalizer


class TrajectoryDataSetSource:
    """Load trajectory datasets from NPZ inputs."""

    def __init__(self, config: TrajectoryConfig) -> None:
        self.config = config

    def _device(self) -> torch.device:
        return torch.device(self.config.device)

    def load(self) -> list[torch.Tensor]:
        source = Path(self.config.dataset)
        device = self._device()
        if not source.exists():
            raise FileNotFoundError(f"Dataset not found: {source}")

        if source.is_file():
            if source.suffix == ".npz":
                return _load_npz(source, device=device)
            raise ValueError(f"Unsupported dataset extension: {source.suffix}. Use .npz")

        sequences: list[torch.Tensor] = []
        for candidate in sorted(source.glob("*.npz")):
            sequences.extend(_load_npz(candidate, device=device))
        if not sequences:
            raise ValueError(f"No dataset files found under {source}")
        return sequences

    def validate(self, trajectory: torch.Tensor | np.ndarray) -> bool:
        if trajectory.ndim != 2:
            return False
        if trajectory.shape[1] != self.config.state_dim:
            return False
        if self.config.horizon is None:
            return True
        return trajectory.shape[0] >= self.config.horizon

    def to_trajectories(
        self, trajectories: list[torch.Tensor] | list[np.ndarray] | None = None
    ) -> torch.Tensor:
        if trajectories is None:
            trajectories = self.load()
        device = self._device()

        if self.config.horizon is None:
            raise ValueError("horizon must be provided to convert trajectories into fixed arrays.")

        if not trajectories:
            raise ValueError("No trajectories provided.")

        trimmed: list[torch.Tensor] = []
        for trajectory in trajectories:
            if not self.validate(trajectory):
                continue
            trimmed.append(
                torch.as_tensor(
                    trajectory[: self.config.horizon],
                    dtype=torch.float32,
                    device=device,
                )
            )

        if not trimmed:
            raise ValueError(
                "No valid trajectory with state_dim="
                f"{self.config.state_dim} and horizon>={self.config.horizon}"
            )
        return torch.stack(trimmed, dim=0)


class ConditionTensorBuilder:
    """Create condition tensors from trajectory arrays as concatenated `[start, goal]`."""

    @staticmethod
    def build(trajectories: torch.Tensor) -> torch.Tensor:
        if trajectories.ndim != 3:
            raise ValueError(f"Expected 3-D trajectories, got {trajectories.ndim} dims")
        start = trajectories[:, 0]
        goal = trajectories[:, -1]
        return torch.cat([start, goal], dim=-1)


class TorchTensorFactory:
    """Convert preprocessed trajectories into torch tensors."""

    def __init__(
        self, normalizer: PlannerStateNormalizer, *, device: str | torch.device | None = None
    ) -> None:
        self.normalizer = normalizer
        self.device = device

    def _normalize_condition(self, trajectories: torch.Tensor) -> torch.Tensor:
        """Normalize start/goal condition blocks with the same normalizer as observations."""
        conditions = ConditionTensorBuilder.build(trajectories)

        state_dim = int(self.normalizer.mean.shape[0])
        if conditions.shape[1] == state_dim:
            return self.normalizer.normalize_tensor(conditions)
        if conditions.shape[1] == 2 * state_dim:
            start = self.normalizer.normalize_tensor(conditions[:, :state_dim])
            goal = self.normalizer.normalize_tensor(conditions[:, state_dim:])
            return torch.cat([start, goal], dim=-1)
        if conditions.shape[1] % state_dim == 0:
            flat = conditions.reshape(-1, state_dim)
            return self.normalizer.normalize_tensor(flat).reshape(conditions.shape[0], -1)
        return conditions

    def to_torch_tensors(self, trajectories: torch.Tensor | np.ndarray) -> tuple[object, object]:
        """Convert arrays to torch tensors for training."""
        if self.device is None:
            if torch.is_tensor(trajectories):
                trajectories_t = trajectories.to(torch.float32)
            else:
                trajectories_t = torch.as_tensor(trajectories, dtype=torch.float32)
        else:
            trajectories_t = torch.as_tensor(trajectories, dtype=torch.float32, device=self.device)
        normalized = self.normalizer.normalize_tensor(trajectories_t)
        cond = self._normalize_condition(trajectories_t)
        obs_t = normalized.to(torch.float32)
        cond_t = cond.to(torch.float32)
        return obs_t, cond_t


def _load_npz(path: Path, *, device: torch.device) -> list[torch.Tensor]:
    payload = np.load(path, allow_pickle=True)
    if "observations" not in payload:
        raise ValueError(f"NPZ file must contain key 'observations': {path}")

    observations = payload["observations"]
    if observations.ndim != 3:
        raise ValueError(f"observations must be 3-D in {path}, got {observations.ndim}")
    return [torch.as_tensor(item, dtype=torch.float32, device=device) for item in observations]
