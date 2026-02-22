"""Dataset utilities for diffusion training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch  # type: ignore

from ..core import PlannerStateNormalizer


@dataclass(frozen=True)
class TrajectoryLoadConfig:
    """Configuration shared by dataset loading and preprocessing steps."""

    path: str | Path
    horizon: int | None = None
    state_dim: int = 3
    seed: int | None = None

    def resolved_path(self) -> Path:
        return Path(self.path)


class TrajectoryDataSetSource:
    """Load trajectory datasets from npz/jsonl inputs."""

    def __init__(self, config: TrajectoryLoadConfig) -> None:
        self.config = config

    def load(self) -> list[np.ndarray]:
        source = self.config.resolved_path()
        if not source.exists():
            raise FileNotFoundError(f"Dataset not found: {source}")

        if source.is_file():
            if source.suffix == ".npz":
                return _load_npz(source)
            if source.suffix == ".jsonl":
                return _load_jsonl(source)
            raise ValueError(f"Unsupported dataset extension: {source.suffix}")

        sequences: list[np.ndarray] = []
        for candidate in sorted(source.glob("*.npz")):
            sequences.extend(_load_npz(candidate))
        for candidate in sorted(source.glob("*.jsonl")):
            sequences.extend(_load_jsonl(candidate))
        if not sequences:
            raise ValueError(f"No dataset files found under {source}")
        return sequences

    def validate(self, trajectory: np.ndarray) -> bool:
        if trajectory.ndim != 2:
            return False
        if trajectory.shape[1] != self.config.state_dim:
            return False
        if self.config.horizon is None:
            return True
        return trajectory.shape[0] >= self.config.horizon

    def to_normalized_numpy(self, trajectories: list[np.ndarray] | None = None) -> np.ndarray:
        if trajectories is None:
            trajectories = self.load()

        if self.config.horizon is None:
            raise ValueError("horizon must be provided to convert trajectories into fixed arrays.")

        if not trajectories:
            raise ValueError("No trajectories provided.")

        trimmed = []
        for trajectory in trajectories:
            if not self.validate(trajectory):
                continue
            trimmed.append(trajectory[: self.config.horizon].astype(float))

        if not trimmed:
            raise ValueError(
                "No valid trajectory with state_dim="
                f"{self.config.state_dim} and horizon>={self.config.horizon}"
            )
        return np.asarray(trimmed, dtype=float)


class ConditionTensorBuilder:
    """Create condition tensors from trajectory arrays as concatenated `[start, goal]`."""

    @staticmethod
    def build(trajectories: np.ndarray) -> np.ndarray:
        if trajectories.ndim != 3:
            raise ValueError(f"Expected 3-D trajectories, got {trajectories.ndim} dims")
        start = trajectories[:, 0]
        goal = trajectories[:, -1]
        return np.concatenate([start, goal], axis=-1)


class TorchTensorFactory:
    """Convert preprocessed trajectories into torch tensors."""

    def __init__(self, normalizer: PlannerStateNormalizer, *, device: str = "cpu") -> None:
        self.normalizer = normalizer
        self.device = device

    def _normalize_condition(self, trajectories: np.ndarray) -> np.ndarray:
        """Normalize start/goal condition blocks with the same normalizer as observations."""
        conditions = ConditionTensorBuilder.build(trajectories)
        if conditions.ndim != 2:
            return conditions

        state_dim = int(self.normalizer.mean.shape[0])
        if conditions.shape[1] == state_dim:
            return self.normalizer.normalize(conditions)
        if conditions.shape[1] == 2 * state_dim:
            start = self.normalizer.normalize(conditions[:, :state_dim])
            goal = self.normalizer.normalize(conditions[:, state_dim:])
            return np.concatenate([start, goal], axis=-1)
        if conditions.shape[1] % state_dim == 0:
            flat = conditions.reshape(-1, state_dim)
            return self.normalizer.normalize(flat).reshape(conditions.shape[0], -1)
        return conditions

    def to_torch_tensors(self, trajectories: np.ndarray) -> tuple[object, object]:
        """Convert arrays to torch tensors for training."""
        normalized = self.normalizer.normalize(trajectories)
        cond = self._normalize_condition(trajectories)
        obs_t = torch.from_numpy(normalized).to(torch.float32).to(self.device)
        cond_t = torch.from_numpy(cond).to(torch.float32).to(self.device)
        return obs_t, cond_t


def _load_npz(path: Path) -> list[np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    if "observations" not in payload:
        raise ValueError(f"NPZ file must contain key 'observations': {path}")

    observations = payload["observations"]
    if observations.ndim != 3:
        raise ValueError(f"observations must be 3-D in {path}, got {observations.ndim}")
    return [np.asarray(item, dtype=float) for item in observations]


def _load_jsonl(path: Path) -> list[np.ndarray]:
    trajectories: list[np.ndarray] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            record = json.loads(raw)
            if "observations" not in record:
                continue
            trajectory = np.asarray(record["observations"], dtype=float)
            if trajectory.ndim != 2:
                raise ValueError("Each jsonl trajectory must be a 2D array [H, D]")
            trajectories.append(trajectory)
    return trajectories
