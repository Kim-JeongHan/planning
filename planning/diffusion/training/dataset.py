"""Dataset utilities for diffusion training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


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


def load_trajectory_sequences(path: str | Path) -> list[np.ndarray]:
    """Load trajectory arrays from npz/jsonl files."""
    source = Path(path)
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


def normalize_sequences(trajectories: list[np.ndarray], horizon: int, state_dim: int) -> np.ndarray:
    if not trajectories:
        raise ValueError("No trajectories provided.")
    trimmed = []
    for trajectory in trajectories:
        if trajectory.ndim != 2:
            continue
        if trajectory.shape[1] != state_dim:
            continue
        if trajectory.shape[0] < horizon:
            continue
        start = trajectory[:horizon]
        trimmed.append(start.astype(float))
    if not trimmed:
        raise ValueError(
            f"No valid trajectory with state_dim={state_dim} and horizon>={horizon}"
        )
    return np.asarray(trimmed, dtype=float)


def make_condition_tensor(trajectories: np.ndarray) -> np.ndarray:
    if trajectories.ndim != 3:
        raise ValueError(f"Expected 3-D trajectories, got {trajectories.ndim} dims")
    start = trajectories[:, 0]
    goal = trajectories[:, -1]
    return np.concatenate([start, goal], axis=-1)


def to_torch_tensors(
    trajectories: np.ndarray, normalizer: object, device: str = "cpu"
) -> tuple[object, object]:
    """Convert arrays to torch tensors for training."""
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise ImportError("torch is required for training dataset conversion.") from exc

    normalized = normalizer.normalize(trajectories)
    cond = make_condition_tensor(trajectories)
    obs_t = torch.from_numpy(normalized).to(torch.float32).to(device)
    cond_t = torch.from_numpy(cond).to(torch.float32).to(device)
    return obs_t, cond_t
