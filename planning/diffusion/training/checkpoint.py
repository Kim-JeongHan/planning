"""Checkpoint persistence helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch  # type: ignore

from ..core import PlannerStateNormalizer


@dataclass(frozen=True)
class CheckpointConfig:
    """Shared checkpoint configuration."""

    dataset: str
    horizon: int
    n_diffusion_steps: int
    root: str | Path = "logs"
    discount: float | None = None


class CheckpointPathManager:
    """Resolve checkpoint paths for logging and loading."""

    def __init__(self, config: CheckpointConfig) -> None:
        self.config = config

    @staticmethod
    def build_checkpoint_prefix(
        kind: str,
        *,
        dataset: str,
        horizon: int,
        n_diffusion_steps: int,
        discount: float | None = None,
    ) -> str:
        if kind == "diffusion":
            return f"diffusion/defaults_H{int(horizon)}_T{int(n_diffusion_steps)}"
        if discount is None:
            discount = 1.0
        return f"values/defaults_H{int(horizon)}_T{int(n_diffusion_steps)}_d{float(discount)}"

    def checkpoint_root(self, kind: str) -> Path:
        prefix = self.build_checkpoint_prefix(
            kind,
            dataset=self.config.dataset,
            horizon=self.config.horizon,
            n_diffusion_steps=self.config.n_diffusion_steps,
            discount=self.config.discount,
        )
        return Path(self.config.root) / self.config.dataset / prefix

    def checkpoint_path(self, kind: str, epoch: int) -> Path:
        return self.checkpoint_root(kind) / f"epoch_{int(epoch):04d}.ckpt"

    def latest(self, kind: str) -> Path:
        return self.checkpoint_root(kind) / "latest.ckpt"

    def best(self, kind: str) -> Path:
        return self.checkpoint_root(kind) / "best.ckpt"


class CheckpointWriter:
    """Persist model, normalizer, and metadata to disk."""

    def __init__(self, path_manager: CheckpointPathManager) -> None:
        self.path_manager = path_manager

    def _build_payload(
        self,
        model: object,
        normalizer: PlannerStateNormalizer,
        meta: dict[str, object],
        model_kind: str,
    ) -> dict[str, object]:
        return {
            "model_class_path": getattr(model, "hparams", {}).get(
                "model_class_path", model.__class__.__module__ + "." + model.__class__.__name__
            ),
            "model_kwargs": {
                "state_dim": int(meta.get("state_dim", getattr(model, "state_dim", 3))),
                "horizon": int(meta.get("horizon", getattr(model, "horizon", 3))),
                "n_diffusion_steps": int(
                    meta.get("n_diffusion_steps", getattr(model, "n_diffusion_steps", 100))
                ),
                "n_hidden": int(meta.get("n_hidden", 256)),
                "n_layers": int(meta.get("n_layers", 2)),
                "condition_dim": int(meta.get("condition_dim", 0)),
            },
            "normalizer": normalizer.to_dict(),
            "meta": dict(meta),
            "model_state_dict": model.state_dict(),
            "ema_state_dict": model.state_dict(),
            "kind": model_kind,
        }

    def save(
        self,
        path: str | Path,
        *,
        model: object,
        normalizer: PlannerStateNormalizer,
        meta: dict[str, object],
        model_kind: str,
    ) -> Path:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        payload = self._build_payload(model, normalizer, meta, model_kind)
        payload["meta"]["kind"] = model_kind
        torch.save(payload, file_path)
        return file_path
