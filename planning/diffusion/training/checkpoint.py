"""Checkpoint persistence helpers."""

from __future__ import annotations

from pathlib import Path

import torch  # type: ignore

from ..config import CheckpointConfig
from ..core import PlannerStateNormalizer


class CheckpointPathManager:
    """Resolve checkpoint paths for logging and loading."""

    def __init__(self, config: CheckpointConfig) -> None:
        self.config = config

    def checkpoint_root(self, kind: str) -> Path:
        horizon = int(self.config.horizon)
        n_steps = int(self.config.n_diffusion_steps)
        if kind == "diffusion":
            prefix = f"diffusion/defaults_H{horizon}_T{n_steps}"
        else:
            discount = 1.0 if self.config.discount is None else float(self.config.discount)
            prefix = f"values/defaults_H{horizon}_T{n_steps}_d{discount}"
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
        ema_state_dict: dict[str, object] | None = None,
    ) -> dict[str, object]:
        hparams = getattr(model, "hparams", {})
        model_state = model.state_dict()  # type: ignore[union-attr]
        return {
            "model_class_path": hparams.get(
                "model_class_path", model.__class__.__module__ + "." + model.__class__.__name__
            ),
            "model_kwargs": {
                "state_dim": int(meta.get("state_dim", getattr(model, "state_dim", 3))),
                "horizon": int(meta.get("horizon", getattr(model, "horizon", 3))),
                "n_diffusion_steps": int(
                    meta.get("n_diffusion_steps", getattr(model, "n_diffusion_steps", 100))
                ),
                "dim": int(hparams.get("dim", getattr(model, "dim", 32))),
                "dim_mults": list(hparams.get("dim_mults", getattr(model, "dim_mults", [1, 2, 4, 8]))),
            },
            "normalizer": normalizer.to_dict(),
            "meta": dict(meta),
            "model_state_dict": model_state,
            # EMA weights are used for inference; fall back to model weights if not provided.
            "ema_state_dict": ema_state_dict if ema_state_dict is not None else model_state,
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
        ema_state_dict: dict[str, object] | None = None,
    ) -> Path:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        payload = self._build_payload(model, normalizer, meta, model_kind, ema_state_dict)
        payload["meta"]["kind"] = model_kind
        torch.save(payload, file_path)
        return file_path
