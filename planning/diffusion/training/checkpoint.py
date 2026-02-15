"""Checkpoint persistence helpers."""

from __future__ import annotations

from pathlib import Path

import torch  # type: ignore

from ..core import PlannerStateNormalizer


def save_checkpoint(
    path: str | Path,
    *,
    model: object,
    normalizer: PlannerStateNormalizer,
    meta: dict[str, object],
    model_kind: str,
) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, object] = {
        "model_class_path": getattr(model, "hparams", {}).get(
            "model_class_path", model.__class__.__module__ + "." + model.__class__.__name__
        ),
        "model_kwargs": {
            "state_dim": int(meta.get("state_dim", getattr(model, "state_dim", 3))),
            "horizon": int(meta.get("horizon", getattr(model, "horizon", 3))),
            "n_diffusion_steps": int(meta.get("n_diffusion_steps", getattr(model, "n_diffusion_steps", 100))),
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
    payload["meta"]["kind"] = model_kind
    torch.save(payload, file_path)


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


def build_checkpoint_path(
    root: str | Path,
    *,
    dataset: str,
    kind: str,
    horizon: int,
    n_diffusion_steps: int,
    epoch: int,
    discount: float | None = None,
) -> Path:
    prefix = build_checkpoint_prefix(
        kind,
        dataset=dataset,
        horizon=horizon,
        n_diffusion_steps=n_diffusion_steps,
        discount=discount,
    )
    file_name = f"epoch_{int(epoch):04d}.ckpt"
    root_path = Path(root) / dataset / prefix
    return root_path / file_name


def build_latest_checkpoint_path(
    root: str | Path,
    *,
    dataset: str,
    kind: str,
    horizon: int,
    n_diffusion_steps: int,
    discount: float | None = None,
) -> Path:
    """Return a fixed filename that is overwritten every epoch."""
    prefix = build_checkpoint_prefix(
        kind,
        dataset=dataset,
        horizon=horizon,
        n_diffusion_steps=n_diffusion_steps,
        discount=discount,
    )
    root_path = Path(root) / dataset / prefix
    return root_path / "latest.ckpt"


def build_best_checkpoint_path(
    root: str | Path,
    *,
    dataset: str,
    kind: str,
    horizon: int,
    n_diffusion_steps: int,
    discount: float | None = None,
) -> Path:
    """Return the best-checkpoint path for the current run."""
    prefix = build_checkpoint_prefix(
        kind,
        dataset=dataset,
        horizon=horizon,
        n_diffusion_steps=n_diffusion_steps,
        discount=discount,
    )
    root_path = Path(root) / dataset / prefix
    return root_path / "best.ckpt"
