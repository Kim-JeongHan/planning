"""Checkpoint persistence helpers."""

from __future__ import annotations

import json
import re
from math import isfinite
from pathlib import Path

import torch  # type: ignore

from ..config import CheckpointConfig
from ..core import PlannerStateNormalizer

_CHECKPOINT_FILE_RE = re.compile(r"epoch_(\d+)\.ckpt")
_CKPT_FILE_RE = re.compile(r"ckpt_(\d+)\.pt")
_LATEST_CKPT_FILE_RE = re.compile(r"latest\.ckpt")
_BEST_CKPT_FILE_RE = re.compile(r"best\.ckpt")


class CheckpointManager:
    """Resolve checkpoint paths and persist payloads."""

    def __init__(self, config: CheckpointConfig) -> None:
        self.config = config
        self.writer = CheckpointWriter(self)
        self.loader = CheckpointLoader(self)

    @classmethod
    def for_loading(cls, checkpoint_path: str, device: str = "cpu") -> CheckpointManager:
        return cls(
            CheckpointConfig(
                horizon=1,
                n_diffusion_steps=1,
                root=".",
                checkpoint_path=checkpoint_path,
                device=device,
            ),
        )

    def checkpoint_root(self, kind: str) -> Path:
        if kind not in {"diffusion", "value"}:
            raise ValueError("kind must be one of: diffusion, value.")
        horizon = int(self.config.horizon)
        n_steps = int(self.config.n_diffusion_steps)
        if kind == "diffusion":
            prefix = f"diffusion/defaults_H{horizon}_T{n_steps}"
        else:
            discount = 1.0 if self.config.discount is None else float(self.config.discount)
            prefix = f"values/defaults_H{horizon}_T{n_steps}_d{discount}"
        return Path(self.config.root) / prefix

    def checkpoint_path(self, kind: str, epoch: int) -> Path:
        return self.checkpoint_root(kind) / f"epoch_{int(epoch):04d}.ckpt"

    def latest(self, kind: str) -> Path:
        return self.checkpoint_root(kind) / "latest.ckpt"

    def best(self, kind: str) -> Path:
        return self.checkpoint_root(kind) / "best.ckpt"

    @property
    def root(self) -> Path:
        return self.loader.root

    def candidates(self) -> list[tuple[int, Path]]:
        return self.loader.candidates()

    def resolve(self, epoch: str | int = "latest") -> Path:
        return self.loader.resolve(epoch)

    def load(self, epoch: str | int = "latest") -> tuple[Path, dict[str, object]]:
        return self.loader.load(epoch)

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
        return self.writer.save(
            path,
            model=model,
            normalizer=normalizer,
            meta=meta,
            model_kind=model_kind,
            ema_state_dict=ema_state_dict,
        )


class CheckpointLoader:
    """Resolve and load persisted checkpoint payloads."""

    def __init__(self, manager: CheckpointManager) -> None:
        self.manager = manager
        self.checkpoint_path = str(manager.config.checkpoint_path)
        self.device = str(manager.config.device)

    def _extract_checkpoint_epoch(self, path: Path) -> int | None:
        name = path.name
        if _LATEST_CKPT_FILE_RE.fullmatch(name) is not None:
            return int(1e9)
        if _BEST_CKPT_FILE_RE.fullmatch(name) is not None:
            return int(1e9) - 1
        for pattern in (_CHECKPOINT_FILE_RE, _CKPT_FILE_RE):
            match = pattern.fullmatch(name)
            if match is not None:
                return int(match.group(1))
        return None

    def _collect_checkpoint_files(self, root: Path) -> list[tuple[int, Path]]:
        epochs: list[tuple[int, Path]] = []
        for path in root.glob("*.ckpt"):
            epoch = self._extract_checkpoint_epoch(path)
            if epoch is not None:
                epochs.append((epoch, path))
        for path in root.glob("*.pt"):
            epoch = self._extract_checkpoint_epoch(path)
            if epoch is not None:
                epochs.append((epoch, path))
        return sorted(epochs, key=lambda item: item[0])

    def _resolve_checkpoint_fallback(
        self, root: Path, candidates: list[tuple[int, Path]]
    ) -> Path | None:
        candidates_only = [path for _, path in candidates]
        if candidates_only:
            return candidates_only[-1]

        checkpoint_json = root / "checkpoint.json"
        if not checkpoint_json.exists():
            return None
        payload = json.loads(checkpoint_json.read_text(encoding="utf-8"))
        latest = payload.get("latest_checkpoint")
        if not latest:
            return None
        explicit = root / latest
        if explicit.exists():
            return explicit
        return None

    def _coerce_epoch_value(self, value: str | int) -> int | None:
        if isinstance(value, int):
            return int(value)
        normalized = value.strip().lower()
        try:
            return int(normalized)
        except ValueError:
            return None

    def _load_checkpoint_payload(self, file_path: Path) -> dict[str, object]:
        if file_path.suffix.lower() in {".pt", ".ckpt"}:
            try:
                return torch.load(file_path, map_location=self.device)
            except Exception as exc:
                raise RuntimeError(f"Failed to load torch checkpoint: {file_path}") from exc
        raise ValueError(f"Unsupported checkpoint format: {file_path.suffix}")

    def _select_best_checkpoint_by_loss(self, candidates: list[tuple[int, Path]]) -> Path | None:
        """Select checkpoint with minimum `meta.loss`; return None if unavailable."""
        best_loss = float("inf")
        best_path: Path | None = None
        for _, candidate_path in candidates:
            try:
                payload = self._load_checkpoint_payload(candidate_path)
            except (RuntimeError, ValueError):
                continue
            meta = payload.get("meta", {})
            raw_loss = meta.get("loss")
            try:
                loss_value = float(raw_loss)
            except (TypeError, ValueError):
                continue
            if not isfinite(loss_value):
                continue
            if loss_value < best_loss:
                best_loss = loss_value
                best_path = candidate_path
        if best_path is not None:
            return best_path
        if candidates:
            return candidates[-1][1]
        return None

    def _resolve_named_checkpoint(
        self, root: Path, requested: str, candidates: list[tuple[int, Path]]
    ) -> Path | None:
        if requested == "latest":
            direct_latest = root / "latest.ckpt"
            if direct_latest.exists():
                return direct_latest
            return None
        if requested == "best":
            direct_best = root / "best.ckpt"
            if direct_best.exists():
                return direct_best
            return self._select_best_checkpoint_by_loss(candidates)
        return None

    def _resolve_checkpoint_by_epoch(
        self, root: Path, target_epoch: int, candidates: list[tuple[int, Path]]
    ) -> Path | None:
        direct_ckpt = root / f"epoch_{target_epoch:04d}.ckpt"
        if direct_ckpt.exists():
            return direct_ckpt
        direct_alt = root / f"ckpt_{target_epoch:04d}.pt"
        if direct_alt.exists():
            return direct_alt
        for candidate_epoch, candidate_path in candidates:
            if candidate_epoch == target_epoch:
                return candidate_path
        return None

    @property
    def root(self) -> Path:
        root = Path(self.checkpoint_path)
        if not root.exists():
            raise FileNotFoundError(
                f"Could not resolve checkpoint root for checkpoint_path={self.checkpoint_path!r}"
            )
        return root

    def candidates(self) -> list[tuple[int, Path]]:
        if self.root.is_file():
            return []
        return self._collect_checkpoint_files(self.root)

    def resolve(self, epoch: str | int = "latest") -> Path:
        root = self.root
        if root.is_file():
            return root

        candidates = self.candidates()
        requested = ""
        if isinstance(epoch, str):
            requested = epoch.strip().lower()
            if requested:
                by_name = self._resolve_named_checkpoint(root, requested, candidates)
                if by_name is not None:
                    return by_name
                direct = root / requested
                if direct.exists():
                    return direct
        target_epoch = self._coerce_epoch_value(epoch)
        if target_epoch is not None:
            by_epoch = self._resolve_checkpoint_by_epoch(root, target_epoch, candidates)
            if by_epoch is not None:
                return by_epoch

        fallback = self._resolve_checkpoint_fallback(root, candidates)
        if fallback is not None:
            return fallback

        if not candidates:
            raise FileNotFoundError(f"No checkpoints found in {root}")
        if not requested:
            return candidates[-1][1]
        raise FileNotFoundError(
            f"Could not find epoch {epoch} in {root}; "
            f"available: {[path.name for _, path in candidates]}"
        )

    def load(self, epoch: str | int = "latest") -> tuple[Path, dict[str, object]]:
        checkpoint_file = self.resolve(epoch)
        payload = self._load_checkpoint_payload(checkpoint_file)
        return checkpoint_file, payload


class CheckpointWriter:
    """Persist model, normalizer, and metadata to disk."""

    def __init__(self, path_manager: CheckpointManager) -> None:
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
                "dim_mults": list(
                    hparams.get("dim_mults", getattr(model, "dim_mults", [1, 2, 4, 8]))
                ),
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
