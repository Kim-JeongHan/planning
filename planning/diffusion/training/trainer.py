"""Training orchestrator for local diffusion/value models."""

from __future__ import annotations

import contextlib
import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import torch  # type: ignore
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from tqdm import tqdm

from ..config import DiffusionTrainingConfig
from ..core import PlannerStateNormalizer
from ..model import DiffusionModel, ValueModel
from .checkpoint import (
    CheckpointManager,
)
from .dataset import TorchTensorFactory, TrajectoryDataSetSource
from .epoch_trainer import DiffusionEpochTrainer, EMAAccumulator, ValueEpochTrainer
from .noise import DiffusionSchedule

_LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stage parameter bundle
# ---------------------------------------------------------------------------


@dataclass
class _StageParams:
    phase: str
    epoch_trainer: object
    model: object
    optimizer: object
    ema: object
    effective_epochs: int
    patience: int | None
    min_delta: float
    log_every: int
    config: object
    normalizer: PlannerStateNormalizer
    checkpoint_manager: object
    train_loader: object
    val_loader: object | None
    summary_writer: object | None
    checkpoint_every: int
    latest_checkpoint_every: int
    keep_last_checkpoints: int
    extra_meta: dict[str, object] | None = field(default=None)


class _SupportsAddText(Protocol):
    def add_text(self, tag: str, text: str, global_step: int) -> object: ...


class _SupportsEpochCalls(Protocol):
    def train_epoch(self, loader: object) -> float: ...

    def evaluate_epoch(self, loader: object) -> float: ...


class _SupportsStateDict(Protocol):
    def state_dict(self) -> dict[str, object]: ...


# ---------------------------------------------------------------------------
# Loss tracker with early-stop support
# ---------------------------------------------------------------------------


class EpochLossTracker:
    """Track running and best loss statistics with early-stop counters."""

    def __init__(self, *, min_delta: float = 0.0, patience: int | None = None) -> None:
        self.min_delta = float(min_delta)
        self.patience = patience
        self.best_loss: float | None = None
        self.best_epoch: int | None = None
        self.prev_loss: float | None = None
        self.no_improve = 0

    def update(self, loss: float, epoch: int) -> tuple[bool, str]:
        is_new_best = False
        if self.best_loss is None or loss < (self.best_loss - self.min_delta):
            self.best_loss = loss
            self.best_epoch = epoch
            self.no_improve = 0
            is_new_best = True
        else:
            self.no_improve += 1
        delta = _format_loss_delta(loss, self.prev_loss)
        self.prev_loss = loss
        return is_new_best, delta

    def should_stop(self) -> bool:
        if self.patience is None:
            return False
        return self.no_improve >= self.patience


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


class DiffusionTrainingPipeline:
    """Orchestrator for diffusion/value training workflow."""

    def __init__(self, **values: object) -> None:
        self.cfg = DiffusionTrainingConfig(**values)

    def run(self) -> list[Path]:
        cfg = self.cfg

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
            np.random.seed(cfg.seed)

        device = torch.device(cfg.device)
        _LOGGER.info("Training device: %s", device)

        writer_dir = Path(cfg.output_path) / "tensorboard"
        writer_dir.mkdir(parents=True, exist_ok=True)
        summary_writer = SummaryWriter(log_dir=str(writer_dir))

        try:
            trajectories, normalizer = self._load_and_prepare_dataset()
            config = self.cfg
            train_loader, val_loader = self._build_data_loaders(trajectories, normalizer, config)
            checkpoint_manager = self._build_checkpoint_manager(config)

            if summary_writer is not None:
                _log_config_summary(summary_writer, config)

            all_ckpts: list[Path] = []

            if cfg.train_diffusion:
                all_ckpts += self._run_diffusion_stage(
                    config=config,
                    normalizer=normalizer,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    checkpoint_manager=checkpoint_manager,
                    summary_writer=summary_writer,
                    training_device=device,
                )

            if cfg.train_value:
                all_ckpts += self._run_value_stage(
                    config=config,
                    normalizer=normalizer,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    checkpoint_manager=checkpoint_manager,
                    summary_writer=summary_writer,
                    training_device=device,
                )
        finally:
            if summary_writer is not None:
                summary_writer.close()

        return [path for path in all_ckpts if path.exists()]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_and_prepare_dataset(
        self,
    ) -> tuple[torch.Tensor, PlannerStateNormalizer]:
        """Load raw sequences, validate configured horizon, and fit normalizer."""
        cfg = self.cfg

        trajectory_config = cfg.to_trajectory_config()
        horizon = cfg.horizon

        source = TrajectoryDataSetSource(trajectory_config)
        trajectories = source.to_trajectories()

        _LOGGER.info(
            "Loaded %d trajectories (horizon=%d, state_dim=%d).",
            trajectories.shape[0],
            horizon,
            cfg.state_dim,
        )
        normalizer = PlannerStateNormalizer.fit(trajectories)
        return trajectories, normalizer

    def _build_data_loaders(
        self,
        trajectories: torch.Tensor,
        normalizer: PlannerStateNormalizer,
        config: DiffusionTrainingConfig,
    ) -> tuple[object, object | None]:
        tensor_factory = TorchTensorFactory(normalizer)
        obs_t, cond_t = tensor_factory.to_torch_tensors(trajectories)
        dataset_tensors = torch.utils.data.TensorDataset(obs_t, cond_t)
        validation_split = float(config.validation_split)
        if validation_split <= 0.0:
            return (
                torch.utils.data.DataLoader(
                    dataset_tensors, batch_size=config.batch_size, shuffle=True
                ),
                None,
            )

        dataset_size = len(dataset_tensors)
        num_val = int(dataset_size * validation_split)
        if num_val < 1 or num_val >= dataset_size:
            return (
                torch.utils.data.DataLoader(
                    dataset_tensors, batch_size=config.batch_size, shuffle=True
                ),
                None,
            )

        generator = (
            torch.Generator().manual_seed(self.cfg.seed) if self.cfg.seed is not None else None
        )
        train_subset, val_subset = torch.utils.data.random_split(
            dataset_tensors,
            [dataset_size - num_val, num_val],
            generator=generator,
        )
        return (
            torch.utils.data.DataLoader(train_subset, batch_size=config.batch_size, shuffle=True),
            torch.utils.data.DataLoader(val_subset, batch_size=config.batch_size, shuffle=False),
        )

    def _build_checkpoint_manager(self, config: DiffusionTrainingConfig) -> CheckpointManager:
        checkpoint_config = config.to_checkpoint_config()
        return CheckpointManager(checkpoint_config)

    def _build_common_stage_params(
        self,
        *,
        phase: str,
        model: object,
        epoch_trainer: object,
        optimizer: object,
        ema: EMAAccumulator,
        effective_epochs: int,
        patience: int | None,
        min_delta: float,
        config: DiffusionTrainingConfig,
        normalizer: PlannerStateNormalizer,
        train_loader: object,
        val_loader: object | None,
        checkpoint_manager: CheckpointManager,
        summary_writer: object | None,
        extra_meta: dict[str, object] | None = None,
    ) -> _StageParams:
        return _StageParams(
            phase=phase,
            epoch_trainer=epoch_trainer,
            model=model,
            optimizer=optimizer,
            ema=ema,
            effective_epochs=effective_epochs,
            patience=patience,
            min_delta=min_delta,
            log_every=_coerce_log_every(effective_epochs, self.cfg.log_every),
            config=config,
            normalizer=normalizer,
            checkpoint_manager=checkpoint_manager,
            train_loader=train_loader,
            val_loader=val_loader,
            summary_writer=summary_writer,
            checkpoint_every=config.checkpoint_every,
            latest_checkpoint_every=config.latest_checkpoint_every,
            keep_last_checkpoints=config.keep_last_checkpoints,
            extra_meta=extra_meta,
        )

    def _run_diffusion_stage(
        self,
        *,
        config: DiffusionTrainingConfig,
        normalizer: PlannerStateNormalizer,
        train_loader: object,
        val_loader: object | None,
        checkpoint_manager: CheckpointManager,
        summary_writer: object | None,
        training_device: torch.device,
    ) -> list[Path]:
        effective_epochs = config.epochs
        if config.diffusion_max_epochs is not None:
            effective_epochs = min(effective_epochs, config.diffusion_max_epochs)
        if effective_epochs <= 0:
            raise ValueError("diffusion_max_epochs / epochs must be at least 1.")

        model = DiffusionModel(
            state_dim=config.state_dim,
            horizon=config.horizon,
            n_diffusion_steps=config.n_diffusion_steps,
            dim=config.n_hidden,
        ).to(training_device)
        schedule = DiffusionSchedule.cosine(n_diffusion_steps=config.n_diffusion_steps)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        ema = EMAAccumulator(model)
        epoch_trainer = DiffusionEpochTrainer(
            model=model, optimizer=optimizer, schedule=schedule, ema=ema
        )
        patience = config.diffusion_patience
        min_delta = float(config.diffusion_min_delta)
        _LOGGER.info(
            "LR schedule: %s (lr=%.4g, step_size=%d, gamma=%.3g, lr_min=%.4g)",
            config.lr_schedule,
            config.learning_rate,
            config.lr_step_size,
            config.lr_gamma,
            config.lr_min,
        )
        params = self._build_common_stage_params(
            phase="diffusion",
            model=model,
            epoch_trainer=epoch_trainer,
            optimizer=optimizer,
            ema=ema,
            effective_epochs=effective_epochs,
            patience=patience,
            min_delta=min_delta,
            config=config,
            normalizer=normalizer,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_manager=checkpoint_manager,
            summary_writer=summary_writer,
        )
        return self._run_stage(params)

    def _run_value_stage(
        self,
        *,
        config: DiffusionTrainingConfig,
        normalizer: PlannerStateNormalizer,
        train_loader: object,
        val_loader: object | None,
        checkpoint_manager: CheckpointManager,
        summary_writer: object | None,
        training_device: torch.device,
    ) -> list[Path]:
        effective_epochs = config.epochs
        if config.value_max_epochs is not None:
            if config.value_max_epochs < 0:
                raise ValueError("value_max_epochs must be >= 0.")
            effective_epochs = min(effective_epochs, config.value_max_epochs)
        if effective_epochs < 1:
            return []

        model = ValueModel(
            state_dim=config.state_dim,
            horizon=config.horizon,
            dim=config.n_hidden,
        ).to(training_device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        ema = EMAAccumulator(model)
        epoch_trainer = ValueEpochTrainer(
            model=model, optimizer=optimizer, normalizer=normalizer, ema=ema
        )
        patience = config.value_patience
        min_delta = float(config.value_min_delta)
        params = self._build_common_stage_params(
            phase="value",
            model=model,
            epoch_trainer=epoch_trainer,
            optimizer=optimizer,
            ema=ema,
            effective_epochs=effective_epochs,
            patience=patience,
            min_delta=min_delta,
            config=config,
            normalizer=normalizer,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_manager=checkpoint_manager,
            summary_writer=summary_writer,
            extra_meta={"discount": config.discount},
        )
        return self._run_stage(params)

    def _run_stage(self, p: _StageParams) -> list[Path]:
        config = cast(DiffusionTrainingConfig, p.config)
        checkpoint_manager = cast(CheckpointManager, p.checkpoint_manager)
        ckpts: list[Path] = []
        tracker = EpochLossTracker(min_delta=p.min_delta, patience=p.patience)
        recent_ckpts: deque[Path] = deque()
        latest_ckpt = checkpoint_manager.latest(p.phase)
        best_ckpt = checkpoint_manager.best(p.phase)
        bar = tqdm(
            range(1, p.effective_epochs + 1),
            desc=f"{p.phase.title()} training",
            unit="epoch",
        )

        for epoch in bar:
            current_lr = _learning_rate_for_epoch(
                config.learning_rate,
                config.lr_schedule,
                epoch,
                total_epochs=p.effective_epochs,
                step_size=config.lr_step_size,
                gamma=config.lr_gamma,
                lr_min=config.lr_min,
            )
            _set_learning_rate(p.optimizer, current_lr)
            train_loss, val_loss, metric_loss = _run_epoch_with_validation(
                p.epoch_trainer, p.train_loader, p.val_loader
            )
            is_new_best, delta = tracker.update(metric_loss, epoch)
            bar.set_postfix(
                {
                    "train": f"{train_loss:.4f}",
                    "val": "n/a" if val_loss is None else f"{val_loss:.4f}",
                    "best": f"{tracker.best_loss:.4f}",  # always set after first update
                    "Δ": delta,
                    "lr": _format_lr(current_lr),
                }
            )
            if epoch % p.log_every == 0 or epoch == 1 or epoch == p.effective_epochs:
                best_label = (
                    ""
                    if tracker.best_loss is None
                    else f" best={tracker.best_loss:.6f} (epoch={tracker.best_epoch})"
                )
                _LOGGER.info(
                    "[%s] epoch %d/%d  train=%.6f  val=%s  lr=%.6f  Δ=%s%s%s",
                    p.phase,
                    epoch,
                    p.effective_epochs,
                    train_loss,
                    "n/a" if val_loss is None else f"{val_loss:.6f}",
                    current_lr,
                    delta,
                    best_label,
                    " [new best]" if is_new_best else "",
                )
            _log_scalar(p.summary_writer, f"{p.phase}/loss", train_loss, epoch)
            if val_loss is not None:
                _log_scalar(p.summary_writer, f"{p.phase}/val_loss", val_loss, epoch)
            _log_scalar(p.summary_writer, f"{p.phase}/lr", current_lr, epoch)
            if tracker.best_loss is not None:
                _log_scalar(p.summary_writer, f"{p.phase}/best_loss", tracker.best_loss, epoch)

            _manage_stage_checkpoints(
                p,
                epoch=epoch,
                metric_loss=metric_loss,
                train_loss=train_loss,
                is_new_best=is_new_best,
                recent_ckpts=recent_ckpts,
                ckpt_paths=ckpts,
                latest_ckpt=latest_ckpt,
                best_ckpt=best_ckpt,
            )

            if tracker.should_stop():
                _LOGGER.info(
                    "[early-stop][%s] Stopping at epoch %d/%d " "(no improvement for %d epochs).",
                    p.phase,
                    epoch,
                    p.effective_epochs,
                    tracker.no_improve,
                )
                break

        _LOGGER.info(
            "[%s] Done. best_epoch=%s  best_loss=%.6f",
            p.phase,
            "n/a" if tracker.best_epoch is None else tracker.best_epoch,
            tracker.best_loss or 0.0,
        )
        bar.close()
        return [path for path in ckpts if path.exists()]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _log_every(epochs: int, target_ticks: int = 20) -> int:
    if epochs <= 1:
        return 1
    return max(1, min(epochs // target_ticks, 100))


def _coerce_log_every(epochs: int, log_every: int | None) -> int:
    if log_every is None or log_every <= 0:
        return _log_every(epochs)
    return max(1, int(log_every))


def _log_scalar(writer: object | None, tag: str, value: float, step: int) -> None:
    if writer is None or not hasattr(writer, "add_scalar"):
        return
    add_scalar = cast(Callable[[str, float, int], object], writer.add_scalar)
    add_scalar(tag, float(value), step)


def _log_config_summary(writer: object, config: DiffusionTrainingConfig) -> None:
    writer_with_text = cast(_SupportsAddText, writer)
    writer_with_text.add_text(
        "training/config",
        (
            f"dataset={config.dataset}\n"
            f"horizon={config.horizon}\n"
            f"state_dim={config.state_dim}\n"
            f"n_diffusion_steps={config.n_diffusion_steps}\n"
            f"batch_size={config.batch_size}\n"
            f"learning_rate={config.learning_rate}\n"
            f"lr_schedule={config.lr_schedule}\n"
            f"epochs={config.epochs}\n"
        ),
        0,
    )


def _should_save_latest(
    *, epoch: int, total_epochs: int, checkpoint_every: int, latest_checkpoint_every: int
) -> bool:
    if latest_checkpoint_every > 0:
        return epoch % latest_checkpoint_every == 0 or epoch == total_epochs
    if checkpoint_every > 0:
        return epoch % checkpoint_every == 0 or epoch == total_epochs
    return epoch == total_epochs


def _learning_rate_for_epoch(
    base_lr: float,
    schedule: str,
    epoch: int,
    *,
    total_epochs: int,
    step_size: int,
    gamma: float,
    lr_min: float,
) -> float:
    if schedule == "constant":
        return float(base_lr)
    if schedule == "step":
        decay_steps = max(0, (epoch - 1) // max(1, step_size))
        return max(float(lr_min), float(base_lr) * (gamma**decay_steps))
    if schedule == "cosine":
        if total_epochs <= 1:
            return float(base_lr)
        progress = (epoch - 1) / max(1, total_epochs - 1)
        cosine_factor = 0.5 * (1.0 + np.cos(np.pi * progress))
        return float(lr_min + (base_lr - lr_min) * cosine_factor)
    raise ValueError(f"Unknown lr schedule: {schedule}")


def _set_learning_rate(optimizer: object, lr: float) -> None:
    if not isinstance(optimizer, torch.optim.Optimizer):
        return
    for group in optimizer.param_groups:
        group["lr"] = lr


def _format_loss_delta(new: float, old: float | None) -> str:
    if old is None:
        return "n/a"
    return f"{new - old:+.6f}"


def _format_lr(lr: float) -> str:
    return f"{lr:.4g}" if lr >= 1e-3 else f"{lr:.3e}"


def _remove_missing_ok(path: Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        path.unlink()


def _append_unique(paths: list[Path], value: Path) -> None:
    if value not in paths:
        paths.append(value)


def _build_checkpoint_meta(
    *,
    config: DiffusionTrainingConfig,
    epoch: int,
    loss: float,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "horizon": config.horizon,
        "state_dim": config.state_dim,
        "n_diffusion_steps": config.n_diffusion_steps,
        "dataset": config.dataset_key,
        "name": config.dataset_key,
        "epoch": int(epoch),
        "loss": float(loss),
        "dim": config.n_hidden,
    }
    if extra:
        payload.update(extra)
    return payload


def _run_epoch_with_validation(
    trainer: object,
    train_loader: object,
    val_loader: object | None,
) -> tuple[float, float | None, float]:
    trainer_impl = cast(_SupportsEpochCalls, trainer)
    train_loss = trainer_impl.train_epoch(train_loader)
    val_loss = trainer_impl.evaluate_epoch(val_loader) if val_loader is not None else None
    metric_loss = val_loss if val_loss is not None else train_loss
    return train_loss, val_loss, metric_loss


def _manage_stage_checkpoints(
    p: _StageParams,
    *,
    epoch: int,
    metric_loss: float,
    train_loss: float,
    is_new_best: bool,
    recent_ckpts: deque[Path],
    ckpt_paths: list[Path],
    latest_ckpt: Path,
    best_ckpt: Path,
) -> None:
    manager = cast(CheckpointManager, p.checkpoint_manager)
    config = cast(DiffusionTrainingConfig, p.config)
    ema_sd: dict[str, object] | None = None

    def _ema_state_dict() -> dict[str, object]:
        nonlocal ema_sd
        if ema_sd is None:
            ema_obj = cast(_SupportsStateDict, p.ema)
            ema_sd = ema_obj.state_dict()
        return ema_sd

    if p.checkpoint_every > 0 and epoch % p.checkpoint_every == 0:
        periodic = manager.checkpoint_path(p.phase, epoch)
        manager.save(
            periodic,
            model=cast(torch.nn.Module, p.model),
            normalizer=p.normalizer,
            meta=_build_checkpoint_meta(
                config=config,
                epoch=epoch,
                loss=train_loss,
                extra=p.extra_meta,
            ),
            model_kind=p.phase,
            ema_state_dict=_ema_state_dict(),
        )
        if p.keep_last_checkpoints > 0 and len(recent_ckpts) >= p.keep_last_checkpoints:
            _remove_missing_ok(recent_ckpts[0])
        recent_ckpts.append(periodic)
        _append_unique(ckpt_paths, periodic)

    if _should_save_latest(
        epoch=epoch,
        total_epochs=p.effective_epochs,
        checkpoint_every=p.checkpoint_every,
        latest_checkpoint_every=p.latest_checkpoint_every,
    ):
        manager.save(
            latest_ckpt,
            model=cast(torch.nn.Module, p.model),
            normalizer=p.normalizer,
            meta=_build_checkpoint_meta(
                config=config,
                epoch=epoch,
                loss=metric_loss,
                extra=p.extra_meta,
            ),
            model_kind=p.phase,
            ema_state_dict=_ema_state_dict(),
        )
        _append_unique(ckpt_paths, latest_ckpt)

    if is_new_best:
        best_meta = _build_checkpoint_meta(
            config=config,
            epoch=epoch,
            loss=metric_loss,
            extra=p.extra_meta,
        )
        best_meta["is_best"] = True
        manager.save(
            best_ckpt,
            model=cast(torch.nn.Module, p.model),
            normalizer=p.normalizer,
            meta=best_meta,
            model_kind=p.phase,
            ema_state_dict=_ema_state_dict(),
        )
        _append_unique(ckpt_paths, best_ckpt)

        output_path = getattr(p.config, "output_path", None)
        if output_path:
            phase_alias = Path(str(output_path)) / f"{p.phase}.pt"
            manager.save(
                phase_alias,
                model=cast(torch.nn.Module, p.model),
                normalizer=p.normalizer,
                meta=best_meta,
                model_kind=p.phase,
                ema_state_dict=_ema_state_dict(),
            )
            _append_unique(ckpt_paths, phase_alias)
