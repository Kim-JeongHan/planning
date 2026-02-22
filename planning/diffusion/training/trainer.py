"""Minimal training loop for local diffusion/value models."""

from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np
import torch  # type: ignore
import torch.nn.functional as functional  # type: ignore
from tqdm import tqdm

from ..config import DiffusionTrainingPipelineConfig
from ..model import DiffusionModel, ValueModel
from .checkpoint import (
    CheckpointConfig,
    CheckpointPathManager,
    CheckpointWriter,
)
from .config import DiffusionTrainingConfig
from .dataset import TorchTensorFactory, TrajectoryDataSetSource, TrajectoryLoadConfig
from .noise import DiffusionSchedule


class EMAAccumulator:
    """Exponential moving average of model parameters (used for inference).

    After each optimizer step, call ``update()`` to blend the current model
    weights into the shadow copy.  The shadow weights are saved as
    ``ema_state_dict`` in every checkpoint.

    ``decay`` of 0.995 follows the default in the Janner et al. diffuser repo.
    """

    def __init__(self, model: object, decay: float = 0.995) -> None:
        self.model = model
        self.decay = float(decay)
        # Clone current parameters into shadow.
        self.shadow: dict[str, torch.Tensor] = {
            name: param.data.clone()
            for name, param in model.named_parameters()  # type: ignore[union-attr]
        }

    def update(self) -> None:
        """Blend current model weights into the EMA shadow copy."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():  # type: ignore[union-attr]
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, object]:
        """Return the EMA shadow weights (used for ``ema_state_dict`` in checkpoints)."""
        return {name: tensor.clone() for name, tensor in self.shadow.items()}


class DiffusionEpochTrainer:
    """Encapsulate one epoch of diffusion-model optimization.

    Trains on the ε-prediction objective (MSE between predicted and actual noise)
    as specified in Janner et al. 2022.  The diffusion model receives only the
    noisy trajectory and timestep — no condition vector.  Goal conditioning is
    handled at inference time via inpainting.
    """

    def __init__(
        self,
        model: object,
        optimizer: object,
        schedule: DiffusionSchedule,
        torch_backend: object,
        functional_backend: object,
        ema: EMAAccumulator | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.schedule = schedule
        self.torch = torch_backend
        self.functional = functional_backend
        self.ema = ema
        self._alpha_bar = self.torch.as_tensor(schedule.alpha_bar, dtype=self.torch.float32)

    def train_epoch(self, loader: object) -> float:
        total = 0.0
        count = 0
        for observations, _condition in loader:
            noise = self.torch.randn_like(observations)
            t = self.torch.randint(
                0,
                self.schedule.n_diffusion_steps,
                (observations.shape[0],),
                device=observations.device,
            )
            alpha_bar = self._alpha_bar.to(
                device=observations.device, dtype=observations.dtype
            )[t]
            alpha_bar = alpha_bar.view(-1, *([1] * (observations.ndim - 1)))
            noisy = self.torch.sqrt(alpha_bar) * observations + self.torch.sqrt(
                1.0 - alpha_bar
            ) * noise
            eps_pred = self.model(noisy, t)
            loss = self.functional.mse_loss(eps_pred, noise)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            if self.ema is not None:
                self.ema.update()
            total += float(loss.item())
            count += 1
        if count == 0:
            return 0.0
        return total / count

    def evaluate_epoch(self, loader: object) -> float:
        self.model.eval()
        total = 0.0
        count = 0
        with self.torch.no_grad():
            for observations, _condition in loader:
                noise = self.torch.randn_like(observations)
                t = self.torch.randint(
                    0,
                    self.schedule.n_diffusion_steps,
                    (observations.shape[0],),
                    device=observations.device,
                )
                alpha_bar = self._alpha_bar.to(
                    device=observations.device, dtype=observations.dtype
                )[t]
                alpha_bar = alpha_bar.view(-1, *([1] * (observations.ndim - 1)))
                noisy = self.torch.sqrt(alpha_bar) * observations + self.torch.sqrt(
                    1.0 - alpha_bar
                ) * noise
                eps_pred = self.model(noisy, t)
                loss = self.functional.mse_loss(eps_pred, noise)
                total += float(loss.item())
                count += 1
        self.model.train()
        if count == 0:
            return 0.0
        return total / count


class ValueEpochTrainer:
    """Encapsulate one epoch of value-model optimization.

    The value model J_φ predicts a proxy for trajectory quality: the mean
    log(1 + distance) from each state to the trajectory's final state.  No
    condition vector is passed to the model; goal conditioning is implicit
    because the trajectory's final state IS the goal in the training data.
    """

    def __init__(
        self,
        model: object,
        optimizer: object,
        normalizer: object,
        torch_backend: object,
        functional_backend: object,
        ema: EMAAccumulator | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.normalizer = normalizer
        self.torch = torch_backend
        self.functional = functional_backend
        self.ema = ema

    def _compute_target(self, observations: object) -> object:
        """Compute distance-to-goal target from the trajectory's final state."""
        goal = observations[:, -1, :]  # [B, D]
        distances = self.torch.norm(observations - goal[:, None, :], dim=-1)  # [B, H]
        return self.torch.log1p(distances.mean(dim=1, keepdim=True))  # [B, 1]

    def train_epoch(self, loader: object) -> float:
        total = 0.0
        count = 0
        for observations, _condition in loader:
            if observations.shape[1] < 1:
                continue
            target = self._compute_target(observations)
            pred = self.model(observations)
            target = target.to(dtype=pred.dtype, device=pred.device)
            loss = self.functional.mse_loss(pred, target)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            if self.ema is not None:
                self.ema.update()
            total += float(loss.item())
            count += 1
        if count == 0:
            return 0.0
        return total / count

    def evaluate_epoch(self, loader: object) -> float:
        self.model.eval()
        total = 0.0
        count = 0
        with self.torch.no_grad():
            for observations, _condition in loader:
                if observations.shape[1] < 1:
                    continue
                target = self._compute_target(observations)
                pred = self.model(observations)
                target = target.to(dtype=pred.dtype, device=pred.device)
                loss = self.functional.mse_loss(pred, target)
                total += float(loss.item())
                count += 1
        self.model.train()
        if count == 0:
            return 0.0
        return total / count


def _log_every(epochs: int, target_ticks: int = 20) -> int:
    """Return logging frequency for long epoch runs."""
    if epochs <= 1:
        return 1
    return max(1, min(epochs // target_ticks, 100))


def _coerce_log_every(epochs: int, log_every: int | None) -> int:
    if log_every is None or log_every <= 0:
        return _log_every(epochs)
    return max(1, int(log_every))


def _create_summary_writer(log_dir: str | None) -> object | None:
    """Create a TensorBoard summary writer if requested."""
    if log_dir is None:
        return None

    try:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime optional dependency
        raise ImportError(
            "TensorBoard is required for summary logging. "
            "Install it in your environment before enabling --tensorboard-log-dir."
        ) from exc

    writer_dir = Path(log_dir)
    writer_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(writer_dir))


def _log_scalar(writer: object | None, tag: str, value: float, step: int) -> None:
    if writer is not None:
        writer.add_scalar(tag, float(value), step)


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


class DiffusionTrainingPipeline:
    """Orchestrator object for diffusion/value training workflow."""

    def __init__(self, **values: object) -> None:
        self.cfg = DiffusionTrainingPipelineConfig(**values)

    def run(self) -> list[Path]:
        """Run the full training pipeline."""
        return self.execute()

    def execute(self) -> list[Path]:
        """Execute training and return saved checkpoint paths."""
        return self._run_impl()

    def _run_impl(self) -> list[Path]:
        cfg = self.cfg

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
            np.random.seed(cfg.seed)

        summary_writer = _create_summary_writer(cfg.tensorboard_log_dir)

        dataset_key = Path(cfg.dataset).stem
        if dataset_key == "":
            dataset_key = "dataset"

        dataset_source = TrajectoryDataSetSource(
            TrajectoryLoadConfig(
                path=cfg.dataset,
                horizon=cfg.horizon,
                state_dim=cfg.state_dim,
                seed=cfg.seed,
            )
        )
        sequences = dataset_source.load()
        horizon = cfg.horizon
        if horizon is None:
            inferred_horizon = max(
                (
                    trajectory.shape[0]
                    for trajectory in sequences
                    if trajectory.ndim == 2 and trajectory.shape[1] == cfg.state_dim
                ),
                default=None,
            )
            if inferred_horizon is None:
                raise ValueError(
                    f"No valid trajectory with state_dim={cfg.state_dim} found in dataset: {cfg.dataset}"
                )
            print(f"[info] --horizon omitted. Using inferred horizon={inferred_horizon}.")
            horizon = inferred_horizon
            dataset_source = TrajectoryDataSetSource(
                TrajectoryLoadConfig(
                    path=cfg.dataset,
                    horizon=horizon,
                    state_dim=cfg.state_dim,
                    seed=cfg.seed,
                )
            )

        config = DiffusionTrainingConfig.from_pipeline_config(
            self.cfg,
            horizon=horizon,
            dataset_name=dataset_key,
            coerce_lr_schedule=_coerce_lr_schedule,
        )

        checkpoint_every, keep_last_checkpoints, best_top_k = _coerce_checkpoint_policy(
            config.checkpoint_every,
            config.keep_last_checkpoints,
            config.best_top_k,
        )
        validation_split = _coerce_validation_split(config.validation_split)
        latest_checkpoint_every = _coerce_latest_checkpoint_every(
            config.latest_checkpoint_every
        )

        trajectories = dataset_source.to_normalized_numpy(trajectories=sequences)
        if trajectories.shape[0] == 0:
            raise ValueError("No valid trajectory data found.")
        print(
            "[info] Loaded "
            f"{trajectories.shape[0]} trajectories, horizon={config.horizon}, "
            f"state_dim={config.state_dim}"
        )

        normalizer = DiffusionTrainingConfig._derive_normalizer(trajectories)
        tensor_factory = TorchTensorFactory(normalizer, device="cpu")
        obs_t, cond_t = tensor_factory.to_torch_tensors(trajectories)
        dataset_tensors = torch.utils.data.TensorDataset(obs_t, cond_t)
        train_loader, val_loader = _build_dataloaders(
            dataset_tensors=dataset_tensors,
            batch_size=config.batch_size,
            validation_split=validation_split,
            seed=cfg.seed,
        )
        schedule_name = config.lr_schedule
        print(
            "[info] Training schedule: "
            f"{schedule_name} (lr={config.learning_rate}, step_size={config.lr_step_size}, "
            f"gamma={config.lr_gamma}, lr_min={config.lr_min})"
        )
        diffusion_model = DiffusionModel(
            state_dim=config.state_dim,
            horizon=config.horizon,
            n_diffusion_steps=config.n_diffusion_steps,
            dim=config.n_hidden,
        )
        # Cosine noise schedule (Nichol & Dhariwal 2021) as used in the paper.
        schedule = DiffusionSchedule.cosine(n_diffusion_steps=config.n_diffusion_steps)
        diffusion_optimizer = torch.optim.AdamW(
            diffusion_model.parameters(), lr=config.learning_rate
        )
        diffusion_ema = EMAAccumulator(diffusion_model)
        diffusion_epoch_trainer = DiffusionEpochTrainer(
            model=diffusion_model,
            optimizer=diffusion_optimizer,
            schedule=schedule,
            torch_backend=torch,
            functional_backend=functional,
            ema=diffusion_ema,
        )
        checkpoint_config = CheckpointConfig(
            dataset=dataset_key,
            horizon=config.horizon,
            n_diffusion_steps=config.n_diffusion_steps,
            root=cfg.output_root,
            discount=config.discount,
        )
        checkpoint_writer = CheckpointWriter(CheckpointPathManager(checkpoint_config))

        diffusion_patience, diffusion_min_delta = _coerce_stop_arguments(
            patience=config.diffusion_patience,
            min_delta=config.diffusion_min_delta,
            name="diffusion",
        )
        value_patience, value_min_delta = _coerce_stop_arguments(
            patience=config.value_patience, min_delta=config.value_min_delta, name="value"
        )

        effective_diffusion_epochs = config.epochs
        if config.diffusion_max_epochs is not None:
            effective_diffusion_epochs = min(
                effective_diffusion_epochs, config.diffusion_max_epochs
            )
        if effective_diffusion_epochs <= 0:
            raise ValueError("diffusion_max_epochs / epochs must be at least 1.")
        diffuse_log_every = _coerce_log_every(
            effective_diffusion_epochs, cfg.log_every
        )

        if summary_writer is not None:
            summary_writer.add_text(
                "training/config",
                (
                    f"dataset={config.dataset}\n"
                    f"horizon={config.horizon}\n"
                    f"state_dim={config.state_dim}\n"
                    f"n_diffusion_steps={config.n_diffusion_steps}\n"
                    f"batch_size={config.batch_size}\n"
                    f"learning_rate={config.learning_rate}\n"
                    f"lr_schedule={config.lr_schedule}\n"
                    f"epochs={effective_diffusion_epochs}\n"
                    f"seed={cfg.seed}\n"
                    f"checkpoint_every={checkpoint_every}\n"
                    f"keep_last_checkpoints={keep_last_checkpoints}\n"
                    f"best_top_k={best_top_k}\n"
                ),
                0,
            )

        diffusion_ckpts = self._run_diffusion_stage(
            config=config,
            dataset_key=dataset_key,
            normalizer=normalizer,
            checkpoint_writer=checkpoint_writer,
            train_loader=train_loader,
            val_loader=val_loader,
            diffusion_model=diffusion_model,
            diffusion_optimizer=diffusion_optimizer,
            diffusion_epoch_trainer=diffusion_epoch_trainer,
            diffusion_ema=diffusion_ema,
            summary_writer=summary_writer,
            diffusion_patience=diffusion_patience,
            diffusion_min_delta=diffusion_min_delta,
            effective_diffusion_epochs=effective_diffusion_epochs,
            diffuse_log_every=diffuse_log_every,
            checkpoint_every=checkpoint_every,
            latest_checkpoint_every=latest_checkpoint_every,
            keep_last_checkpoints=keep_last_checkpoints,
            best_top_k=best_top_k,
        )

        if not config.train_value:
            if summary_writer is not None:
                summary_writer.close()
            return [path for path in diffusion_ckpts if path.exists()]

        value_model = ValueModel(
            state_dim=config.state_dim,
            horizon=config.horizon,
            dim=config.n_hidden,
        )
        value_optimizer = torch.optim.AdamW(value_model.parameters(), lr=config.learning_rate)
        value_ema = EMAAccumulator(value_model)
        value_epoch_trainer = ValueEpochTrainer(
            model=value_model,
            optimizer=value_optimizer,
            normalizer=normalizer,
            torch_backend=torch,
            functional_backend=functional,
            ema=value_ema,
        )

        effective_value_epochs = config.epochs if config.train_value else 0
        if config.value_max_epochs is not None and config.value_max_epochs < 0:
            raise ValueError("value_max_epochs must be >= 0.")
        if config.value_max_epochs is not None:
            effective_value_epochs = min(effective_value_epochs, config.value_max_epochs)
        if effective_value_epochs < 1:
            if summary_writer is not None:
                summary_writer.close()
            return [path for path in diffusion_ckpts if path.exists()]

        value_log_every = _coerce_log_every(effective_value_epochs, cfg.log_every)
        value_ckpts = self._run_value_stage(
            config=config,
            dataset_key=dataset_key,
            normalizer=normalizer,
            checkpoint_writer=checkpoint_writer,
            train_loader=train_loader,
            val_loader=val_loader,
            value_model=value_model,
            value_optimizer=value_optimizer,
            value_epoch_trainer=value_epoch_trainer,
            value_ema=value_ema,
            summary_writer=summary_writer,
            value_patience=value_patience,
            value_min_delta=value_min_delta,
            effective_value_epochs=effective_value_epochs,
            value_log_every=value_log_every,
            checkpoint_every=checkpoint_every,
            latest_checkpoint_every=latest_checkpoint_every,
            keep_last_checkpoints=keep_last_checkpoints,
            best_top_k=best_top_k,
        )

        if summary_writer is not None:
            summary_writer.close()
        return [path for path in (diffusion_ckpts + value_ckpts) if path.exists()]

    def _run_diffusion_stage(
        self,
        *,
        config: DiffusionTrainingConfig,
        dataset_key: str,
        normalizer: object,
        checkpoint_writer: CheckpointWriter,
        train_loader: object,
        val_loader: object | None,
        diffusion_model: object,
        diffusion_optimizer: object,
        diffusion_epoch_trainer: DiffusionEpochTrainer,
        diffusion_ema: EMAAccumulator,
        summary_writer: object | None,
        diffusion_patience: int | None,
        diffusion_min_delta: float,
        effective_diffusion_epochs: int,
        diffuse_log_every: int,
        checkpoint_every: int,
        latest_checkpoint_every: int,
        keep_last_checkpoints: int,
        best_top_k: int,
    ) -> list[Path]:
        diffusion_ckpts: list[Path] = []
        diffusion_tracker = EpochLossTracker(
            min_delta=diffusion_min_delta, patience=diffusion_patience
        )
        diffusion_recent_ckpts: list[Path] = []
        diffusion_best_records: list[tuple[float, int, Path]] = []

        diffusion_bar = tqdm(
            range(1, effective_diffusion_epochs + 1),
            desc="Diffusion training",
            unit="epoch",
        )
        diffusion_latest_ckpt = checkpoint_writer.path_manager.latest("diffusion")
        diffusion_best_ckpt = checkpoint_writer.path_manager.best("diffusion")
        for epoch in diffusion_bar:
            current_lr = _learning_rate_for_epoch(
                config.learning_rate,
                config.lr_schedule,
                epoch,
                total_epochs=effective_diffusion_epochs,
                step_size=config.lr_step_size,
                gamma=config.lr_gamma,
                lr_min=config.lr_min,
            )
            _set_learning_rate(diffusion_optimizer, current_lr)
            train_loss, val_loss, metric_loss = _run_epoch_with_validation(
                diffusion_epoch_trainer, train_loader, val_loader
            )
            is_new_best, delta = diffusion_tracker.update(metric_loss, epoch)
            best_tag = " [new best]" if is_new_best else ""
            diffusion_bar.set_postfix(
                {
                    "train_loss": f"{train_loss:.4f}",
                    "val_loss": "n/a" if val_loss is None else f"{val_loss:.4f}",
                    "best": f"{diffusion_tracker.best_loss:.4f}",
                    "delta": delta,
                    "lr": _format_lr(current_lr),
                }
            )
            if (
                epoch % diffuse_log_every == 0
                or epoch == 1
                or epoch == effective_diffusion_epochs
            ):
                best_label = (
                    ""
                    if diffusion_tracker.best_loss is None
                    else (
                        f" best={diffusion_tracker.best_loss:.6f} "
                        f"(epoch={diffusion_tracker.best_epoch})"
                    )
                )
                print(
                    f"[diffusion] epoch {epoch}/{effective_diffusion_epochs} "
                    f"train_loss={train_loss:.6f} "
                    f"val_loss={'n/a' if val_loss is None else f'{val_loss:.6f}'} "
                    f"lr={current_lr:.6f} "
                    f"delta={delta}{best_label}{best_tag}"
                )
            _log_scalar(summary_writer, "diffusion/loss", train_loss, epoch)
            if val_loss is not None:
                _log_scalar(summary_writer, "diffusion/val_loss", val_loss, epoch)
            _log_scalar(summary_writer, "diffusion/lr", current_lr, epoch)
            if diffusion_tracker.best_loss is not None:
                _log_scalar(
                    summary_writer,
                    "diffusion/best_loss",
                    diffusion_tracker.best_loss,
                    epoch,
                )
            diffusion_best_records = _manage_stage_checkpoints(
                phase="diffusion",
                epoch=epoch,
                total_epochs=effective_diffusion_epochs,
                model=diffusion_model,
                ema=diffusion_ema,
                normalizer=normalizer,
                checkpoint_writer=checkpoint_writer,
                config=config,
                dataset_key=dataset_key,
                metric_loss=metric_loss,
                train_loss=train_loss,
                is_new_best=is_new_best,
                best_records=diffusion_best_records,
                recent_ckpts=diffusion_recent_ckpts,
                ckpt_paths=diffusion_ckpts,
                latest_ckpt=diffusion_latest_ckpt,
                best_ckpt=diffusion_best_ckpt,
                checkpoint_every=checkpoint_every,
                latest_checkpoint_every=latest_checkpoint_every,
                keep_last_checkpoints=keep_last_checkpoints,
                best_top_k=best_top_k,
            )

            if diffusion_tracker.should_stop():
                print(
                    f"[early-stop][diffusion] Stopping at epoch {epoch}/{effective_diffusion_epochs} "
                    f"due to no improvement for {diffusion_tracker.no_improve} epochs."
                )
                break

        print(
            "[summary][diffusion] best epoch="
            f"{'n/a' if diffusion_tracker.best_epoch is None else diffusion_tracker.best_epoch}, "
            f"best loss={diffusion_tracker.best_loss:.6f}"
        )
        diffusion_bar.close()
        return [path for path in diffusion_ckpts if path.exists()]

    def _run_value_stage(
        self,
        *,
        config: DiffusionTrainingConfig,
        dataset_key: str,
        normalizer: object,
        checkpoint_writer: CheckpointWriter,
        train_loader: object,
        val_loader: object | None,
        value_model: object,
        value_optimizer: object,
        value_epoch_trainer: ValueEpochTrainer,
        value_ema: EMAAccumulator,
        summary_writer: object | None,
        value_patience: int | None,
        value_min_delta: float,
        effective_value_epochs: int,
        value_log_every: int,
        checkpoint_every: int,
        latest_checkpoint_every: int,
        keep_last_checkpoints: int,
        best_top_k: int,
    ) -> list[Path]:
        value_ckpts: list[Path] = []
        value_tracker = EpochLossTracker(min_delta=value_min_delta, patience=value_patience)
        value_recent_ckpts: list[Path] = []
        value_best_records: list[tuple[float, int, Path]] = []
        value_bar = tqdm(
            range(1, effective_value_epochs + 1),
            desc="Value training",
            unit="epoch",
        )
        value_latest_ckpt = checkpoint_writer.path_manager.latest("value")
        value_best_ckpt = checkpoint_writer.path_manager.best("value")
        for epoch in value_bar:
            current_lr = _learning_rate_for_epoch(
                config.learning_rate,
                config.lr_schedule,
                epoch,
                total_epochs=effective_value_epochs,
                step_size=config.lr_step_size,
                gamma=config.lr_gamma,
                lr_min=config.lr_min,
            )
            _set_learning_rate(value_optimizer, current_lr)
            value_train_loss, value_val_loss, value_metric_loss = _run_epoch_with_validation(
                value_epoch_trainer, train_loader, val_loader
            )
            is_new_best, delta = value_tracker.update(value_metric_loss, epoch)
            best_tag = " [new best]" if is_new_best else ""
            value_bar.set_postfix(
                {
                    "train_loss": f"{value_train_loss:.4f}",
                    "val_loss": "n/a" if value_val_loss is None else f"{value_val_loss:.4f}",
                    "best": f"{value_tracker.best_loss:.4f}",
                    "delta": delta,
                    "lr": _format_lr(current_lr),
                }
            )
            if epoch % value_log_every == 0 or epoch == 1 or epoch == effective_value_epochs:
                best_label = (
                    ""
                    if value_tracker.best_loss is None
                    else (
                        f" best={value_tracker.best_loss:.6f} "
                        f"(epoch={value_tracker.best_epoch})"
                    )
                )
                print(
                    f"[value] epoch {epoch}/{effective_value_epochs} "
                    f"train_loss={value_train_loss:.6f} "
                    f"val_loss={'n/a' if value_val_loss is None else f'{value_val_loss:.6f}'} "
                    f"lr={current_lr:.6f} "
                    f"delta={delta}{best_label}{best_tag}"
                )
            _log_scalar(summary_writer, "value/loss", value_train_loss, epoch)
            if value_val_loss is not None:
                _log_scalar(summary_writer, "value/val_loss", value_val_loss, epoch)
            _log_scalar(summary_writer, "value/lr", current_lr, epoch)
            if value_tracker.best_loss is not None:
                _log_scalar(summary_writer, "value/best_loss", value_tracker.best_loss, epoch)
            value_best_records = _manage_stage_checkpoints(
                phase="value",
                epoch=epoch,
                total_epochs=effective_value_epochs,
                model=value_model,
                ema=value_ema,
                normalizer=normalizer,
                checkpoint_writer=checkpoint_writer,
                config=config,
                dataset_key=dataset_key,
                metric_loss=value_metric_loss,
                train_loss=value_train_loss,
                is_new_best=is_new_best,
                best_records=value_best_records,
                recent_ckpts=value_recent_ckpts,
                ckpt_paths=value_ckpts,
                latest_ckpt=value_latest_ckpt,
                best_ckpt=value_best_ckpt,
                checkpoint_every=checkpoint_every,
                latest_checkpoint_every=latest_checkpoint_every,
                keep_last_checkpoints=keep_last_checkpoints,
                best_top_k=best_top_k,
                extra_meta={"discount": config.discount},
            )

            if value_tracker.should_stop():
                print(
                    f"[early-stop][value] Stopping at epoch {epoch}/{effective_value_epochs} "
                    f"due to no improvement for {value_tracker.no_improve} epochs."
                )
                break

        print(
            "[summary][value] best epoch="
            f"{'n/a' if value_tracker.best_epoch is None else value_tracker.best_epoch}, "
            f"best loss={value_tracker.best_loss:.6f}"
        )
        value_bar.close()
        return [path for path in value_ckpts if path.exists()]


def _coerce_lr_schedule(
    schedule: str, *, step_size: int, gamma: float, lr_min: float
) -> str:
    schedule = schedule.strip().lower()
    if schedule not in {"constant", "step", "cosine"}:
        raise ValueError(
            "Unsupported lr-schedule. Choose one of: constant, step, cosine."
        )
    if schedule == "step" and step_size <= 0:
        raise ValueError("lr_step_size must be positive when lr_schedule='step'.")
    if not 0.0 < gamma < 1.0 and schedule == "step":
        raise ValueError("lr_gamma must be in the interval (0, 1) for lr_schedule='step'.")
    if lr_min < 0:
        raise ValueError("lr_min must be non-negative.")
    return schedule


def _coerce_stop_arguments(
    *, patience: int | None, min_delta: float, name: str
) -> tuple[int | None, float]:
    if patience is not None and patience < 1:
        raise ValueError(f"{name}_patience must be >= 1 when enabled.")
    if min_delta < 0.0:
        raise ValueError(f"{name}_min_delta must be >= 0.")
    return patience, float(min_delta)


def _coerce_checkpoint_policy(
    checkpoint_every: int, keep_last_checkpoints: int, best_top_k: int
) -> tuple[int, int, int]:
    if checkpoint_every < 0:
        raise ValueError("checkpoint_every must be >= 0.")
    if keep_last_checkpoints < 0:
        raise ValueError("keep_last_checkpoints must be >= 0.")
    if best_top_k < 1:
        raise ValueError("best_top_k must be >= 1.")
    return int(checkpoint_every), int(keep_last_checkpoints), int(best_top_k)


def _coerce_validation_split(validation_split: float) -> float:
    if not 0.0 <= float(validation_split) < 1.0:
        raise ValueError("validation_split must be in [0.0, 1.0).")
    return float(validation_split)


def _coerce_latest_checkpoint_every(latest_checkpoint_every: int) -> int:
    if latest_checkpoint_every < 0:
        raise ValueError("latest_checkpoint_every must be >= 0.")
    return int(latest_checkpoint_every)


def _should_save_latest(
    *,
    epoch: int,
    total_epochs: int,
    checkpoint_every: int,
    latest_checkpoint_every: int,
) -> bool:
    if latest_checkpoint_every > 0:
        return epoch % latest_checkpoint_every == 0 or epoch == total_epochs
    if checkpoint_every > 0:
        return epoch % checkpoint_every == 0 or epoch == total_epochs
    return epoch == total_epochs


def _build_dataloaders(
    dataset_tensors: object,
    batch_size: int,
    validation_split: float,
    seed: int | None = None,
) -> tuple[object, object | None]:
    train_dataset = dataset_tensors
    if validation_split <= 0.0:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        return train_loader, None

    dataset_size = len(dataset_tensors)
    num_validation = int(dataset_size * validation_split)
    if num_validation < 1 or num_validation >= dataset_size:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        return train_loader, None

    num_train = dataset_size - num_validation
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)

    train_subset, validation_subset = torch.utils.data.random_split(
        dataset_tensors,
        [num_train, num_validation],
        generator=generator,
    )
    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_subset,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, validation_loader


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
    for group in optimizer.param_groups:
        group["lr"] = lr


def _format_loss_delta(new: float, old: float | None) -> str:
    if old is None:
        return "n/a"
    return f"{new - old:+.6f}"


def _format_lr(lr: float) -> str:
    if lr >= 1e-3:
        return f"{lr:.4g}"
    return f"{lr:.3e}"


def _remove_missing_ok(path: Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        path.unlink()


def _prune_kept_checkpoints(kept: list[Path], keep_last: int) -> None:
    if keep_last <= 0:
        return
    while len(kept) > keep_last:
        stale = kept.pop(0)
        _remove_missing_ok(stale)


def _append_unique(paths: list[Path], value: Path) -> None:
    if value not in paths:
        paths.append(value)


def _best_checkpoint_path(path_manager: CheckpointPathManager, kind: str, epoch: int) -> Path:
    return path_manager.checkpoint_root(kind) / f"best_epoch_{int(epoch):04d}.ckpt"


def _build_checkpoint_meta(
    *,
    config: DiffusionTrainingConfig,
    dataset_key: str,
    epoch: int,
    loss: float,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "horizon": config.horizon,
        "state_dim": config.state_dim,
        "n_diffusion_steps": config.n_diffusion_steps,
        "dataset": dataset_key,
        "name": dataset_key,
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
    train_loss = trainer.train_epoch(train_loader)
    val_loss = trainer.evaluate_epoch(val_loader) if val_loader is not None else None
    metric_loss = val_loss if val_loader is not None else train_loss
    return train_loss, val_loss, metric_loss


def _manage_stage_checkpoints(
    *,
    phase: str,
    epoch: int,
    total_epochs: int,
    model: object,
    ema: EMAAccumulator,
    normalizer: object,
    checkpoint_writer: CheckpointWriter,
    config: DiffusionTrainingConfig,
    dataset_key: str,
    metric_loss: float,
    train_loss: float,
    is_new_best: bool,
    best_records: list[tuple[float, int, Path]],
    recent_ckpts: list[Path],
    ckpt_paths: list[Path],
    latest_ckpt: Path,
    best_ckpt: Path,
    checkpoint_every: int,
    latest_checkpoint_every: int,
    keep_last_checkpoints: int,
    best_top_k: int,
    extra_meta: dict[str, object] | None = None,
) -> list[tuple[float, int, Path]]:
    ema_sd = ema.state_dict()
    if checkpoint_every > 0 and epoch % checkpoint_every == 0:
        periodic_ckpt_path = checkpoint_writer.path_manager.checkpoint_path(phase, epoch)
        checkpoint_writer.save(
            periodic_ckpt_path,
            model=model,
            normalizer=normalizer,
            meta=_build_checkpoint_meta(
                config=config,
                dataset_key=dataset_key,
                epoch=epoch,
                loss=train_loss,
                extra=extra_meta,
            ),
            model_kind=phase,
            ema_state_dict=ema_sd,
        )
        if recent_ckpts is not None:
            recent_ckpts.append(periodic_ckpt_path)
            _prune_kept_checkpoints(recent_ckpts, keep_last_checkpoints)
        _append_unique(ckpt_paths, periodic_ckpt_path)

    best_candidate_ckpt = _best_checkpoint_path(checkpoint_writer.path_manager, phase, epoch)
    best_records, candidate_is_best = _update_best_checkpoints(
        current_best_records=best_records,
        candidate_loss=metric_loss,
        candidate_epoch=epoch,
        candidate_path=best_candidate_ckpt,
        best_top_k=best_top_k,
    )
    if candidate_is_best:
        candidate_meta = _build_checkpoint_meta(
            config=config,
            dataset_key=dataset_key,
            epoch=epoch,
            loss=metric_loss,
            extra=extra_meta,
        )
        candidate_meta.update({"is_best_ranked": True})
        checkpoint_writer.save(
            best_candidate_ckpt,
            model=model,
            normalizer=normalizer,
            meta=candidate_meta,
            model_kind=phase,
            ema_state_dict=ema_sd,
        )
        _append_unique(ckpt_paths, best_candidate_ckpt)

    if _should_save_latest(
        epoch=epoch,
        total_epochs=total_epochs,
        checkpoint_every=checkpoint_every,
        latest_checkpoint_every=latest_checkpoint_every,
    ):
        checkpoint_writer.save(
            latest_ckpt,
            model=model,
            normalizer=normalizer,
            meta=_build_checkpoint_meta(
                config=config,
                dataset_key=dataset_key,
                epoch=epoch,
                loss=metric_loss,
                extra=extra_meta,
            ),
            model_kind=phase,
            ema_state_dict=ema_sd,
        )
        _append_unique(ckpt_paths, latest_ckpt)

    if is_new_best:
        best_meta = _build_checkpoint_meta(
            config=config,
            dataset_key=dataset_key,
            epoch=epoch,
            loss=metric_loss,
            extra=extra_meta,
        )
        best_meta.update({"is_best": True})
        checkpoint_writer.save(
            best_ckpt,
            model=model,
            normalizer=normalizer,
            meta=best_meta,
            model_kind=phase,
            ema_state_dict=ema_sd,
        )
        if best_ckpt not in ckpt_paths:
            ckpt_paths.append(best_ckpt)

    return best_records


def _update_best_checkpoints(
    current_best_records: list[tuple[float, int, Path]],
    candidate_loss: float,
    candidate_epoch: int,
    candidate_path: Path,
    best_top_k: int,
) -> tuple[list[tuple[float, int, Path]], bool]:
    candidates = [*current_best_records, (candidate_loss, candidate_epoch, candidate_path)]
    candidates.sort(key=lambda item: (item[0], item[1]))
    next_best = candidates[:best_top_k]

    stale_paths = {path for _, _, path in current_best_records} - {
        path for _, _, path in next_best
    }
    for stale_path in stale_paths:
        _remove_missing_ok(stale_path)

    candidate_in_best = any(path == candidate_path for _, _, path in next_best)
    return next_best, candidate_in_best
