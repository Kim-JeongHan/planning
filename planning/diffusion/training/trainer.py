"""Minimal training loop for local diffusion/value models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch  # type: ignore
import torch.nn.functional as functional  # type: ignore
from tqdm import tqdm

from ..model import SimpleDiffusionModel, SimpleValueModel
from .checkpoint import (
    CheckpointConfig,
    CheckpointPathManager,
    CheckpointWriter,
)
from .config import DiffusionTrainingConfig
from .dataset import (
    ConditionTensorBuilder,
    TorchTensorFactory,
    TrajectoryDataSetSource,
    TrajectoryLoadConfig,
)
from .noise import DiffusionSchedule


class _ConditionProjector:
    """Pad/truncate condition tensors to a fixed embedding width."""

    def __init__(self, target_dim: int, torch_backend: object) -> None:
        self.target_dim = int(target_dim)
        self.torch = torch_backend

    def project(
        self,
        condition: object,
        *,
        device: object,
        dtype: object,
    ) -> object:
        condition_t = condition
        if self.target_dim <= 0:
            return condition_t[:, :0]

        if condition_t.shape[1] >= self.target_dim:
            return condition_t[:, : self.target_dim]

        pad = self.torch.zeros(
            (condition_t.shape[0], self.target_dim - condition_t.shape[1]),
            device=device,
            dtype=dtype,
        )
        return self.torch.cat([condition_t, pad], dim=-1)


class DiffusionEpochTrainer:
    """Encapsulate one epoch of diffusion-model optimization."""

    def __init__(
        self,
        model: object,
        optimizer: object,
        schedule: DiffusionSchedule,
        condition_dim: int,
        torch_backend: object,
        functional_backend: object,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.schedule = schedule
        self.condition_dim = condition_dim
        self.torch = torch_backend
        self.functional = functional_backend
        self.condition_projector = _ConditionProjector(condition_dim, torch_backend)

    def train_epoch(self, loader: object) -> float:
        total = 0.0
        count = 0
        for observations, condition in loader:
            condition_t = self.condition_projector.project(
                condition,
                device=observations.device,
                dtype=observations.dtype,
            )
            noise = self.torch.randn_like(observations)
            t = self.torch.randint(
                0,
                self.schedule.n_diffusion_steps,
                (observations.shape[0],),
                device=observations.device,
            )
            alpha_bar = self.torch.as_tensor(
                self.schedule.alpha_bar, device=observations.device, dtype=self.torch.float32
            )[t]
            while alpha_bar.ndim < observations.ndim:
                alpha_bar = alpha_bar[:, None, None]
            noisy = self.torch.sqrt(alpha_bar) * observations + self.torch.sqrt(
                1.0 - alpha_bar
            ) * noise
            eps_pred = self.model(noisy, t, condition_t)
            loss = self.functional.mse_loss(eps_pred, noise)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total += float(loss.item())
            count += 1
        if count == 0:
            return 0.0
        return total / count


class ValueEpochTrainer:
    """Encapsulate one epoch of value-model optimization."""

    def __init__(
        self,
        model: object,
        optimizer: object,
        condition_dim: int,
        normalizer: object,
        torch_backend: object,
        functional_backend: object,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.condition_dim = condition_dim
        self.normalizer = normalizer
        self.torch = torch_backend
        self.functional = functional_backend
        self.condition_projector = _ConditionProjector(condition_dim, torch_backend)

    def _resolve_goal(self, observations: object, condition: object | None) -> object:
        state_dim = int(observations.shape[-1])
        if self.normalizer is not None:
            mean = self.torch.as_tensor(
                getattr(self.normalizer, "mean", None)[:state_dim],
                device=observations.device,
            )
            std = self.torch.as_tensor(
                getattr(self.normalizer, "std", None)[:state_dim],
                device=observations.device,
            )
            trajectories = observations * std + mean
            default_goal = trajectories[:, -1, :state_dim]
        else:
            default_goal = observations[:, -1, :state_dim]
            trajectories = observations

        if condition is None or (self.torch.is_tensor(condition) and condition.numel() == 0):
            return default_goal, default_goal, trajectories

        condition_t = condition
        if not self.torch.is_tensor(condition_t):
            condition_t = self.torch.as_tensor(
                condition_t,
                device=observations.device,
                dtype=observations.dtype,
            )

        if condition_t.ndim == 1:
            condition_t = condition_t.reshape(1, -1)
        elif condition_t.ndim != 2:
            condition_t = condition_t.reshape(condition_t.shape[0], -1)

        if condition_t.shape[1] >= 2 * state_dim or condition_t.shape[1] >= state_dim:
            goal = condition_t[:, -state_dim:]
        else:
            goal = self.torch.zeros(
                (condition_t.shape[0], state_dim),
                device=observations.device,
                dtype=observations.dtype,
            )
            width = min(state_dim, condition_t.shape[1])
            goal[:, :width] = condition_t[:, :width]

        return goal, condition_t, trajectories

    def train_epoch(self, loader: object) -> float:
        total = 0.0
        count = 0
        for observations, condition in loader:
            if observations.shape[1] < 1:
                continue

            goal, condition_t, trajectories = self._resolve_goal(observations, condition)
            if goal.shape[0] != observations.shape[0]:
                goal = goal[: observations.shape[0]]

            distances = self.torch.norm(
                trajectories[:, :, : observations.shape[-1]] - goal[:, None, :], dim=-1
            )
            target = self.torch.log1p(self.torch.mean(distances, dim=1, keepdim=True))
            value_condition = self.condition_projector.project(
                condition_t,
                device=observations.device,
                dtype=observations.dtype,
            )
            pred = self.model(observations, value_condition)
            target = target.to(dtype=pred.dtype, device=pred.device)
            loss = self.functional.mse_loss(pred, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total += float(loss.item())
            count += 1
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
    """Pure orchestrator object for diffusion/value training workflow."""

    def __init__(
        self,
        *,
        dataset: str,
        output_root: str,
        horizon: int | None,
        state_dim: int,
        n_diffusion_steps: int,
        epochs: int,
        batch_size: int,
        learning_rate: float = 1e-3,
        lr_schedule: str = "constant",
        lr_step_size: int = 100,
        lr_gamma: float = 0.5,
        lr_min: float = 1e-5,
        discount: float = 1.0,
        seed: int | None = None,
        train_value: bool = True,
        log_every: int | None = None,
        diffusion_max_epochs: int | None = None,
        value_max_epochs: int | None = None,
        diffusion_patience: int | None = None,
        value_patience: int | None = None,
        diffusion_min_delta: float = 0.0,
        value_min_delta: float = 0.0,
    ) -> None:
        self.dataset = dataset
        self.output_root = output_root
        self.horizon = horizon
        self.state_dim = state_dim
        self.n_diffusion_steps = n_diffusion_steps
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lr_schedule = lr_schedule
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.lr_min = lr_min
        self.discount = discount
        self.seed = seed
        self.train_value = train_value
        self.log_every = log_every
        self.diffusion_max_epochs = diffusion_max_epochs
        self.value_max_epochs = value_max_epochs
        self.diffusion_patience = diffusion_patience
        self.value_patience = value_patience
        self.diffusion_min_delta = diffusion_min_delta
        self.value_min_delta = value_min_delta

    def run(self) -> list[Path]:
        return _run_training_impl(
            dataset=self.dataset,
            output_root=self.output_root,
            horizon=self.horizon,
            state_dim=self.state_dim,
            n_diffusion_steps=self.n_diffusion_steps,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            lr_schedule=self.lr_schedule,
            lr_step_size=self.lr_step_size,
            lr_gamma=self.lr_gamma,
            lr_min=self.lr_min,
            seed=self.seed,
            discount=self.discount,
            train_value=self.train_value,
            log_every=self.log_every,
            diffusion_max_epochs=self.diffusion_max_epochs,
            value_max_epochs=self.value_max_epochs,
            diffusion_patience=self.diffusion_patience,
            value_patience=self.value_patience,
            diffusion_min_delta=self.diffusion_min_delta,
            value_min_delta=self.value_min_delta,
        )


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


def _run_training_impl(  # noqa: C901
    *,
    dataset: str,
    output_root: str = "logs",
    horizon: int | None = None,
    state_dim: int = 3,
    n_diffusion_steps: int = 100,
    epochs: int = 1,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    lr_schedule: str = "constant",
    lr_step_size: int = 100,
    lr_gamma: float = 0.5,
    lr_min: float = 1e-5,
    seed: int | None = None,
    discount: float = 1.0,
    train_value: bool = True,
    log_every: int | None = None,
    diffusion_max_epochs: int | None = None,
    value_max_epochs: int | None = None,
    diffusion_patience: int | None = None,
    value_patience: int | None = None,
    diffusion_min_delta: float = 0.0,
    value_min_delta: float = 0.0,
) -> list[Path]:
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    dataset_key = Path(dataset).stem
    if dataset_key == "":
        dataset_key = "dataset"

    dataset_source = TrajectoryDataSetSource(
        TrajectoryLoadConfig(
            path=dataset,
            horizon=horizon,
            state_dim=state_dim,
            seed=seed,
        )
    )
    sequences = dataset_source.load()
    if horizon is None:
        inferred_horizon = max(
            (
                trajectory.shape[0]
                for trajectory in sequences
                if trajectory.ndim == 2 and trajectory.shape[1] == state_dim
            ),
            default=None,
        )
        if inferred_horizon is None:
            raise ValueError(
                f"No valid trajectory with state_dim={state_dim} found in dataset: {dataset}"
            )
        print(f"[info] --horizon omitted. Using inferred horizon={inferred_horizon}.")
        horizon = inferred_horizon
        dataset_source = TrajectoryDataSetSource(
            TrajectoryLoadConfig(
                path=dataset,
                horizon=horizon,
                state_dim=state_dim,
                seed=seed,
            )
        )

    config = DiffusionTrainingConfig(
        dataset=dataset,
        horizon=horizon,
        state_dim=state_dim,
        n_diffusion_steps=n_diffusion_steps,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        lr_schedule=_coerce_lr_schedule(
            lr_schedule,
            step_size=lr_step_size,
            gamma=lr_gamma,
            lr_min=lr_min,
        ),
        lr_step_size=lr_step_size,
        lr_gamma=lr_gamma,
        lr_min=lr_min,
        discount=discount,
        train_value=train_value,
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
    obs_t, _ = tensor_factory.to_torch_tensors(trajectories)
    cond_t = torch.as_tensor(
        ConditionTensorBuilder.build(trajectories), dtype=torch.float32, device=obs_t.device
    )
    dataset_tensors = torch.utils.data.TensorDataset(obs_t, cond_t)
    loader = torch.utils.data.DataLoader(
        dataset_tensors,
        batch_size=config.batch_size,
        shuffle=True,
    )
    condition_dim = cond_t.shape[-1]
    print(f"[info] Condition dimension: {condition_dim}")

    schedule_name = config.lr_schedule
    print(
        "[info] Training schedule: "
        f"{schedule_name} (lr={config.learning_rate}, step_size={config.lr_step_size}, "
        f"gamma={config.lr_gamma}, lr_min={config.lr_min})"
    )
    diffusion_model = SimpleDiffusionModel.create(
        state_dim=config.state_dim,
        horizon=config.horizon,
        n_diffusion_steps=config.n_diffusion_steps,
        n_hidden=config.n_hidden,
        n_layers=config.n_layers,
        condition_dim=condition_dim,
    )
    schedule = DiffusionSchedule.linear(
        n_diffusion_steps=config.n_diffusion_steps,
        beta_start=1e-4,
        beta_end=2e-2,
    )
    diffusion_optimizer = torch.optim.AdamW(diffusion_model.parameters(), lr=config.learning_rate)
    diffusion_epoch_trainer = DiffusionEpochTrainer(
        model=diffusion_model,
        optimizer=diffusion_optimizer,
        schedule=schedule,
        condition_dim=condition_dim,
        torch_backend=torch,
        functional_backend=functional,
    )
    checkpoint_config = CheckpointConfig(
        dataset=dataset_key,
        horizon=config.horizon,
        n_diffusion_steps=config.n_diffusion_steps,
        root=output_root,
        discount=config.discount,
    )
    checkpoint_writer = CheckpointWriter(CheckpointPathManager(checkpoint_config))
    diffusion_ckpts = []
    diffusion_tracker = EpochLossTracker(
        min_delta=diffusion_min_delta, patience=diffusion_patience
    )
    effective_diffusion_epochs = config.epochs
    if diffusion_max_epochs is not None:
        effective_diffusion_epochs = min(effective_diffusion_epochs, diffusion_max_epochs)
    if effective_diffusion_epochs <= 0:
        raise ValueError("diffusion_max_epochs / epochs must be at least 1.")
    diffuse_log_every = _coerce_log_every(effective_diffusion_epochs, log_every)

    diffusion_patience, diffusion_min_delta = _coerce_stop_arguments(
        patience=diffusion_patience, min_delta=diffusion_min_delta, name="diffusion"
    )
    value_patience, value_min_delta = _coerce_stop_arguments(
        patience=value_patience, min_delta=value_min_delta, name="value"
    )

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
        loss = diffusion_epoch_trainer.train_epoch(loader)
        is_new_best, delta = diffusion_tracker.update(loss, epoch)
        best_tag = " [new best]" if is_new_best else ""
        diffusion_bar.set_postfix(
            {
                "loss": f"{loss:.4f}",
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
                f"loss={loss:.6f} lr={current_lr:.6f} "
                f"delta={delta}{best_label}{best_tag}"
            )
        checkpoint_writer.save(
            diffusion_latest_ckpt,
            model=diffusion_model,
            normalizer=normalizer,
            meta={
                "horizon": config.horizon,
                "state_dim": config.state_dim,
                "n_diffusion_steps": config.n_diffusion_steps,
                "dataset": dataset_key,
                "name": dataset_key,
                "epoch": epoch,
                "loss": float(loss),
                "n_hidden": config.n_hidden,
                "n_layers": config.n_layers,
                "condition_dim": condition_dim,
            },
            model_kind="diffusion",
        )
        if is_new_best:
            checkpoint_writer.save(
                diffusion_best_ckpt,
                model=diffusion_model,
                normalizer=normalizer,
                meta={
                    "horizon": config.horizon,
                    "state_dim": config.state_dim,
                    "n_diffusion_steps": config.n_diffusion_steps,
                    "dataset": dataset_key,
                    "name": dataset_key,
                    "epoch": epoch,
                    "loss": float(loss),
                    "n_hidden": config.n_hidden,
                    "n_layers": config.n_layers,
                    "condition_dim": condition_dim,
                    "is_best": True,
                },
                model_kind="diffusion",
            )
            if diffusion_best_ckpt not in diffusion_ckpts:
                diffusion_ckpts.append(diffusion_best_ckpt)
        if epoch == 1:
            diffusion_ckpts.append(diffusion_latest_ckpt)

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

    value_ckpts: list[Path] = []
    if not train_value:
        diffusion_bar.close()
        return diffusion_ckpts

    value_model = SimpleValueModel.create(
        state_dim=config.state_dim,
        horizon=config.horizon,
        n_hidden=config.n_hidden,
        n_layers=config.n_layers,
        condition_dim=condition_dim,
    )
    value_optimizer = torch.optim.AdamW(value_model.parameters(), lr=config.learning_rate)
    value_epoch_trainer = ValueEpochTrainer(
        model=value_model,
        optimizer=value_optimizer,
        condition_dim=condition_dim,
        normalizer=normalizer,
        torch_backend=torch,
        functional_backend=functional,
    )
    effective_value_epochs = config.epochs if train_value else 0
    if value_max_epochs is not None and value_max_epochs < 0:
        raise ValueError("value_max_epochs must be >= 0.")
    if value_max_epochs is not None:
        effective_value_epochs = min(effective_value_epochs, value_max_epochs)
    if effective_value_epochs < 1:
        return diffusion_ckpts
    value_log_every = _coerce_log_every(effective_value_epochs, log_every)

    value_latest_ckpt = checkpoint_writer.path_manager.latest("value")
    value_best_ckpt = checkpoint_writer.path_manager.best("value")
    value_tracker = EpochLossTracker(min_delta=value_min_delta, patience=value_patience)
    value_bar = tqdm(range(1, effective_value_epochs + 1), desc="Value training", unit="epoch")
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
        value_loss = value_epoch_trainer.train_epoch(loader)
        is_new_best, delta = value_tracker.update(value_loss, epoch)
        best_tag = " [new best]" if is_new_best else ""
        value_bar.set_postfix(
            {
                "loss": f"{value_loss:.4f}",
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
                f"loss={value_loss:.6f} lr={current_lr:.6f} "
                f"delta={delta}{best_label}{best_tag}"
            )
        checkpoint_writer.save(
            value_latest_ckpt,
            model=value_model,
            normalizer=normalizer,
            meta={
                "horizon": config.horizon,
                "state_dim": config.state_dim,
                "n_diffusion_steps": config.n_diffusion_steps,
                "dataset": dataset_key,
                "name": dataset_key,
                "epoch": epoch,
                "loss": float(value_loss),
                "discount": config.discount,
                "condition_dim": condition_dim,
            },
            model_kind="value",
        )
        if is_new_best:
            checkpoint_writer.save(
                value_best_ckpt,
                model=value_model,
                normalizer=normalizer,
                meta={
                    "horizon": config.horizon,
                    "state_dim": config.state_dim,
                    "n_diffusion_steps": config.n_diffusion_steps,
                    "dataset": dataset_key,
                    "name": dataset_key,
                    "epoch": epoch,
                    "loss": float(value_loss),
                    "discount": config.discount,
                    "condition_dim": condition_dim,
                    "is_best": True,
                },
                model_kind="value",
            )
            if value_best_ckpt not in value_ckpts:
                value_ckpts.append(value_best_ckpt)
        if epoch == 1:
            value_ckpts.append(value_latest_ckpt)

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
    diffusion_bar.close()

    return diffusion_ckpts + value_ckpts
