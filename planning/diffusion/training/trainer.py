"""Minimal training loop for local diffusion/value models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch  # type: ignore
import torch.nn.functional as functional  # type: ignore
from tqdm import tqdm

from ..model import SimpleDiffusionModel, SimpleValueModel
from .checkpoint import (
    build_best_checkpoint_path,
    build_latest_checkpoint_path,
    save_checkpoint,
)
from .config import DiffusionTrainingConfig
from .dataset import (
    load_trajectory_sequences,
    make_condition_tensor,
    normalize_sequences,
    to_torch_tensors,
)
from .noise import DiffusionSchedule


def _prepare_condition(
    condition: object,
    target_dim: int,
    torch: object,
    *,
    device: object,
    dtype: object,
) -> object:
    if target_dim <= 0:
        return condition[:, :0]

    if condition.shape[1] >= target_dim:
        return condition[:, :target_dim]

    pad = torch.zeros(
        (condition.shape[0], target_dim - condition.shape[1]), device=device, dtype=dtype
    )
    return torch.cat([condition, pad], dim=-1)


def _train_diffusion_epoch(
    model: object,
    loader: object,
    optimizer: object,
    schedule: DiffusionSchedule,
    condition_dim: int,
    torch: object,
    functional: object,
) -> float:
    total = 0.0
    count = 0
    for observations, condition in loader:
        condition_t = _prepare_condition(
            condition,
            condition_dim,
            torch=torch,
            device=observations.device,
            dtype=observations.dtype,
        )
        noise = torch.randn_like(observations)
        t = torch.randint(0, schedule.n_diffusion_steps, (observations.shape[0],), device=observations.device)
        alpha_bar = torch.as_tensor(schedule.alpha_bar, device=observations.device, dtype=torch.float32)[t]
        while alpha_bar.ndim < observations.ndim:
            alpha_bar = alpha_bar[:, None, None]
        noisy = torch.sqrt(alpha_bar) * observations + torch.sqrt(1.0 - alpha_bar) * noise
        eps_pred = model(noisy, t, condition_t)
        loss = functional.mse_loss(eps_pred, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss.item())
        count += 1
    if count == 0:
        return 0.0
    return total / count


def _train_value_epoch(
    model: object,
    loader: object,
    optimizer: object,
    condition_dim: int,
    normalizer: object,
    torch: object,
    functional: object,
) -> float:
    total = 0.0
    count = 0
    for observations, condition in loader:
        if observations.shape[1] < 1:
            continue

        state_dim = int(observations.shape[-1])
        if normalizer is not None:
            mean = torch.as_tensor(
                getattr(normalizer, "mean", None)[:state_dim],
                device=observations.device,
            )
            std = torch.as_tensor(
                getattr(normalizer, "std", None)[:state_dim],
                device=observations.device,
            )
            trajectories = observations * std + mean
            terminal = trajectories[:, -1, :state_dim]
            default_goal = terminal
        else:
            terminal = observations[:, -1, :state_dim]
            default_goal = terminal
            trajectories = observations

        if condition is None or condition.numel() == 0:
            goal = default_goal
        else:
            if not torch.is_tensor(condition):
                condition = torch.as_tensor(condition, device=observations.device, dtype=observations.dtype)
            if condition.ndim == 1:
                condition = condition.reshape(1, -1)
            elif condition.ndim != 2:
                condition = condition.reshape(condition.shape[0], -1)

            if condition.shape[1] >= 2 * state_dim or condition.shape[1] >= state_dim:
                goal = condition[:, -state_dim:]
            else:
                goal = torch.zeros(
                    (condition.shape[0], state_dim),
                    device=observations.device,
                    dtype=observations.dtype,
                )
                width = min(state_dim, condition.shape[1])
                goal[:, :width] = condition[:, :width]

        if goal.shape[0] != observations.shape[0]:
            goal = goal[:observations.shape[0]]

        distances = torch.norm(trajectories[:, :, :state_dim] - goal[:, None, :], dim=-1)
        target = torch.log1p(torch.mean(distances, dim=1, keepdim=True))
        value_condition = _prepare_condition(
            condition,
            condition_dim,
            torch=torch,
            device=observations.device,
            dtype=observations.dtype,
        )
        pred = model(observations, value_condition)
        target = target.to(dtype=pred.dtype, device=pred.device)
        loss = functional.mse_loss(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
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


def train(  # noqa: C901
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

    sequences = load_trajectory_sequences(dataset)
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

    trajectories = normalize_sequences(sequences, horizon=config.horizon, state_dim=config.state_dim)
    if trajectories.shape[0] == 0:
        raise ValueError("No valid trajectory data found.")
    print(
        "[info] Loaded "
        f"{trajectories.shape[0]} trajectories, horizon={config.horizon}, "
        f"state_dim={config.state_dim}"
    )

    normalizer = DiffusionTrainingConfig._derive_normalizer(trajectories)  # type: ignore[attr-defined]
    obs_t, _ = to_torch_tensors(trajectories, normalizer, device="cpu")
    cond_t = torch.as_tensor(
        make_condition_tensor(trajectories), dtype=torch.float32, device=obs_t.device
    )
    dataset_tensors = torch.utils.data.TensorDataset(obs_t, cond_t)
    loader = torch.utils.data.DataLoader(dataset_tensors, batch_size=config.batch_size, shuffle=True)
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
    diffusion_ckpts = []
    diffusion_best_loss: float | None = None
    diffusion_best_epoch: int | None = None
    diffusion_prev_loss: float | None = None
    diffusion_no_improve = 0
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
    diffusion_latest_ckpt = build_latest_checkpoint_path(
        output_root,
        dataset=dataset_key,
        kind="diffusion",
        horizon=config.horizon,
        n_diffusion_steps=config.n_diffusion_steps,
    )
    diffusion_best_ckpt = build_best_checkpoint_path(
        output_root,
        dataset=dataset_key,
        kind="diffusion",
        horizon=config.horizon,
        n_diffusion_steps=config.n_diffusion_steps,
    )
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
        loss = _train_diffusion_epoch(
            diffusion_model,
            loader,
            diffusion_optimizer,
            schedule,
            condition_dim=condition_dim,
            torch=torch,
            functional=functional,
        )
        delta = _format_loss_delta(loss, diffusion_prev_loss)
        diffusion_prev_loss = loss
        is_new_best = False
        if diffusion_best_loss is None or loss < (diffusion_best_loss - diffusion_min_delta):
            diffusion_best_loss = loss
            diffusion_best_epoch = epoch
            diffusion_no_improve = 0
            is_new_best = True
            best_tag = " [new best]"
        else:
            diffusion_no_improve += 1
            best_tag = ""
        diffusion_bar.set_postfix(
            {
                "loss": f"{loss:.4f}",
                "best": f"{diffusion_best_loss:.4f}",
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
                if diffusion_best_loss is None
                else f" best={diffusion_best_loss:.6f} (epoch={diffusion_best_epoch})"
            )
            print(
                f"[diffusion] epoch {epoch}/{effective_diffusion_epochs} "
                f"loss={loss:.6f} lr={current_lr:.6f} "
                f"delta={delta}{best_label}{best_tag}"
            )
        save_checkpoint(
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
            save_checkpoint(
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

        if (
            diffusion_patience is not None
            and diffusion_no_improve >= diffusion_patience
        ):
            print(
                f"[early-stop][diffusion] Stopping at epoch {epoch}/{effective_diffusion_epochs} "
                f"due to no improvement for {diffusion_no_improve} epochs."
            )
            break

    print(
        "[summary][diffusion] best epoch="
        f"{'n/a' if diffusion_best_epoch is None else diffusion_best_epoch}, "
        f"best loss={diffusion_best_loss:.6f}"
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
    effective_value_epochs = config.epochs if train_value else 0
    if value_max_epochs is not None and value_max_epochs < 0:
        raise ValueError("value_max_epochs must be >= 0.")
    if value_max_epochs is not None:
        effective_value_epochs = min(effective_value_epochs, value_max_epochs)
    if effective_value_epochs < 1:
        return diffusion_ckpts
    value_log_every = _coerce_log_every(effective_value_epochs, log_every)

    value_latest_ckpt = build_latest_checkpoint_path(
        output_root,
        dataset=dataset_key,
        kind="value",
        horizon=config.horizon,
        n_diffusion_steps=config.n_diffusion_steps,
        discount=config.discount,
    )
    value_best_ckpt = build_best_checkpoint_path(
        output_root,
        dataset=dataset_key,
        kind="value",
        horizon=config.horizon,
        n_diffusion_steps=config.n_diffusion_steps,
        discount=config.discount,
    )
    value_best_loss: float | None = None
    value_best_epoch: int | None = None
    value_prev_loss: float | None = None
    value_no_improve = 0
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
        value_loss = _train_value_epoch(
            model=value_model,
            loader=loader,
            optimizer=value_optimizer,
            condition_dim=condition_dim,
            normalizer=normalizer,
            torch=torch,
            functional=functional,
        )
        delta = _format_loss_delta(value_loss, value_prev_loss)
        value_prev_loss = value_loss
        is_new_best = False
        if value_best_loss is None or value_loss < (value_best_loss - value_min_delta):
            value_best_loss = value_loss
            value_best_epoch = epoch
            value_no_improve = 0
            is_new_best = True
            best_tag = " [new best]"
        else:
            value_no_improve += 1
            best_tag = ""
        value_bar.set_postfix(
            {
                "loss": f"{value_loss:.4f}",
                "best": f"{value_best_loss:.4f}",
                "delta": delta,
                "lr": _format_lr(current_lr),
            }
        )
        if epoch % value_log_every == 0 or epoch == 1 or epoch == effective_value_epochs:
            best_label = (
                ""
                if value_best_loss is None
                else f" best={value_best_loss:.6f} (epoch={value_best_epoch})"
            )
            print(
                f"[value] epoch {epoch}/{effective_value_epochs} "
                f"loss={value_loss:.6f} lr={current_lr:.6f} "
                f"delta={delta}{best_label}{best_tag}"
            )
        save_checkpoint(
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
            save_checkpoint(
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

        if value_patience is not None and value_no_improve >= value_patience:
            print(
                f"[early-stop][value] Stopping at epoch {epoch}/{effective_value_epochs} "
                f"due to no improvement for {value_no_improve} epochs."
            )
            break

    print(
        "[summary][value] best epoch="
        f"{'n/a' if value_best_epoch is None else value_best_epoch}, "
        f"best loss={value_best_loss:.6f}"
    )
    value_bar.close()
    diffusion_bar.close()

    return diffusion_ckpts + value_ckpts
