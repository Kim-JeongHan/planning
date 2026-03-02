"""Epoch trainers and EMA accumulator for diffusion/value models."""

from __future__ import annotations

import contextlib

import torch  # type: ignore
import torch.nn.functional as functional  # type: ignore

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
        state = {
            name: tensor.clone()
            for name, tensor in self.model.state_dict().items()  # type: ignore[union-attr]
        }
        for name, tensor in self.shadow.items():
            state[name] = tensor.clone()
        return state


class BaseEpochTrainer:
    """Common epoch loop shared by diffusion and value trainers.

    Subclasses implement ``_compute_loss(observations)`` and optionally
    ``_filter_batch(observations)`` to skip invalid batches.
    """

    def __init__(
        self,
        model: object,
        optimizer: object,
        ema: EMAAccumulator | None,
        model_device: torch.device,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.ema = ema
        self._model_device = model_device

    def _compute_loss(self, observations: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _filter_batch(self, observations: torch.Tensor) -> bool:
        """Return False to skip this batch."""
        return True

    def _run_loop(self, loader: object, *, training: bool) -> float:
        if training:
            self.model.train()  # type: ignore[union-attr]
        else:
            self.model.eval()  # type: ignore[union-attr]

        total: torch.Tensor | None = None
        count = 0
        ctx = contextlib.nullcontext() if training else torch.no_grad()
        with ctx:
            for observations, _condition in loader:
                observations = observations.to(self._model_device, non_blocking=True)
                if not self._filter_batch(observations):
                    continue
                loss = self._compute_loss(observations)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)  # type: ignore[union-attr]
                    loss.backward()
                    self.optimizer.step()  # type: ignore[union-attr]
                    if self.ema is not None:
                        self.ema.update()
                total = loss.detach() if total is None else total + loss.detach()
                count += 1

        if training:
            self.model.train()  # type: ignore[union-attr]
        return float((total / count).item()) if count > 0 and total is not None else 0.0

    def train_epoch(self, loader: object) -> float:
        return self._run_loop(loader, training=True)

    def evaluate_epoch(self, loader: object) -> float:
        return self._run_loop(loader, training=False)


class DiffusionEpochTrainer(BaseEpochTrainer):
    """One epoch of diffusion-model optimization.

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
        ema: EMAAccumulator | None = None,
    ) -> None:
        super().__init__(
            model=model,
            optimizer=optimizer,
            ema=ema,
            model_device=next(model.parameters()).device,  # type: ignore[union-attr]
        )
        self.schedule = schedule
        self._alpha_bar = torch.as_tensor(schedule.alpha_bar, dtype=torch.float32)

    def _compute_loss(self, observations: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(observations)
        t = torch.randint(
            0,
            self.schedule.n_diffusion_steps,
            (observations.shape[0],),
            device=observations.device,
        )
        alpha_bar = self._alpha_bar.to(device=observations.device, dtype=observations.dtype)[t]
        alpha_bar = alpha_bar.view(-1, *([1] * (observations.ndim - 1)))
        noisy = torch.sqrt(alpha_bar) * observations + torch.sqrt(1.0 - alpha_bar) * noise
        eps_pred = self.model(noisy, t)  # type: ignore[union-attr]
        return functional.mse_loss(eps_pred, noise)


class ValueEpochTrainer(BaseEpochTrainer):
    """One epoch of value-model optimization.

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
        ema: EMAAccumulator | None = None,
    ) -> None:
        super().__init__(
            model=model,
            optimizer=optimizer,
            ema=ema,
            model_device=next(model.parameters()).device,  # type: ignore[union-attr]
        )
        self.normalizer = normalizer

    def _filter_batch(self, observations: torch.Tensor) -> bool:
        return observations.shape[1] >= 1

    def _compute_loss(self, observations: torch.Tensor) -> torch.Tensor:
        goal = observations[:, -1, :]  # [B, D]
        distances = torch.norm(observations - goal[:, None, :], dim=-1)  # [B, H]
        target = torch.log1p(distances.mean(dim=1, keepdim=True))  # [B, 1]
        pred = self.model(observations)  # type: ignore[union-attr]
        target = target.to(dtype=pred.dtype, device=pred.device)
        return functional.mse_loss(pred, target)
