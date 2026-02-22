"""Diffusion model epoch trainer and EMA accumulator."""

from __future__ import annotations

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
        ema: EMAAccumulator | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.schedule = schedule
        self.ema = ema
        self._alpha_bar = torch.as_tensor(schedule.alpha_bar, dtype=torch.float32)
        self._model_device = next(model.parameters()).device  # type: ignore[union-attr]

    def train_epoch(self, loader: object) -> float:
        total_loss: torch.Tensor | None = None
        count = 0
        for observations, _condition in loader:
            observations = observations.to(self._model_device, non_blocking=True)
            noise = torch.randn_like(observations)
            t = torch.randint(
                0,
                self.schedule.n_diffusion_steps,
                (observations.shape[0],),
                device=observations.device,
            )
            alpha_bar = self._alpha_bar.to(
                device=observations.device, dtype=observations.dtype
            )[t]
            alpha_bar = alpha_bar.view(-1, *([1] * (observations.ndim - 1)))
            noisy = torch.sqrt(alpha_bar) * observations + torch.sqrt(
                1.0 - alpha_bar
            ) * noise
            eps_pred = self.model(noisy, t)
            loss = functional.mse_loss(eps_pred, noise)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            if self.ema is not None:
                self.ema.update()
            detached_loss = loss.detach()
            total_loss = detached_loss if total_loss is None else total_loss + detached_loss
            count += 1
        if count == 0 or total_loss is None:
            return 0.0
        return float((total_loss / count).item())

    def evaluate_epoch(self, loader: object) -> float:
        self.model.eval()
        total_loss: torch.Tensor | None = None
        count = 0
        with torch.no_grad():
            for observations, _condition in loader:
                observations = observations.to(self._model_device, non_blocking=True)
                noise = torch.randn_like(observations)
                t = torch.randint(
                    0,
                    self.schedule.n_diffusion_steps,
                    (observations.shape[0],),
                    device=observations.device,
                )
                alpha_bar = self._alpha_bar.to(
                    device=observations.device, dtype=observations.dtype
                )[t]
                alpha_bar = alpha_bar.view(-1, *([1] * (observations.ndim - 1)))
                noisy = torch.sqrt(alpha_bar) * observations + torch.sqrt(
                    1.0 - alpha_bar
                ) * noise
                eps_pred = self.model(noisy, t)
                loss = functional.mse_loss(eps_pred, noise)
                detached_loss = loss.detach()
                total_loss = detached_loss if total_loss is None else total_loss + detached_loss
                count += 1
        self.model.train()
        if count == 0 or total_loss is None:
            return 0.0
        return float((total_loss / count).item())
