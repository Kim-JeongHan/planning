"""Guided value model epoch trainer."""

from __future__ import annotations

import torch  # type: ignore
import torch.nn.functional as functional  # type: ignore

from .diffusion_trainer import EMAAccumulator


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
        ema: EMAAccumulator | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.normalizer = normalizer
        self.ema = ema
        self._model_device = next(model.parameters()).device  # type: ignore[union-attr]

    def _compute_target(self, observations: object) -> object:
        """Compute distance-to-goal target from the trajectory's final state."""
        goal = observations[:, -1, :]  # [B, D]
        distances = torch.norm(observations - goal[:, None, :], dim=-1)  # [B, H]
        return torch.log1p(distances.mean(dim=1, keepdim=True))  # [B, 1]

    def train_epoch(self, loader: object) -> float:
        total_loss: torch.Tensor | None = None
        count = 0
        for observations, _condition in loader:
            observations = observations.to(self._model_device, non_blocking=True)
            if observations.shape[1] < 1:
                continue
            target = self._compute_target(observations)
            pred = self.model(observations)
            target = target.to(dtype=pred.dtype, device=pred.device)
            loss = functional.mse_loss(pred, target)
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
                if observations.shape[1] < 1:
                    continue
                target = self._compute_target(observations)
                pred = self.model(observations)
                target = target.to(dtype=pred.dtype, device=pred.device)
                loss = functional.mse_loss(pred, target)
                detached_loss = loss.detach()
                total_loss = detached_loss if total_loss is None else total_loss + detached_loss
                count += 1
        self.model.train()
        if count == 0 or total_loss is None:
            return 0.0
        return float((total_loss / count).item())
