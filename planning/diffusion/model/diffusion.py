"""Diffusion model definitions."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch  # type: ignore

from .utils import ConditionNormalizer, MLPBackbone


class SimpleDiffusionModel:
    """MLP baseline model for noise prediction."""

    def __new__(cls, *args: object, **kwargs: object) -> SimpleDiffusionModel:
        return super().__new__(cls)

    @staticmethod
    def create(
        *,
        state_dim: int,
        horizon: int,
        n_diffusion_steps: int,
        n_hidden: int = 256,
        n_layers: int = 2,
        condition_dim: int = 0,
    ) -> DiffusionModel:
        time_dim = 2
        input_dim = horizon * state_dim + time_dim + condition_dim
        output_dim = horizon * state_dim
        model = MLPBackbone(
            input_dim,
            output_dim,
            hidden_dim=n_hidden,
            n_layers=n_layers,
        )

        model.horizon = int(horizon)
        model.state_dim = int(state_dim)
        model.n_diffusion_steps = int(n_diffusion_steps)
        model.condition_dim = int(condition_dim)
        model.hparams = {
            "state_dim": int(state_dim),
            "horizon": int(horizon),
            "n_diffusion_steps": int(n_diffusion_steps),
            "n_hidden": int(n_hidden),
            "n_layers": int(n_layers),
            "condition_dim": int(condition_dim),
            "model_class_path": "planning.diffusion.model.DiffusionModel",
        }
        return DiffusionModel(
            model,
            state_dim=state_dim,
            horizon=horizon,
            n_diffusion_steps=n_diffusion_steps,
            n_hidden=n_hidden,
            n_layers=n_layers,
            condition_dim=condition_dim,
        )


class DiffusionModel:
    """Torch wrapper around an MLP backbone for diffusion."""

    def __init__(
        self,
        backbone: MLPBackbone,
        *,
        state_dim: int,
        horizon: int,
        n_diffusion_steps: int,
        n_hidden: int,
        n_layers: int,
        condition_dim: int,
    ) -> None:
        self.backbone = backbone
        self.horizon = int(horizon)
        self.state_dim = int(state_dim)
        self.n_diffusion_steps = int(n_diffusion_steps)
        self.n_hidden = int(n_hidden)
        self.n_layers = int(n_layers)
        self.condition_dim = int(condition_dim)
        self._condition_normalizer = ConditionNormalizer(self.condition_dim)
        self.hparams = {
            "state_dim": self.state_dim,
            "horizon": self.horizon,
            "n_diffusion_steps": self.n_diffusion_steps,
            "n_hidden": self.n_hidden,
            "n_layers": self.n_layers,
            "condition_dim": self.condition_dim,
            "model_class_path": "planning.diffusion.model.DiffusionModel",
        }

    def __call__(
        self,
        x: object,
        t: object,
        condition: dict[str, object] | None = None,
    ) -> torch.Tensor:
        base = x.reshape(x.shape[0], -1)
        t = t.view(-1, 1).float()
        t_emb = torch.cat([torch.sin(t / 10000.0), torch.cos(t / 10000.0)], dim=-1)
        condition = self._condition_normalizer(
            condition,
            batch_size=x.shape[0],
            device=x.device,
            dtype=x.dtype,
        )
        inp = torch.cat([base, t_emb, condition], dim=-1)
        return self.backbone(inp).reshape(-1, self.horizon, self.state_dim)

    def forward(
        self,
        x: object,
        t: object,
        condition: dict[str, object] | None = None,
    ) -> torch.Tensor:
        return self.__call__(x, t, condition)

    def predict_numpy(
        self, x: np.ndarray, t: np.ndarray, condition: dict[str, object] | None = None
    ) -> np.ndarray:
        with torch.no_grad():
            x_t = torch.as_tensor(x, dtype=torch.float32)
            t_t = torch.as_tensor(t, dtype=torch.long)
            if condition:
                cond_values = np.asarray(next(iter(condition.values())), dtype=float)
                cond_t = torch.as_tensor(cond_values, dtype=torch.float32)
            else:
                cond_t = None
            return self.__call__(x_t, t_t, cond_t).cpu().numpy()

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    def parameters(self) -> Iterable[torch.nn.Parameter]:
        return self.backbone.parameters()

    def eval(self) -> DiffusionModel:
        self.backbone.eval()
        return self

    def load_state_dict(
        self,
        state_dict: dict[str, object],
        strict: bool = True,
    ) -> tuple[list[str], list[str]]:
        return self.backbone.load_state_dict(state_dict, strict=strict)

    def state_dict(self) -> dict[str, object]:
        return self.backbone.state_dict()

    def set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)


__all__ = ["DiffusionModel", "SimpleDiffusionModel"]
