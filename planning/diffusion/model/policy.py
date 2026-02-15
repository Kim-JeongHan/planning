"""Policy/value model definitions."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from .utils import ConditionNormalizer, MLPBackbone


class SimpleValueModel:
    """MLP baseline value model for optional guidance objective."""

    def __new__(cls, *args: object, **kwargs: object) -> SimpleValueModel:
        return super().__new__(cls)

    @staticmethod
    def create(
        *,
        state_dim: int,
        horizon: int,
        n_hidden: int = 256,
        n_layers: int = 2,
        condition_dim: int = 0,
        _n_diffusion_steps: int | None = None,
        **_: object,
    ) -> ValueModel:
        del _n_diffusion_steps
        input_dim = horizon * state_dim + condition_dim
        output_dim = 1
        model = MLPBackbone(
            input_dim,
            output_dim,
            hidden_dim=n_hidden,
            n_layers=n_layers,
        )
        return ValueModel(
            model,
            state_dim=state_dim,
            horizon=horizon,
            n_hidden=n_hidden,
            n_layers=n_layers,
            condition_dim=condition_dim,
        )


class ValueModel:
    """Torch wrapper for value model."""

    def __init__(
        self,
        backbone: MLPBackbone,
        *,
        state_dim: int,
        horizon: int,
        n_hidden: int,
        n_layers: int,
        condition_dim: int = 0,
    ) -> None:
        self.backbone = backbone
        self.state_dim = int(state_dim)
        self.horizon = int(horizon)
        self.n_hidden = int(n_hidden)
        self.n_layers = int(n_layers)
        self.condition_dim = int(condition_dim)
        self._condition_normalizer = ConditionNormalizer(self.condition_dim)
        self.hparams = {
            "state_dim": self.state_dim,
            "horizon": self.horizon,
            "n_hidden": self.n_hidden,
            "n_layers": self.n_layers,
            "condition_dim": self.condition_dim,
            "model_class_path": "planning.diffusion.model.ValueModel",
        }

    def __call__(
        self,
        state: object,
        condition: dict[str, object] | None = None,
    ) -> torch.Tensor:
        flat_state = state.reshape(state.shape[0], -1)
        condition = self._condition_normalizer(
            condition,
            batch_size=state.shape[0],
            device=state.device,
            dtype=state.dtype,
        )
        inp = torch.cat([flat_state, condition], dim=-1)
        if inp.shape[1] != self.backbone.backbone[0].in_features:
            target = self.backbone.backbone[0].in_features
            if inp.shape[1] > target:
                inp = inp[:, :target]
            else:
                pad = torch.zeros((inp.shape[0], target - inp.shape[1]), dtype=inp.dtype, device=inp.device)
                inp = torch.cat([inp, pad], dim=-1)
        return self.backbone(inp)

    def forward(
        self,
        state: object,
        condition: dict[str, object] | None = None,
    ) -> torch.Tensor:
        return self.__call__(state, condition)

    def eval(self) -> ValueModel:
        self.backbone.eval()
        return self

    def load_state_dict(
        self, state_dict: dict[str, object], strict: bool = True
    ) -> tuple[list[str], list[str]]:
        return self.backbone.load_state_dict(state_dict, strict=strict)

    def state_dict(self) -> dict[str, object]:
        return self.backbone.state_dict()

    def parameters(self) -> Iterable[torch.nn.Parameter]:
        return self.backbone.parameters()

    def set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)


__all__ = ["SimpleValueModel", "ValueModel"]
