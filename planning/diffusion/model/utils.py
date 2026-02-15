"""Common neural-network helpers for diffusion/value models."""

from __future__ import annotations

from collections.abc import Iterable

import torch  # type: ignore
import torch.nn as nn  # type: ignore


class ConditionNormalizer:
    """Normalize and project condition tensors to a fixed dimension."""

    def __init__(self, condition_dim: int) -> None:
        self.condition_dim = int(condition_dim)

    def __call__(
        self,
        condition: object,
        *,
        batch_size: int,
        device: object,
        dtype: object,
    ) -> torch.Tensor:
        """Normalize condition features to ``[batch_size, condition_dim]``."""
        if self.condition_dim <= 0:
            return torch.zeros((batch_size, 0), device=device, dtype=dtype)

        if condition is None:
            return torch.zeros((batch_size, self.condition_dim), device=device, dtype=dtype)

        if not torch.is_tensor(condition):
            condition_tensor = torch.as_tensor(condition, device=device, dtype=dtype)
        else:
            condition_tensor = condition.to(device=device, dtype=dtype)

        if condition_tensor.ndim == 1:
            condition_tensor = condition_tensor.reshape(1, -1)
        elif condition_tensor.ndim != 2:
            condition_tensor = condition_tensor.reshape(condition_tensor.shape[0], -1)

        if condition_tensor.shape[0] == 1:
            condition_tensor = condition_tensor.expand(batch_size, -1)
        elif condition_tensor.shape[0] != batch_size:
            raise ValueError("Condition batch size must match observation batch size.")

        if condition_tensor.shape[1] >= self.condition_dim:
            return condition_tensor[:, : self.condition_dim]

        pad = torch.zeros(
            (batch_size, self.condition_dim - condition_tensor.shape[1]),
            device=device,
            dtype=dtype,
        )
        return torch.cat([condition_tensor, pad], dim=-1)


class MLPBackbone:
    """Simple MLP backbone used by both diffusion and value models."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: int = 256,
        n_layers: int = 2,
    ) -> None:
        layers: list[torch.nn.Module] = []
        in_dim = int(input_dim)
        width = int(hidden_dim)
        for _ in range(max(1, int(n_layers))):
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.SiLU())
            in_dim = width
        layers.append(nn.Linear(in_dim, int(output_dim)))
        self.backbone = nn.Sequential(*layers)

    def to(self, *args: object, **kwargs: object) -> MLPBackbone:
        """Mirror core nn.Module API used by checkpoints and scripts."""
        self.backbone = self.backbone.to(*args, **kwargs)
        return self

    def __call__(self, x: object) -> torch.Tensor:
        return self.backbone(x)

    def parameters(self) -> Iterable[torch.nn.Parameter]:
        return self.backbone.parameters()

    def state_dict(self) -> dict[str, object]:
        return self.backbone.state_dict()

    def load_state_dict(
        self, state_dict: dict[str, object], strict: bool = True
    ) -> tuple[list[str], list[str]]:
        return self.backbone.load_state_dict(state_dict, strict=strict)

    def eval(self) -> MLPBackbone:
        self.backbone.eval()
        return self

    def __repr__(self) -> str:
        return f"MLPBackbone(input_dim={self.backbone[0].in_features}, output_dim={self.backbone[-1].out_features})"


__all__ = [
    "ConditionNormalizer",
    "MLPBackbone",
]
