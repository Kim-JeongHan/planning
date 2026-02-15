"""Diffusion noise schedule helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DiffusionSchedule:
    """Minimal noise schedule definition used by local sampling and training."""

    beta_start: float
    beta_end: float
    n_diffusion_steps: int
    mode: str = "linear"

    def __post_init__(self) -> None:
        if self.n_diffusion_steps <= 0:
            raise ValueError("n_diffusion_steps must be positive.")
        if self.beta_end <= self.beta_start:
            raise ValueError("beta_end should be greater than beta_start for linear schedule.")

    @property
    def betas(self) -> np.ndarray:
        return np.linspace(self.beta_start, self.beta_end, self.n_diffusion_steps, dtype=float)

    @property
    def alpha(self) -> np.ndarray:
        return 1.0 - self.betas

    @property
    def alpha_bar(self) -> np.ndarray:
        return np.cumprod(self.alpha, axis=0)

    @property
    def posterior_variance(self) -> np.ndarray:
        one = np.ones_like(self.betas)
        shifted = np.roll(self.alpha_bar, 1, axis=0)
        shifted[0] = 1.0
        return np.clip(
            self.betas * (1.0 - shifted) / np.maximum(one - self.alpha_bar, 1e-8),
            1e-12,
            None,
        )

    @classmethod
    def linear(
        cls,
        *,
        n_diffusion_steps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ) -> DiffusionSchedule:
        return cls(beta_start=beta_start, beta_end=beta_end, n_diffusion_steps=n_diffusion_steps)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> DiffusionSchedule:
        if not payload:
            return cls.linear()
        return cls(
            beta_start=float(payload.get("beta_start", 1e-4)),
            beta_end=float(payload.get("beta_end", 2e-2)),
            n_diffusion_steps=int(payload.get("n_diffusion_steps", 100)),
            mode=str(payload.get("mode", "linear")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "n_diffusion_steps": self.n_diffusion_steps,
            "mode": self.mode,
        }

    def beta(self, t: int) -> float:
        return float(self.betas[int(t)])

    def sample_t(self, batch_size: int, rng: np.random.Generator | None = None) -> np.ndarray:
        random = np.random.default_rng() if rng is None else rng
        return random.integers(0, self.n_diffusion_steps, size=int(batch_size))

    def q_sample(self, x_start: np.ndarray, t: np.ndarray, noise: np.ndarray | None = None) -> np.ndarray:
        t_idx = np.asarray(t, dtype=int).reshape(-1)
        if noise is None:
            noise = np.random.normal(size=x_start.shape)
        if noise.shape != x_start.shape:
            noise = np.broadcast_to(noise, x_start.shape)
        alpha_bar = self.alpha_bar[t_idx]
        while alpha_bar.ndim < x_start.ndim:
            alpha_bar = alpha_bar[(...,) + (None,) * (x_start.ndim - 1)]
        return np.sqrt(alpha_bar) * x_start + np.sqrt(1.0 - alpha_bar) * noise
