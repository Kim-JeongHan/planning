"""Diffusion noise schedule helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _cosine_betas(n_diffusion_steps: int, s: float = 0.008) -> np.ndarray:
    """Cosine noise schedule from Nichol & Dhariwal (2021).

    Computes betas such that alpha_bar follows a cosine curve:
        alpha_bar(t) = cos((t/T + s) / (1 + s) * pi/2)^2 / cos(s / (1+s) * pi/2)^2
    """
    t = np.linspace(0, n_diffusion_steps, n_diffusion_steps + 1, dtype=float)
    alpha_bar = np.cos((t / n_diffusion_steps + s) / (1.0 + s) * math.pi / 2.0) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return np.clip(betas, 0.0, 0.999)


@dataclass(frozen=True)
class DiffusionSchedule:
    """Noise schedule definition used by local sampling and training.

    Supports both linear (Ho et al. 2020) and cosine (Nichol & Dhariwal 2021) schedules.
    For cosine mode, ``beta_start`` / ``beta_end`` are ignored; betas are derived from
    the alpha_bar cosine curve.
    """

    beta_start: float
    beta_end: float
    n_diffusion_steps: int
    mode: str = "linear"
    # Precomputed betas are stored for cosine mode (not user-supplied).
    _precomputed_betas: tuple[float, ...] = field(default=(), compare=False)

    def __post_init__(self) -> None:
        if self.n_diffusion_steps <= 0:
            raise ValueError("n_diffusion_steps must be positive.")
        if self.mode == "linear" and self.beta_end <= self.beta_start:
            raise ValueError("beta_end should be greater than beta_start for linear schedule.")

    @property
    def betas(self) -> np.ndarray:
        if self.mode == "cosine" and self._precomputed_betas:
            return np.asarray(self._precomputed_betas, dtype=float)
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
    def cosine(
        cls,
        *,
        n_diffusion_steps: int = 100,
        s: float = 0.008,
    ) -> DiffusionSchedule:
        """Cosine noise schedule (Nichol & Dhariwal, 2021).

        The offset ``s`` prevents beta from being too small near t=0, which can
        cause the model to underestimate noise at the first step.
        """
        betas = _cosine_betas(n_diffusion_steps, s=s)
        return cls(
            beta_start=float(betas[0]),
            beta_end=float(betas[-1]),
            n_diffusion_steps=n_diffusion_steps,
            mode="cosine",
            _precomputed_betas=tuple(float(b) for b in betas),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> DiffusionSchedule:
        if not payload:
            return cls.cosine()
        mode = str(payload.get("mode", "cosine"))
        if mode == "cosine":
            return cls.cosine(n_diffusion_steps=int(payload.get("n_diffusion_steps", 100)))
        return cls(
            beta_start=float(payload.get("beta_start", 1e-4)),
            beta_end=float(payload.get("beta_end", 2e-2)),
            n_diffusion_steps=int(payload.get("n_diffusion_steps", 100)),
            mode="linear",
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
        alpha_bar = alpha_bar.reshape(-1, *([1] * (x_start.ndim - 1)))
        return np.sqrt(alpha_bar) * x_start + np.sqrt(1.0 - alpha_bar) * noise
