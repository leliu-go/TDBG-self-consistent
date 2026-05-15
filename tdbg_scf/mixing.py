"""Mixing algorithms for fixed-point self-consistency U = F(U)."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


class Mixer:
    def update(self, x: np.ndarray, fx: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass
class LinearMixer(Mixer):
    alpha: float = 0.08

    def update(self, x: np.ndarray, fx: np.ndarray) -> np.ndarray:
        y = (1.0 - self.alpha) * x + self.alpha * fx
        return y - np.mean(y)


@dataclass
class AndersonMixer(Mixer):
    beta: float = 0.5
    memory: int = 6
    fallback_alpha: float = 0.05
    xs: list[np.ndarray] = field(default_factory=list)
    fs: list[np.ndarray] = field(default_factory=list)

    def update(self, x: np.ndarray, fx: np.ndarray) -> np.ndarray:
        r = fx - x
        self.xs.append(np.array(x, dtype=float))
        self.fs.append(np.array(r, dtype=float))
        if len(self.xs) > self.memory + 1:
            self.xs.pop(0)
            self.fs.pop(0)

        if len(self.fs) < 2:
            y = x + self.fallback_alpha * r
            return y - np.mean(y)

        # Anderson type-II on residual differences.
        F = np.array(self.fs)
        X = np.array(self.xs)
        dF = (F[1:] - F[:-1]).T
        dX = (X[1:] - X[:-1]).T
        try:
            gamma, *_ = np.linalg.lstsq(dF, r, rcond=None)
            step = r - (dX + self.beta * dF) @ gamma
            y = x + self.beta * step
        except np.linalg.LinAlgError:
            y = x + self.fallback_alpha * r
        return y - np.mean(y)


def make_mixer(kind: str = "anderson", **kwargs) -> Mixer:
    kind = kind.lower()
    if kind in ("linear", "simple"):
        return LinearMixer(**kwargs)
    if kind in ("anderson", "andreson"):
        return AndersonMixer(**kwargs)
    raise ValueError(f"Unknown mixer kind: {kind}")
