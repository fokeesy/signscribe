"""One Euro filter: smooths jitter at rest, adds almost no lag when moving fast.

Reference: Casiez, Roussel & Vogel, "1 EUR Filter: A Simple Speed-based Low-pass Filter
for Noisy Input in Interactive Systems", CHI 2012.
"""

from __future__ import annotations

import math

import numpy as np


def _alpha(cutoff: np.ndarray | float, dt: float) -> np.ndarray | float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Vectorised One Euro filter for arrays of any shape (e.g. 21x3 landmarks).

    Args:
        min_cutoff: cutoff in Hz at rest. Lower = smoother but laggier when still.
        beta: how quickly the cutoff opens up with speed. Higher = less lag on fast moves.
        d_cutoff: cutoff for the derivative estimate.
        reset_gap: seconds without samples after which the filter restarts.
    """

    def __init__(
        self,
        min_cutoff: float = 2.0,
        beta: float = 20.0,
        d_cutoff: float = 1.5,
        reset_gap: float = 0.4,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset_gap = reset_gap
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None
        self._t: float | None = None

    def reset(self) -> None:
        self._x = None
        self._dx = None
        self._t = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._x is None or self._t is None or (t - self._t) > self.reset_gap or t <= self._t:
            self._x = x.copy()
            self._dx = np.zeros_like(x)
            self._t = t
            return x
        dt = max(t - self._t, 1e-3)
        dx = (x - self._x) / dt
        a_d = _alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1.0 - a_d) * self._dx
        speed = np.abs(self._dx)
        cutoff = self.min_cutoff + self.beta * speed
        a = _alpha(cutoff, dt)
        self._x = a * x + (1.0 - a) * self._x
        self._t = t
        return self._x.copy()


class ScalarEMA:
    """Tiny exponential moving average for scalars (angles, speeds...)."""

    def __init__(self, alpha: float = 0.3, value: float | None = None) -> None:
        self.alpha = alpha
        self.value = value

    def __call__(self, x: float) -> float:
        self.value = x if self.value is None else self.alpha * x + (1 - self.alpha) * self.value
        return self.value

    def reset(self) -> None:
        self.value = None
