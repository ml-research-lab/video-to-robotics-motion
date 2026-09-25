"""Speed-adaptive smoothing for noisy hand-tracking signals.

A fixed exponential filter is a bad fit here: loose enough to follow a
deliberate move and it lets a motionless hand's landmark jitter through;
tight enough to kill that jitter and it lags every real movement. The One
Euro filter (Casiez, Roussel & Vogel, 2012) fixes this by raising its own
cutoff frequency with the signal's estimated speed, so a still hand gets
heavy smoothing and a moving one gets almost none.
"""

from __future__ import annotations

import math

import numpy as np


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
    return 1.0 / (1.0 + tau / max(dt, 1e-6))


class OneEuroFilter:
    """One Euro filter over a fixed-length vector (or scalar)."""

    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = np.atleast_1d(np.asarray(min_cutoff, dtype=float))
        self.beta = np.atleast_1d(np.asarray(beta, dtype=float))
        self.d_cutoff = float(d_cutoff)
        self._value: np.ndarray | None = None
        self._derivative: np.ndarray | None = None

    def reset(self) -> None:
        self._value = None
        self._derivative = None

    @property
    def value(self) -> np.ndarray | None:
        return None if self._value is None else self._value.copy()

    def __call__(self, x, dt: float) -> np.ndarray:
        x = np.atleast_1d(np.asarray(x, dtype=float))
        if self._value is None:
            self._value = x.copy()
            self._derivative = np.zeros_like(x)
            return self._value.copy()

        dt = max(float(dt), 1e-6)
        derivative = (x - self._value) / dt
        a_d = _alpha(self.d_cutoff, dt)
        self._derivative = a_d * derivative + (1 - a_d) * self._derivative

        cutoff = self.min_cutoff + self.beta * np.abs(self._derivative)
        alpha = np.array([_alpha(float(c), dt) for c in np.broadcast_to(cutoff, x.shape)])
        self._value = alpha * x + (1 - alpha) * self._value
        return self._value.copy()


class AngleFilter:
    """A One Euro filter for an angle, unwrapped across the +/-pi branch cut."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0) -> None:
        self._filter = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self._unwrapped: float | None = None

    def reset(self) -> None:
        self._filter.reset()
        self._unwrapped = None

    def __call__(self, angle: float, dt: float) -> float:
        angle = float(angle)
        if self._unwrapped is None:
            self._unwrapped = angle
        else:
            diff = (angle - self._unwrapped + math.pi) % (2 * math.pi) - math.pi
            self._unwrapped += diff
        filtered = float(self._filter(self._unwrapped, dt)[0])
        return float((filtered + math.pi) % (2 * math.pi) - math.pi)


class RateLimiter:
    """Caps how fast a vector of commands may change, per control step."""

    def __init__(self, max_speed) -> None:
        self.max_speed = np.atleast_1d(np.asarray(max_speed, dtype=float))
        self._value: np.ndarray | None = None

    def reset(self, value=None) -> None:
        self._value = None if value is None else np.asarray(value, dtype=float).copy()

    def __call__(self, command, dt: float) -> np.ndarray:
        command = np.asarray(command, dtype=float)
        if self._value is None:
            self._value = command.copy()
            return self._value.copy()
        limit = self.max_speed * max(float(dt), 1e-6)
        step = np.clip(command - self._value, -limit, limit)
        self._value = self._value + step
        return self._value.copy()
