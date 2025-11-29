"""Logging helpers for innovation energy, MSE, and spike stats."""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def innovation_power(e: Array) -> float:
    return float(np.dot(e, e))


def state_mse(x_hat: Array, x_true: Array) -> float:
    diff = x_hat - x_true
    return float(np.dot(diff, diff))


def firing_rate(spikes: Array, dt: float) -> float:
    return float(np.sum(spikes) / (spikes.size * dt))

