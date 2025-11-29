"""Core SCN-inspired spiking substrate utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


Array = np.ndarray


def init_decoder(K: int, N: int, rng: np.random.Generator) -> Array:
    """Initialize an overcomplete decoder with unit-norm columns."""
    D = rng.normal(size=(K, N))
    norms = np.linalg.norm(D, axis=0, keepdims=True) + 1e-8
    return D / norms


def fast_matrix(D: Array) -> Array:
    """Compute Ω_f = -DᵀD (competition / reset)."""
    return -D.T @ D


@dataclass
class SpikingState:
    """Minimal membrane + trace container."""

    v: Array
    r: Array


def spike_step(
    state: SpikingState,
    input_current: Array,
    dt: float,
    lambda_: float,
    threshold: float = 1.0,
    v_reset: float = 0.0,
) -> Tuple[Array, SpikingState]:
    """Single Euler step inspired by SCN dynamics (at most one spike per dt)."""
    dv = (-lambda_ * state.v + input_current) * dt
    v_new = state.v + dv
    spikes = np.zeros_like(state.v)
    above = np.where(v_new >= threshold)[0]
    if above.size > 0:
        spike_idx = above[np.argmax(v_new[above])]
        spikes[spike_idx] = 1.0 / dt  # Dirac-like pulse
        v_new[spike_idx] = v_reset
    dr = (-lambda_ * state.r + spikes) * dt
    r_new = state.r + dr
    return spikes, SpikingState(v=v_new, r=r_new)

