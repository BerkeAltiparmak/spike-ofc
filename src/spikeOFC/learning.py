"""Local three-factor learning rules."""

from __future__ import annotations

from typing import Optional

import numpy as np


Array = np.ndarray


def update_Wy(
    W_y: Array,
    e: Array,
    r_delay: Array,
    eta: float,
    decay: float = 0.0,
) -> Array:
    """Apply ΔW_y ∝ e rᵀ on the delayed trace."""
    W_y += eta * np.outer(e, r_delay)
    if decay > 0.0:
        W_y -= decay * W_y
    return W_y


def update_G(
    G: Array,
    Ge: Array,
    e: Array,
    eta: float,
    clip: Optional[float] = None,
) -> Array:
    """Apply ΔG ∝ (G e) eᵀ with optional norm clipping."""
    G += eta * np.outer(Ge, e)
    if clip is not None:
        row_norms = np.linalg.norm(G, axis=1, keepdims=True) + 1e-6
        G = np.where(
            row_norms > clip,
            G * (clip / row_norms),
            G,
        )
    return G


def update_Omega_s(
    Omega_s: Array,
    Ge: Array,
    r: Array,
    eta: float,
    symmetrize: bool = True,
) -> Array:
    """Apply ΔΩ_s ∝ (G e) rᵀ; optionally keep Ω_s symmetric."""
    Omega_s += eta * np.outer(Ge, r)
    if symmetrize:
        Omega_s = 0.5 * (Omega_s + Omega_s.T)
    return Omega_s

