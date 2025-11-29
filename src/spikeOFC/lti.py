"""Linear time-invariant (LTI) helpers.

Follows the expectations from `docs/DATA_STRUCTURES.md` and `docs/SIM_PLAN.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


Array = np.ndarray


@dataclass
class LTISystem:
    """Continuous-time linear system with additive Gaussian noises."""

    A: Array
    C: Array
    process_noise_cov: Array
    measurement_noise_cov: Array

    def step(self, x: Array, dt: float, rng: np.random.Generator) -> Array:
        """Advance the latent state by one Euler step."""
        noise = rng.multivariate_normal(
            mean=np.zeros(self.A.shape[0]),
            cov=self.process_noise_cov * dt,
        )
        x_next = x + dt * (self.A @ x) + noise
        return x_next

    def observe(self, x: Array, rng: np.random.Generator) -> Array:
        """Generate delayed/noisy sensor measurements."""
        meas_noise = rng.multivariate_normal(
            mean=np.zeros(self.C.shape[0]),
            cov=self.measurement_noise_cov,
        )
        return self.C @ x + meas_noise


def make_double_integrator(
    dt: float,
    sigma_process: float,
    sigma_measure: float,
) -> LTISystem:
    """Create a standard double integrator plant."""
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    C = np.array([[1.0, 0.0]])
    Q = (sigma_process ** 2) * np.eye(A.shape[0])
    R = (sigma_measure ** 2) * np.eye(C.shape[0])
    return LTISystem(A=A, C=C, process_noise_cov=Q, measurement_noise_cov=R)


def step(x: Array, A: Array, dt: float, noise: Optional[Array] = None) -> Array:
    """Functional helper mirroring the simple API stated in DATA_STRUCTURES."""
    residual = noise if noise is not None else 0.0
    return x + dt * (A @ x) + residual

