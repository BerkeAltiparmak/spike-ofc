"""Deterministic analytic baselines (e.g., discrete Kalman filter)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

Array = np.ndarray


@dataclass
class DiscreteKalman:
    """Standard time-varying discrete Kalman filter."""

    A: Array
    C: Array
    Q: Array
    R: Array
    x_hat: Array
    P: Array

    def step(self, y: Array) -> Tuple[Array, Array]:
        """Advance one step and return (state_estimate, innovation)."""
        x_pred = self.A @ self.x_hat
        P_pred = self.A @ self.P @ self.A.T + self.Q

        S = self.C @ P_pred @ self.C.T + self.R
        K = P_pred @ self.C.T @ np.linalg.inv(S)

        innovation = y - self.C @ x_pred
        self.x_hat = x_pred + K @ innovation
        self.P = (np.eye(self.P.shape[0]) - K @ self.C) @ P_pred

        return self.x_hat.copy(), innovation.copy()

