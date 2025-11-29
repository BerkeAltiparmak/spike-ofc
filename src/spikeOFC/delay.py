"""Delay-line utilities for aligning prediction signals."""

from __future__ import annotations

from collections import deque
from typing import Deque

import numpy as np


class DelayLine:
    """Fixed-length FIFO storing past traces."""

    def __init__(self, size: int, tau_steps: int):
        self.size = size
        self.tau_steps = tau_steps
        self._buffer: Deque[np.ndarray] = deque(maxlen=tau_steps + 1)
        self.reset()

    def reset(self) -> None:
        """Clear the buffer and fill with zeros."""
        self._buffer.clear()
        zero = np.zeros(self.size)
        for _ in range(self.tau_steps + 1):
            self._buffer.append(zero.copy())

    def push(self, value: np.ndarray) -> None:
        """Insert a new trace sample."""
        if value.shape[0] != self.size:
            raise ValueError(f"Expected size {self.size}, got {value.shape}")
        self._buffer.append(value.copy())

    def read(self) -> np.ndarray:
        """Return r(t-τ)."""
        return self._buffer[0].copy()

