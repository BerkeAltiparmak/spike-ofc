"""Spike-OFC model glue code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from . import delay, scn_core


Array = np.ndarray


@dataclass
class SpikeOFCParams:
    """Container for learnable and fixed matrices."""

    D: Array
    Omega_f: Array
    Omega_s: Array
    W_y: Array
    G: Array
    tau_steps: int
    lambda_: float
    bias_current: Array
    innovation_gain: float


@dataclass
class SpikeOFCState:
    """Dynamic variables of the estimator population."""

    v: Array
    r: Array
    s: Array


def init_state(N: int, v_std: float = 0.0, rng: np.random.Generator | None = None) -> SpikeOFCState:
    if v_std > 0.0:
        rng = rng or np.random.default_rng()
        v0 = rng.normal(scale=v_std, size=N)
    else:
        v0 = np.zeros(N)
    r0 = np.zeros(N)
    s0 = np.zeros(N)
    return SpikeOFCState(v=v0, r=r0, s=s0)


class SpikeOFCModel:
    """High-level orchestrator for predict → compare → correct."""

    def __init__(self, params: SpikeOFCParams):
        self.params = params

    def predict_sensors(self, r_delay: Array) -> Array:
        return self.params.W_y @ r_delay

    @staticmethod
    def innovation(y: Array, y_hat: Array) -> Array:
        return y - y_hat

    def innovation_current(self, e: Array) -> Array:
        return self.params.G @ e

    def voltage_rhs(self, state: SpikeOFCState, innovation_drive: Array) -> Array:
        prediction_drive = self.params.Omega_s @ state.r
        fast_drive = self.params.Omega_f @ state.s
        return (
            prediction_drive
            + fast_drive
            + self.params.innovation_gain * innovation_drive
            + self.params.bias_current
        )

    def decode(self, state: SpikeOFCState) -> Array:
        return self.params.D @ state.r

    def step(
        self,
        state: SpikeOFCState,
        y: Array,
        delay_line: delay.DelayLine,
        dt: float,
        threshold: float = 1.0,
        v_reset: float = 0.0,
    ) -> Dict[str, Any]:
        r_delay = delay_line.read()
        y_hat = self.predict_sensors(r_delay)
        e = self.innovation(y, y_hat)
        Ge = self.innovation_current(e)
        total_input = self.voltage_rhs(state, Ge)
        spikes, new_spiking_state = scn_core.spike_step(
            scn_core.SpikingState(v=state.v, r=state.r),
            total_input,
            dt=dt,
            lambda_=self.params.lambda_,
            threshold=threshold,
            v_reset=v_reset,
        )
        next_state = SpikeOFCState(
            v=new_spiking_state.v,
            r=new_spiking_state.r,
            s=spikes,
        )
        delay_line.push(next_state.r)
        return {
            "state": next_state,
            "y_hat": y_hat,
            "e": e,
            "Ge": Ge,
            "r_delay": r_delay,
        }

