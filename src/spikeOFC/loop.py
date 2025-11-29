"""Simulation loop wiring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from . import baselines, delay, learning, logging as log_utils, lti, spikeOFC_model


@dataclass
class SimulationConfig:
    dt: float
    T: float
    eta_wy: float
    eta_g: float
    eta_omega_s: float
    wy_decay: float = 0.0
    omega_symmetrize: bool = True
    threshold: float = 1.0
    v_reset: float = 0.0
    record_spikes: bool = False


@dataclass
class SimulationOutputs:
    logs: Dict[str, List[float]]
    state: spikeOFC_model.SpikeOFCState
    spike_history: Optional[np.ndarray] = None
    decoded_history: Optional[np.ndarray] = None
    true_history: Optional[np.ndarray] = None


def simulate(
    model: spikeOFC_model.SpikeOFCModel,
    plant: lti.LTISystem,
    estimator_state: spikeOFC_model.SpikeOFCState,
    delay_line: delay.DelayLine,
    x0: np.ndarray,
    rng: np.random.Generator,
    config: SimulationConfig,
    baseline: Optional[baselines.DiscreteKalman] = None,
) -> SimulationOutputs:
    """Run the predict → compare → correct loop."""
    steps = int(config.T / config.dt)
    x = x0.copy()
    logs: Dict[str, List[float]] = {
        "innovation": [],
        "mse": [],
        "firing_rate": [],
    }
    if baseline is not None:
        logs["kalman_mse"] = []
        logs["kalman_innovation"] = []
    spike_buffer: List[np.ndarray] | None = [] if config.record_spikes else None
    decode_buffer: List[np.ndarray] = []
    true_buffer: List[np.ndarray] = []
    delay_line.reset()
    delay_line.push(estimator_state.r)

    for _ in range(steps):
        x = plant.step(x, config.dt, rng)
        y = plant.observe(x, rng)
        true_buffer.append(x.copy())
        outputs = model.step(
            estimator_state,
            y,
            delay_line,
            dt=config.dt,
            threshold=config.threshold,
            v_reset=config.v_reset,
        )
        estimator_state = outputs["state"]
        e = outputs["e"]
        Ge = outputs["Ge"]
        r_delay = outputs["r_delay"]
        # Learning updates (in-place)
        learning.update_Wy(model.params.W_y, e, r_delay, config.eta_wy, config.wy_decay)
        learning.update_G(model.params.G, Ge, e, config.eta_g)
        learning.update_Omega_s(
            model.params.Omega_s,
            Ge,
            estimator_state.r,
            config.eta_omega_s,
            symmetrize=config.omega_symmetrize,
        )
        # Metrics
        x_hat = model.params.D @ estimator_state.r
        logs["innovation"].append(log_utils.innovation_power(e))
        logs["mse"].append(log_utils.state_mse(x_hat, x))
        decode_buffer.append(x_hat.copy())
        logs["firing_rate"].append(log_utils.firing_rate(estimator_state.s, config.dt))
        if spike_buffer is not None:
            spike_buffer.append(estimator_state.s.copy())
        if baseline is not None:
            x_kalman, innovation_kalman = baseline.step(y)
            logs["kalman_mse"].append(log_utils.state_mse(x_kalman, x))
            logs["kalman_innovation"].append(log_utils.innovation_power(innovation_kalman))

    spike_history = None
    if spike_buffer is not None and len(spike_buffer) > 0:
        spike_history = np.stack(spike_buffer, axis=0)
    decoded_history = np.stack(decode_buffer, axis=0) if decode_buffer else None
    true_history = np.stack(true_buffer, axis=0) if true_buffer else None

    return SimulationOutputs(
        logs=logs,
        state=estimator_state,
        spike_history=spike_history,
        decoded_history=decoded_history,
        true_history=true_history,
    )

