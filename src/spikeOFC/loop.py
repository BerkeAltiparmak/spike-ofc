"""Simulation loop wiring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from . import delay, learning, logging as log_utils, lti, spikeOFC_model


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


def simulate(
    model: spikeOFC_model.SpikeOFCModel,
    plant: lti.LTISystem,
    estimator_state: spikeOFC_model.SpikeOFCState,
    delay_line: delay.DelayLine,
    x0: np.ndarray,
    rng: np.random.Generator,
    config: SimulationConfig,
) -> Dict[str, List[float]]:
    """Run the predict → compare → correct loop."""
    steps = int(config.T / config.dt)
    x = x0.copy()
    logs: Dict[str, List[float]] = {
        "innovation": [],
        "mse": [],
        "firing_rate": [],
    }
    delay_line.reset()
    delay_line.push(estimator_state.r)

    for _ in range(steps):
        x = plant.step(x, config.dt, rng)
        y = plant.observe(x, rng)
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
        logs["firing_rate"].append(log_utils.firing_rate(estimator_state.s, config.dt))

    return logs

