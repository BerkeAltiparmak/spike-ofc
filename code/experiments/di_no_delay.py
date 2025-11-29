"""Double integrator experiment without measurement delay."""

from __future__ import annotations

import numpy as np

from spikeOFC import config as cfg
from spikeOFC import delay, lti, loop, scn_core, spikeOFC_model


def build_components(run_cfg: cfg.RunConfig):
    rng = np.random.default_rng(run_cfg.seed)
    D = scn_core.init_decoder(run_cfg.K, run_cfg.N, rng)
    Omega_f = scn_core.fast_matrix(D)
    Omega_s = np.zeros((run_cfg.N, run_cfg.N))
    W_y = 0.01 * rng.standard_normal((run_cfg.Q, run_cfg.N))
    G = 0.01 * rng.standard_normal((run_cfg.N, run_cfg.Q))
    tau_steps = max(0, int(round(run_cfg.tau / run_cfg.dt)))
    params = spikeOFC_model.SpikeOFCParams(
        D=D,
        Omega_f=Omega_f,
        Omega_s=Omega_s,
        W_y=W_y,
        G=G,
        tau_steps=tau_steps,
        lambda_=1.0,
    )
    model = spikeOFC_model.SpikeOFCModel(params)
    state = spikeOFC_model.init_state(run_cfg.N)
    delay_line = delay.DelayLine(size=run_cfg.N, tau_steps=tau_steps)
    plant = lti.make_double_integrator(
        dt=run_cfg.dt,
        sigma_process=0.05,
        sigma_measure=0.05,
    )
    return model, state, delay_line, plant, rng


def main():
    run_cfg = cfg.parse_args()
    model, state, delay_line, plant, rng = build_components(run_cfg)
    sim_cfg = loop.SimulationConfig(
        dt=run_cfg.dt,
        T=run_cfg.T,
        eta_wy=run_cfg.eta_wy,
        eta_g=run_cfg.eta_g,
        eta_omega_s=run_cfg.eta_omega_s,
    )
    outputs = loop.simulate(
        model=model,
        plant=plant,
        estimator_state=state,
        delay_line=delay_line,
        x0=np.zeros(run_cfg.K),
        rng=rng,
        config=sim_cfg,
    )
    logs = outputs.logs
    print("Simulation completed.")
    print(f"Innovation power (final): {logs['innovation'][-1]:.4e}")
    print(f"State MSE (final): {logs['mse'][-1]:.4e}")
    print(f"Average firing rate: {np.mean(logs['firing_rate']):.4e} Hz")


if __name__ == "__main__":
    main()

