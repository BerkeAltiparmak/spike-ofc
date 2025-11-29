"""Double integrator experiment without measurement delay."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from src.spikeOFC import config as cfg
from src.spikeOFC import delay, lti, loop, scn_core, spikeOFC_model


def _make_rng(seed: int):
    default_rng = getattr(np.random, "default_rng", None)
    if default_rng is None:
        return np.random.RandomState(seed)
    return default_rng(seed)


def build_components(run_cfg: cfg.RunConfig):
    rng = _make_rng(run_cfg.seed)
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


def _prepare_run_dir(run_cfg: cfg.RunConfig) -> Path:
    root = Path(run_cfg.run_dir)
    root.mkdir(parents=True, exist_ok=True)
    tag = run_cfg.tag or run_cfg.experiment
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = root / f"{tag}_{timestamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_config(run_cfg: cfg.RunConfig, out_dir: Path) -> None:
    with (out_dir / "config.json").open("w") as f:
        json.dump(asdict(run_cfg), f, indent=2)


def _write_logs(logs: dict[str, list[float]], dt: float, out_dir: Path) -> None:
    times = np.arange(len(logs["innovation"])) * dt
    with (out_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "innovation", "mse", "firing_rate"])
        for idx, t in enumerate(times):
            writer.writerow(
                [
                    f"{t:.6f}",
                    f"{logs['innovation'][idx]:.6e}",
                    f"{logs['mse'][idx]:.6e}",
                    f"{logs['firing_rate'][idx]:.6e}",
                ]
            )


def _plot_metrics(
    logs: dict[str, list[float]],
    dt: float,
    spike_history: Optional[np.ndarray],
    out_dir: Path,
) -> None:
    times = np.arange(len(logs["innovation"])) * dt
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(times, logs["innovation"], label="||e||^2")
    axes[0].set_ylabel("Innovation energy")
    axes[0].legend()
    axes[1].plot(times, logs["mse"], color="tab:orange", label="||x̂ - x||^2")
    axes[1].set_ylabel("State MSE")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "metrics.png", dpi=150)
    plt.close(fig)

    if spike_history is not None:
        fig, ax = plt.subplots(figsize=(9, 4))
        extent = [times[0], times[-1] if len(times) else 0, 0, spike_history.shape[1]]
        ax.imshow(
            spike_history.T,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=[0, times[-1] if len(times) else 0, 0, spike_history.shape[1]],
        )
        ax.set_ylabel("Neuron index")
        ax.set_xlabel("Time (s)")
        ax.set_title("Spike raster")
        fig.tight_layout()
        fig.savefig(out_dir / "spikes.png", dpi=150)
        plt.close(fig)


def main():
    run_cfg = cfg.parse_args()
    model, state, delay_line, plant, rng = build_components(run_cfg)
    out_dir = _prepare_run_dir(run_cfg)
    _write_config(run_cfg, out_dir)
    sim_cfg = loop.SimulationConfig(
        dt=run_cfg.dt,
        T=run_cfg.T,
        eta_wy=run_cfg.eta_wy,
        eta_g=run_cfg.eta_g,
        eta_omega_s=run_cfg.eta_omega_s,
        record_spikes=run_cfg.record_spikes or run_cfg.make_plots,
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
    _write_logs(logs, run_cfg.dt, out_dir)
    if run_cfg.make_plots:
        _plot_metrics(logs, run_cfg.dt, outputs.spike_history, out_dir)
    print("Simulation completed.")
    print(f"Innovation power (final): {logs['innovation'][-1]:.4e}")
    print(f"State MSE (final): {logs['mse'][-1]:.4e}")
    print(f"Average firing rate: {np.mean(logs['firing_rate']):.4e} Hz")
    print(f"Artifacts written to: {out_dir}")


if __name__ == "__main__":
    main()

