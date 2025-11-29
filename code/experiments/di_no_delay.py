"""Double integrator experiment without measurement delay."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.spikeOFC import baselines, config as cfg, delay, lti, loop, scn_core, spikeOFC_model


def _make_rng(seed: int):
    default_rng = getattr(np.random, "default_rng", None)
    if default_rng is None:
        return np.random.RandomState(seed)
    return default_rng(seed)


def _stationary_kalman_gain(
    A_d: np.ndarray, C: np.ndarray, Q_d: np.ndarray, R: np.ndarray, steps: int = 2000
) -> np.ndarray:
    """Iteratively converge to the steady-state Kalman gain."""
    K = np.zeros((A_d.shape[0], C.shape[0]))
    P = np.eye(A_d.shape[0])
    for _ in range(steps):
        S = C @ P @ C.T + R
        K = P @ C.T @ np.linalg.inv(S)
        P = A_d @ P @ A_d.T + Q_d - A_d @ P @ C.T @ np.linalg.inv(S) @ C @ P @ A_d.T
    return K


def build_components(run_cfg: cfg.RunConfig):
    rng = _make_rng(run_cfg.seed)
    if run_cfg.decoder_basis == "identity":
        if run_cfg.N < run_cfg.K:
            raise ValueError("identity decoder basis requires N >= K")
        D = np.zeros((run_cfg.K, run_cfg.N))
        np.fill_diagonal(D[:, : run_cfg.K], run_cfg.decoder_scale)
    else:
        D = run_cfg.decoder_scale * scn_core.init_decoder(run_cfg.K, run_cfg.N, rng)
    if run_cfg.decoder_scales:
        scales = np.fromstring(run_cfg.decoder_scales, sep=",")
        if scales.size != run_cfg.K:
            raise ValueError(f"Expected {run_cfg.K} decoder scales, got {scales.size}")
        D = D * scales[:, None]
    if run_cfg.bounding_box > 0:
        D = D / run_cfg.bounding_box
    Omega_f = scn_core.fast_matrix(D)
    plant = lti.make_double_integrator(
        dt=run_cfg.dt,
        sigma_process=0.05,
        sigma_measure=0.05,
    )
    tau_steps = max(0, int(round(run_cfg.tau / run_cfg.dt)))

    predictor_matrix = plant.A + run_cfg.lambda_decay * np.eye(run_cfg.K)
    Omega_s = run_cfg.omega_scale * (D.T @ predictor_matrix @ D)

    W_y = run_cfg.init_wy_scale * rng.standard_normal((run_cfg.Q, run_cfg.N))
    G = run_cfg.init_g_scale * rng.standard_normal((run_cfg.N, run_cfg.Q))

    baseline = None
    A_d = np.eye(run_cfg.K) + run_cfg.dt * plant.A
    Q_d = plant.process_noise_cov * run_cfg.dt
    Kf = None
    if run_cfg.use_kalman:
        Kf = _stationary_kalman_gain(A_d, plant.C, Q_d, plant.measurement_noise_cov)
        baseline = baselines.DiscreteKalman(
            A=A_d,
            C=plant.C,
            Q=Q_d,
            R=plant.measurement_noise_cov,
            x_hat=np.zeros(run_cfg.K),
            P=np.eye(run_cfg.K),
        )

    if run_cfg.teacher_forced and Kf is not None:
        W_y = plant.C @ D
        G = D.T @ Kf

    params = spikeOFC_model.SpikeOFCParams(
        D=D,
        Omega_f=Omega_f,
        Omega_s=Omega_s,
        W_y=W_y,
        G=G,
        tau_steps=tau_steps,
        lambda_=run_cfg.lambda_decay,
        bias_current=np.full(run_cfg.N, run_cfg.bias_current),
        innovation_gain=run_cfg.innovation_gain,
    )
    model = spikeOFC_model.SpikeOFCModel(params)
    state = spikeOFC_model.init_state(run_cfg.N, v_std=run_cfg.init_v_std, rng=rng)
    delay_line = delay.DelayLine(size=run_cfg.N, tau_steps=tau_steps)
    return model, state, delay_line, plant, rng, baseline


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


def _ordered_metrics(logs: dict[str, list[float]]) -> List[str]:
    base = ["innovation", "mse", "firing_rate"]
    if "r_norm" in logs:
        base.append("r_norm")
    if "Ge_norm" in logs:
        base.append("Ge_norm")
    extras = []
    if "kalman_innovation" in logs:
        extras.append("kalman_innovation")
    if "kalman_mse" in logs:
        extras.append("kalman_mse")
    return base + extras


def _write_logs(logs: dict[str, list[float]], dt: float, out_dir: Path) -> None:
    metric_keys = _ordered_metrics(logs)
    times = np.arange(len(logs["innovation"])) * dt
    with (out_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t"] + metric_keys)
        for idx, t in enumerate(times):
            row = [f"{t:.6f}"]
            for key in metric_keys:
                row.append(f"{logs[key][idx]:.6e}")
            writer.writerow(row)


def _write_traces(decoded: Optional[np.ndarray], true: Optional[np.ndarray], out_dir: Path) -> None:
    if decoded is None or true is None:
        return
    np.savez(out_dir / "state_traces.npz", x_hat=decoded, x_true=true)


def _write_param_stats(model: spikeOFC_model.SpikeOFCModel, out_dir: Path) -> None:
    stats = {
        "Wy_fro": float(np.linalg.norm(model.params.W_y)),
        "G_fro": float(np.linalg.norm(model.params.G)),
        "Omega_s_fro": float(np.linalg.norm(model.params.Omega_s)),
    }
    with (out_dir / "param_stats.json").open("w") as f:
        json.dump(stats, f, indent=2)


def _plot_metrics(
    logs: dict[str, list[float]],
    dt: float,
    spike_history: Optional[np.ndarray],
    out_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    times = np.arange(len(logs["innovation"])) * dt
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(times, logs["innovation"], label="Spike-OFC ||e||^2")
    if "kalman_innovation" in logs:
        axes[0].plot(times, logs["kalman_innovation"], label="Kalman ||e||^2", linestyle="--")
    axes[0].set_ylabel("Innovation energy")
    axes[0].legend()
    axes[1].plot(times, logs["mse"], color="tab:orange", label="Spike-OFC ||x̂ - x||^2")
    if "kalman_mse" in logs:
        axes[1].plot(times, logs["kalman_mse"], color="tab:green", linestyle="--", label="Kalman ||x̂ - x||^2")
    axes[1].set_ylabel("State MSE")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "metrics.png", dpi=150)
    plt.close(fig)

    if spike_history is not None:
        fig, ax = plt.subplots(figsize=(9, 4))
        extent = [0, times[-1] if len(times) else 0, 0, spike_history.shape[1]]
        ax.imshow(
            spike_history.T,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=extent,
        )
        ax.set_ylabel("Neuron index")
        ax.set_xlabel("Time (s)")
        ax.set_title("Spike raster")
        fig.tight_layout()
        fig.savefig(out_dir / "spikes.png", dpi=150)
        plt.close(fig)


def main():
    run_cfg = cfg.parse_args()
    model, state, delay_line, plant, rng, baseline = build_components(run_cfg)
    out_dir = _prepare_run_dir(run_cfg)
    _write_config(run_cfg, out_dir)
    eta_wy = 0.0 if run_cfg.teacher_forced else run_cfg.eta_wy
    eta_g = 0.0 if run_cfg.teacher_forced else run_cfg.eta_g
    eta_omega_s = 0.0 if run_cfg.teacher_forced else run_cfg.eta_omega_s
    sim_cfg = loop.SimulationConfig(
        dt=run_cfg.dt,
        T=run_cfg.T,
        eta_wy=eta_wy,
        eta_g=eta_g,
        eta_omega_s=eta_omega_s,
        record_spikes=run_cfg.record_spikes or run_cfg.make_plots,
        threshold=run_cfg.threshold,
        v_reset=run_cfg.v_reset,
    )
    outputs = loop.simulate(
        model=model,
        plant=plant,
        estimator_state=state,
        delay_line=delay_line,
        x0=np.zeros(run_cfg.K),
        rng=rng,
        config=sim_cfg,
        baseline=baseline,
    )
    logs = outputs.logs
    _write_logs(logs, run_cfg.dt, out_dir)
    _write_traces(outputs.decoded_history, outputs.true_history, out_dir)
    _write_param_stats(model, out_dir)
    if run_cfg.make_plots:
        _plot_metrics(logs, run_cfg.dt, outputs.spike_history, out_dir)
    print("Simulation completed.")
    print(f"{'Spike-OFC innovation (final):':35s} {logs['innovation'][-1]:.4e}")
    print(f"{'Spike-OFC MSE (final):':35s} {logs['mse'][-1]:.4e}")
    if "kalman_innovation" in logs:
        print(f"{'Kalman innovation (final):':35s} {logs['kalman_innovation'][-1]:.4e}")
    if "kalman_mse" in logs:
        print(f"{'Kalman MSE (final):':35s} {logs['kalman_mse'][-1]:.4e}")
    print(f"{'Average firing rate:':35s} {np.mean(logs['firing_rate']):.4e} Hz")
    print(f"Artifacts written to: {out_dir}")


if __name__ == "__main__":
    main()

