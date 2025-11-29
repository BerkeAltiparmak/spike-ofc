"""Config helpers and CLI argument parsing."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass
class RunConfig:
    experiment: str
    K: int
    Q: int
    N: int
    dt: float
    T: float
    tau: float
    seed: int
    eta_wy: float
    eta_g: float
    eta_omega_s: float
    run_dir: str
    tag: Optional[str]
    make_plots: bool
    record_spikes: bool
    use_kalman: bool
    threshold: float
    v_reset: float
    bias_current: float
    innovation_gain: float
    init_v_std: float
    init_wy_scale: float
    init_g_scale: float
    teacher_forced: bool
    lambda_decay: float
    omega_scale: float
    decoder_scale: float
    decoder_scales: Optional[str]
    decoder_basis: str
    bounding_box: float


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Spike-OFC experiments.")
    parser.add_argument("--experiment", default="di_no_delay")
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--Q", type=int, default=1)
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eta_wy", type=float, default=1e-3)
    parser.add_argument("--eta_g", type=float, default=1e-3)
    parser.add_argument("--eta_omega_s", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.05, help="Spiking threshold.")
    parser.add_argument("--v_reset", type=float, default=0.0, help="Voltage reset value after spikes.")
    parser.add_argument("--bias-current", type=float, default=0.2, help="Constant bias injected into estimator dendrites.")
    parser.add_argument("--innovation-gain", type=float, default=10.0, help="Gain applied to innovation current Ge.")
    parser.add_argument("--init-v-std", type=float, default=0.05, help="Stddev for initial voltages to break symmetry.")
    parser.add_argument("--init-wy-scale", type=float, default=0.1, help="Stddev multiplier for Wy init.")
    parser.add_argument("--init-g-scale", type=float, default=0.1, help="Stddev multiplier for G init.")
    parser.add_argument("--teacher-forced", action="store_true", help="Use analytic Wy/G (no learning) to check the SCN substrate.")
    parser.add_argument("--lambda-decay", type=float, default=1.0, help="Membrane leak constant λ.")
    parser.add_argument("--omega-scale", type=float, default=1.0, help="Global multiplier on Ω_s.")
    parser.add_argument("--decoder-scale", type=float, default=1.0, help="Scale applied to decoder D (controls code variance).")
    parser.add_argument(
        "--decoder-scales",
        type=str,
        default=None,
        help="Comma-separated per-dimension scales applied row-wise to D (length K).",
    )
    parser.add_argument(
        "--decoder-basis",
        choices=["random", "identity"],
        default="random",
        help="Decoder basis initialization (identity requires N>=K).",
    )
    parser.add_argument(
        "--bounding-box",
        type=float,
        default=10.0,
        help="Divide decoder columns by this factor (mirrors SCN bounding box).",
    )
    parser.add_argument("--run-dir", type=str, default="runs", help="Directory to store logs/plots.")
    parser.add_argument("--tag", type=str, default=None, help="Optional tag for the run folder.")
    parser.add_argument("--no-plots", action="store_true", help="Disable matplotlib plots.")
    parser.add_argument(
        "--record-spikes",
        action="store_true",
        help="Store spike rasters even if plots are disabled.",
    )
    parser.add_argument(
        "--no-kalman",
        action="store_true",
        help="Disable the analytic Kalman baseline comparison.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> RunConfig:
    args = build_arg_parser().parse_args(argv)
    return RunConfig(
        experiment=args.experiment,
        K=args.K,
        Q=args.Q,
        N=args.N,
        dt=args.dt,
        T=args.T,
        tau=args.tau,
        seed=args.seed,
        eta_wy=args.eta_wy,
        eta_g=args.eta_g,
        eta_omega_s=args.eta_omega_s,
        run_dir=args.run_dir,
        tag=args.tag,
        make_plots=not args.no_plots,
        record_spikes=args.record_spikes,
        use_kalman=not args.no_kalman,
        threshold=args.threshold,
        v_reset=args.v_reset,
        bias_current=args.bias_current,
        innovation_gain=args.innovation_gain,
        init_v_std=args.init_v_std,
        init_wy_scale=args.init_wy_scale,
        init_g_scale=args.init_g_scale,
        teacher_forced=args.teacher_forced,
        lambda_decay=args.lambda_decay,
        omega_scale=args.omega_scale,
        decoder_scale=args.decoder_scale,
        decoder_scales=args.decoder_scales,
        decoder_basis=args.decoder_basis,
        bounding_box=args.bounding_box,
    )

