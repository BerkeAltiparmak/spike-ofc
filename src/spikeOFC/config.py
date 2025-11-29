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
    )

