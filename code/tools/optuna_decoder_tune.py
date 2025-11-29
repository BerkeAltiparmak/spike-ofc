"""Optuna tuner for decoder row scales in teacher-forced runs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import optuna


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune decoder scales using Optuna.")
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials.")
    parser.add_argument(
        "--command",
        default="python -m code.experiments.di_no_delay",
        help="Command used to launch experiments.",
    )
    parser.add_argument(
        "--base-args",
        default="--teacher-forced --record-spikes --bounding-box 100 --decoder-scale 10 --innovation-gain 40",
        help="Common CLI args (excluding decoder scales and tag).",
    )
    parser.add_argument("--state-dim", type=int, default=2, help="Latent dimension K.")
    parser.add_argument(
        "--scale-range",
        type=float,
        nargs=2,
        default=(0.1, 10.0),
        help="Lower/upper bounds for decoder scales (log-uniform).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("optuna_decoder_scales.json"),
        help="File to store best scales (+ objective).",
    )
    return parser.parse_args()


def run_experiment(command: str, args: str, tag: str) -> Path:
    """Launch di_no_delay with the provided args; return run directory path."""
    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", "") or "src"
    cmd = f"{command} {args} --tag {tag}"
    subprocess.run(cmd, shell=True, check=True, env=env, cwd=Path.cwd())
    # Runs land in runs/<tag>_<timestamp>; find newest match
    runs_dir = Path("runs")
    candidates = sorted(runs_dir.glob(f"{tag}_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise RuntimeError(f"No run directory found for tag {tag}")
    return candidates[0]


def load_var_ratio(run_dir: Path) -> np.ndarray:
    data = np.load(run_dir / "state_traces.npz")
    return np.var(data["x_hat"], axis=0) / np.var(data["x_true"], axis=0)


def load_metrics(run_dir: Path) -> Dict[str, float]:
    with (run_dir / "metrics.csv").open() as f:
        # take final line
        last = None
        for line in f:
            pass
        if last is None:
            f.seek(0)
        else:
            return {}
    return {}


def objective(args: argparse.Namespace, trial: optuna.Trial) -> float:
    scales = [
        trial.suggest_float(f"decoder_scale_{i}", args.scale_range[0], args.scale_range[1], log=True)
        for i in range(args.state_dim)
    ]
    scales_str = ",".join(f"{s:.6g}" for s in scales)
    tag = f"optuna_decoder_{trial.number}"
    run_args = f"{args.base_args} --decoder-scales {scales_str}"
    run_dir = run_experiment(args.command, run_args, tag)
    var_ratio = load_var_ratio(run_dir)
    mse_gap = np.sum((var_ratio - 1.0) ** 2)
    trial.set_user_attr("run_dir", str(run_dir))
    trial.set_user_attr("var_ratio", var_ratio.tolist())
    return mse_gap


def main() -> None:
    args = parse_args()
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(args, t), n_trials=args.trials)
    best = {
        "decoder_scales": [
            study.best_trial.params[f"decoder_scale_{i}"] for i in range(args.state_dim)
        ],
        "objective": study.best_value,
        "run_dir": study.best_trial.user_attrs.get("run_dir"),
        "var_ratio": study.best_trial.user_attrs.get("var_ratio"),
    }
    args.output.write_text(json.dumps(best, indent=2))
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()

