"""Optuna tuner for decoder scales + gains in teacher-forced runs."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Dict, List

import numpy as np
import optuna


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune decoder scales using Optuna.")
    parser.add_argument("--trials", type=int, default=24, help="Number of Optuna trials.")
    parser.add_argument(
        "--command",
        default="python -m code.experiments.di_no_delay",
        help="Command used to launch experiments.",
    )
    parser.add_argument(
        "--base-args",
        default="--teacher-forced --record-spikes",
        help="Common CLI args (excluding searched hyperparameters and tag).",
    )
    parser.add_argument("--state-dim", type=int, default=2, help="Latent dimension K.")
    parser.add_argument("--target-fire", type=float, default=5.0, help="Target firing rate (Hz).")
    parser.add_argument("--fire-weight", type=float, default=0.2, help="Weight for firing-rate penalty.")
    parser.add_argument("--var-weight", type=float, default=1.0, help="Weight for variance penalty.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("optuna_decoder_scales.json"),
        help="File to store best scales (+ objective).",
    )
    return parser.parse_args()


def run_experiment(command: str, args: str, tag: str) -> Path:
    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", "") or "src"
    cmd = f"{command} {args} --tag {tag}"
    subprocess.run(cmd, shell=True, check=True, env=env, cwd=Path.cwd())
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
        header = f.readline().strip().split(",")
        last = f.readlines()[-1].strip().split(",")
    mapping = dict(zip(header, last))
    return {
        "firing_rate": float(mapping.get("firing_rate", "0")),
        "innovation": float(mapping.get("innovation", "0")),
        "mse": float(mapping.get("mse", "0")),
    }


def objective(args: argparse.Namespace, trial: optuna.Trial) -> float:
    decoder_scale_global = trial.suggest_float("decoder_scale_global", 5.0, 20.0, log=True)
    bounding_box = trial.suggest_float("bounding_box", 30.0, 200.0, log=True)
    innovation_gain = trial.suggest_float("innovation_gain", 10.0, 100.0)
    bias_current = trial.suggest_float("bias_current", 0.05, 0.8)
    threshold_scale = trial.suggest_float("threshold_scale", 0.5, 2.0)
    lambda_decay = trial.suggest_float("lambda_decay", 0.2, 1.0)
    omega_scale = trial.suggest_float("omega_scale", 0.5, 3.0, log=True)

    scales = [
        trial.suggest_float(f"decoder_scale_{i}", 0.2, 5.0, log=True)
        for i in range(args.state_dim)
    ]
    scales_str = ",".join(f"{s:.6g}" for s in scales)

    tag = f"optuna_decoder_{trial.number}"
    run_args = (
        f"{args.base_args} "
        f"--decoder-scale {decoder_scale_global:.6g} "
        f"--decoder-scales {scales_str} "
        f"--bounding-box {bounding_box:.6g} "
        f"--innovation-gain {innovation_gain:.6g} "
        f"--bias-current {bias_current:.6g} "
        f"--threshold-scale {threshold_scale:.6g} "
        f"--lambda-decay {lambda_decay:.6g} "
        f"--omega-scale {omega_scale:.6g}"
    )
    run_dir = run_experiment(args.command, run_args, tag)
    var_ratio = load_var_ratio(run_dir)
    metrics = load_metrics(run_dir)

    # Variance penalty: log-space distance from 1
    var_loss = sum((math.log(max(v, 1e-6)) - math.log(1.0)) ** 2 for v in var_ratio)

    # Firing-rate penalty: only penalize rates above target
    fire_rate = metrics["firing_rate"]
    fire_loss = max(fire_rate - args.target_fire, 0.0) / args.target_fire

    objective_value = args.var_weight * var_loss + args.fire_weight * fire_loss

    trial.set_user_attr("run_dir", str(run_dir))
    trial.set_user_attr("var_ratio", var_ratio.tolist())
    trial.set_user_attr("firing_rate", fire_rate)
    return objective_value


def main() -> None:
    args = parse_args()
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(args, t), n_trials=args.trials)
    best = {
        "decoder_scales": [
            study.best_trial.params[f"decoder_scale_{i}"] for i in range(args.state_dim)
        ],
        "decoder_scale_global": study.best_trial.params["decoder_scale_global"],
        "bounding_box": study.best_trial.params["bounding_box"],
        "innovation_gain": study.best_trial.params["innovation_gain"],
        "bias_current": study.best_trial.params["bias_current"],
        "threshold_scale": study.best_trial.params["threshold_scale"],
        "lambda_decay": study.best_trial.params["lambda_decay"],
        "omega_scale": study.best_trial.params["omega_scale"],
        "objective": study.best_value,
        "run_dir": study.best_trial.user_attrs.get("run_dir"),
        "var_ratio": study.best_trial.user_attrs.get("var_ratio"),
        "firing_rate": study.best_trial.user_attrs.get("firing_rate"),
    }
    args.output.write_text(json.dumps(best, indent=2))
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()

