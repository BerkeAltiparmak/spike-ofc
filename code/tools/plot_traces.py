"""Plot decoded vs true state trajectories for a recorded run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot x_hat vs x for a run.")
    parser.add_argument("run_dir", type=Path, help="Path to run directory (contains state_traces.npz).")
    parser.add_argument("--output", type=str, default="state_traces.png", help="Filename for the plot.")
    return parser.parse_args()


def load_dt(run_dir: Path) -> float:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return 1e-3
    with config_path.open() as f:
        config = json.load(f)
    return float(config.get("dt", 1e-3))


def main() -> None:
    args = parse_args()
    npz_path = args.run_dir / "state_traces.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"{npz_path} not found.")

    data = np.load(npz_path)
    x_hat = data["x_hat"]
    x_true = data["x_true"]
    dt = load_dt(args.run_dir)
    times = np.arange(x_hat.shape[0]) * dt

    fig, axes = plt.subplots(x_hat.shape[1], 1, figsize=(8, 2.5 * x_hat.shape[1]), sharex=True)
    if x_hat.shape[1] == 1:
        axes = [axes]
    for idx, ax in enumerate(axes):
        ax.plot(times, x_true[:, idx], label="x (true)", color="black", linewidth=1.2)
        ax.plot(times, x_hat[:, idx], label="x̂ (decoded)", color="tab:orange", linewidth=1.0)
        ax.set_ylabel(f"Dim {idx}")
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    output_path = args.run_dir / args.output
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()

