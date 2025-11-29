"""Suggest decoder scale updates based on recorded variance ratios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune decoder row scales via variance ratios.")
    parser.add_argument("run_dir", type=Path, help="Run directory containing config.json and state_traces.npz.")
    parser.add_argument("--target", type=float, default=1.0, help="Desired var(x̂)/var(x) per dimension.")
    parser.add_argument(
        "--mix",
        type=float,
        default=0.5,
        help="Blending factor between old and suggested scale (0=no change, 1=full update).",
    )
    parser.add_argument(
        "--min-scale",
        type=float,
        default=1e-3,
        help="Lower bound for any decoder scale to avoid collapse.",
    )
    parser.add_argument(
        "--max-scale",
        type=float,
        default=1e3,
        help="Upper bound for any decoder scale to avoid explosion.",
    )
    return parser.parse_args()


def load_config(run_dir: Path) -> dict:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"{cfg_path} missing.")
    with cfg_path.open() as f:
        return json.load(f)


def load_var_ratio(run_dir: Path) -> np.ndarray:
    traces_path = run_dir / "state_traces.npz"
    if not traces_path.exists():
        raise FileNotFoundError(f"{traces_path} missing (run with --record-spikes to create it).")
    data = np.load(traces_path)
    x_hat, x_true = data["x_hat"], data["x_true"]
    return np.var(x_hat, axis=0) / np.var(x_true, axis=0)


def parse_scales(config: dict) -> np.ndarray:
    k = config.get("K")
    if k is None:
        raise ValueError("Config missing 'K'.")
    if config.get("decoder_scales"):
        scales = np.fromstring(config["decoder_scales"], sep=",")
        if scales.size != k:
            raise ValueError(f"Expected {k} decoder scales, got {scales.size}")
        return scales
    return np.ones(k)


def suggest_scales(
    old_scales: np.ndarray,
    var_ratio: np.ndarray,
    target: float,
    mix: float,
    min_scale: float,
    max_scale: float,
) -> np.ndarray:
    # Avoid division by zero
    safe_var = np.clip(var_ratio, 1e-6, None)
    desired = old_scales * np.sqrt(target / safe_var)
    return np.clip(old_scales + mix * (desired - old_scales), min_scale, max_scale)


def format_scales(scales: np.ndarray) -> str:
    return ",".join(f"{s:.6g}" for s in scales)


def main() -> None:
    args = parse_args()
    config = load_config(args.run_dir)
    var_ratio = load_var_ratio(args.run_dir)
    old_scales = parse_scales(config)
    new_scales = suggest_scales(
        old_scales,
        var_ratio,
        args.target,
        args.mix,
        args.min_scale,
        args.max_scale,
    )
    print(f"Run: {args.run_dir.name}")
    print(f"Current decoder_scales: {format_scales(old_scales)}")
    print(f"Measured var_ratio: {[f'{v:.3g}' for v in var_ratio]}")
    print(f"Suggested decoder_scales: {format_scales(new_scales)}")
    print("Example CLI flag:")
    print(f"--decoder-scales {format_scales(new_scales)}")


if __name__ == "__main__":
    main()

