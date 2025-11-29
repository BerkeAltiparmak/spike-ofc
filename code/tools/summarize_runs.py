"""Summarize metrics for one or more run directories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Spike-OFC run directories.")
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="Run directory/directories (use shell globs if desired).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV file to write the summary table.",
    )
    return parser.parse_args()


def load_metrics(run_dir: Path) -> List[Dict[str, float]]:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"{metrics_path} missing.")
    with metrics_path.open() as f:
        reader = csv.DictReader(f)
        rows = [{k: float(v) for k, v in row.items()} for row in reader]
    return rows


def load_config(run_dir: Path) -> Dict[str, Any]:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return {}
    with cfg_path.open() as f:
        return json.load(f)


def load_var_ratio(run_dir: Path) -> List[float] | None:
    traces_path = run_dir / "state_traces.npz"
    if not traces_path.exists():
        return None
    data = np.load(traces_path)
    x_hat = data["x_hat"]
    x_true = data["x_true"]
    var_ratio = np.var(x_hat, axis=0) / np.var(x_true, axis=0)
    return var_ratio.tolist()


def summarize_run(run_dir: Path) -> Dict[str, Any]:
    metrics = load_metrics(run_dir)
    final = metrics[-1]
    avg_mse = np.mean([row["mse"] for row in metrics])
    avg_innov = np.mean([row["innovation"] for row in metrics])
    avg_r_norm = (
        np.mean([row["r_norm"] for row in metrics]) if "r_norm" in metrics[0] else None
    )
    var_ratio = load_var_ratio(run_dir)
    config = load_config(run_dir)
    return {
        "run": run_dir.name,
        "avg_mse": avg_mse,
        "final_mse": final["mse"],
        "avg_innov": avg_innov,
        "final_innov": final["innovation"],
        "avg_r_norm": avg_r_norm,
        "final_firing_rate": final.get("firing_rate", float("nan")),
        "var_ratio": var_ratio,
        "decoder_scale": config.get("decoder_scale"),
        "innovation_gain": config.get("innovation_gain"),
        "bounding_box": config.get("bounding_box"),
    }


def print_summary(rows: List[Dict[str, Any]]) -> None:
    header = [
        "run",
        "avg_mse",
        "final_mse",
        "avg_innov",
        "final_innov",
        "avg_r_norm",
        "final_firing_rate",
        "var_ratio",
    ]
    print("\t".join(header))
    for row in rows:
        var_str = (
            "[" + ", ".join(f"{v:.3e}" for v in row["var_ratio"]) + "]"
            if row["var_ratio"] is not None
            else "-"
        )
        print(
            "\t".join(
                [
                    row["run"],
                    f"{row['avg_mse']:.3e}",
                    f"{row['final_mse']:.3e}",
                    f"{row['avg_innov']:.3e}",
                    f"{row['final_innov']:.3e}",
                    f"{row['avg_r_norm']:.3e}" if row["avg_r_norm"] is not None else "-",
                    f"{row['final_firing_rate']:.3e}",
                    var_str,
                ]
            )
        )


def write_csv(rows: List[Dict[str, Any]], output: Path) -> None:
    fields = [
        "run",
        "avg_mse",
        "final_mse",
        "avg_innov",
        "final_innov",
        "avg_r_norm",
        "final_firing_rate",
        "var_ratio",
        "decoder_scale",
        "innovation_gain",
        "bounding_box",
    ]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = row.copy()
            flat["var_ratio"] = (
                ";".join(f"{v:.6f}" for v in row["var_ratio"])
                if row["var_ratio"] is not None
                else ""
            )
            writer.writerow(flat)
    print(f"Wrote summary to {output}")


def main() -> None:
    args = parse_args()
    run_dirs = []
    for spec in args.run_dirs:
        if spec.is_dir():
            run_dirs.append(spec)
        else:
            matches = list(spec.parent.glob(spec.name))
            run_dirs.extend([p for p in matches if p.is_dir()])
    if not run_dirs:
        raise SystemExit("No run directories found.")
    rows = [summarize_run(run) for run in sorted(run_dirs)]
    print_summary(rows)
    if args.output:
        write_csv(rows, args.output)


if __name__ == "__main__":
    main()

