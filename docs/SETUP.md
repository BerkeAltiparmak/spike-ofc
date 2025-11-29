# SETUP

## Python
- Python ≥ 3.10
- Dependencies: numpy, scipy, matplotlib, (optional) jax or pytorch for speed

## Install
pip install -e .

## Run a first experiment
python -m code.experiments.di_no_delay --K 2 --Q 1 --N 64 --dt 0.001 --T 10.0 --eta_wy 1e-3 --eta_g 1e-3 --eta_omega_s 1e-4

> Tips:
> - During development you can skip installation by running  
>   `PYTHONPATH=src python -m code.experiments.di_no_delay ...`
> - Artifacts land under `runs/<tag>_<timestamp>/` with `config.json`, `metrics.csv`, `metrics.png`, and `spikes.png`.
> - Use `--no-plots` to skip matplotlib entirely, or `--record-spikes` to keep rasters when plotting is disabled.
> - Spiking excitability is tunable via `--threshold`, `--bias-current`, `--innovation-gain`, and init scales (`--init-wy-scale`, `--init-g-scale`, `--init-v-std`).
> - `state_traces.npz` contains both true and decoded states; `param_stats.json` records matrix norms for fast debugging.
> - Use `--teacher-forced` (with Kalman enabled) to plug in analytic `C D` / `DᵀK_f` and verify the SCN substrate independently of learning.

## Plots & logs
- CSV logs in `runs/…`
- Matplotlib plots: innovation power, MSE, spike rasters