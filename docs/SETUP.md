# SETUP

## Python
- Python ≥ 3.10
- Dependencies: numpy, scipy, matplotlib, (optional) jax or pytorch for speed

## Install
pip install -e .

## Run a first experiment
python -m code.experiments.di_no_delay --K 2 --Q 1 --N 64 --dt 0.001 --T 10.0 --eta_wy 1e-3 --eta_g 1e-3 --eta_oms 1e-4

> Tip: during development you can skip installation by running  
> `PYTHONPATH=src python -m code.experiments.di_no_delay ...`

## Plots & logs
- CSV logs in `runs/…`
- Matplotlib plots: innovation power, MSE, spike rasters