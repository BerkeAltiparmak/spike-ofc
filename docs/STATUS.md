# Spike-OFC Status (2025-11-29)

## Work completed so far
- **Scaffolding & CLI**
  - Implemented `spikeOFC` Python package (`lti`, `scn_core`, `spikeOFC_model`, `learning`, `delay`, `loop`, `config`).
  - Added `code/experiments/di_no_delay.py` as the main entry point with config serialization, logging, plots, and a Kalman baseline overlay.
  - Created `code/tools/plot_traces.py` to visualize decoded vs. true states from saved runs.

- **Simulation loop diagnostics**
  - `loop.simulate` now records spike rasters, decoded state history, true state history, and per-step norms (`‖r‖`, `‖Ge‖`).
  - Each run stores:
    - `metrics.csv`: innovation, MSE, firing rate, `r_norm`, `Ge_norm`, (plus Kalman counterparts).
    - `state_traces.npz`: arrays `x_hat[t]`, `x_true[t]`.
    - `param_stats.json`: Frobenius norms of `W_y`, `G`, `Ω_s`.
    - Plots (`metrics.png`, `spikes.png`).

- **Teacher-forced (analytic) mode**
  - CLI flag `--teacher-forced` inserts analytic weights (`W_y = C D`, `G = DᵀK_f`) and freezes learning rates.
  - Steady-state Kalman gain obtained via an iterative Riccati solver (no SciPy dependency).
  - Provides a ground-truth check for the SCN substrate before enabling learning.

## Current situation / observed issue
- Even in teacher-forced mode, the decoded state `x̂ = D r` carries much less variance than the true state:
  - Baseline run `tf_check2_20251129-030435`: `var(x̂)/var(x) ≈ [3e-3, 2e-3]`, RMSE `[1.7e-2, 6.4e-2]`, average `‖r‖₂ ≈ 5e-3`.
  - Aggressive gain run `tf_gain_sweep_20251129-031156` (`threshold 0.02`, `bias_current 0.5`, `innovation_gain 25`, `lambda_decay 0.3`, `omega_scale 2`): firing rate ≈21 Hz, `var(x̂)/var(x)` improves to `[1.3e-2, 5.3e-2]`, but still <0.1 so the decoded trajectory remains underscaled.
- Diagnostics show innovation power still > Kalman and decoded states lag in amplitude. The SCN substrate needs further rescaling (decoder magnitude, Ω_s gain, leak) before learning can be evaluated.

## Next steps
1. **Continue substrate gain/scale sweeps in teacher-forced mode**
   - Now that `--lambda-decay` and `--omega-scale` exist, experiment with decoder scaling (e.g., multiply `D` by a configurable factor) and innovation gain to push `var(x̂)/var(x)` toward 1.
   - Automate sweeps (Optuna/grid) with objective = teacher-forced RMSE minus Kalman RMSE.
2. **Inspect diagnostics after each sweep**
   - Use `state_traces.png`, variance ratios, `r_norm`, and `Ge_norm` to ensure we neither starve nor saturate the code.
3. **Only after the substrate matches analytic Kalman**
   - Re-enable learning rates and check that innovation/MSE converge toward the reference curves.
4. **Future tasks (once stable)**
   - Introduce τ>0 delay tests, dropout/noise ablations, and automated hyperparameter tuning for the learning rules.

