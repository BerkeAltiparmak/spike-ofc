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
- Even in teacher-forced mode, the decoded state `x̂ = D r` stays nearly zero:
  - Variance ratio `var(x̂)/var(x) ≈ 10⁻³` in `runs/tf_check2_20251129-030435`.
  - RMSE per dimension `[1.7e-2, 6.4e-2]`, far above the analytic Kalman output.
  - Average `‖r‖₂ ≈ 5e-3` and `‖Ge‖₂ ≈ 1e-2`, so the estimator population receives almost no current.
- Because the substrate is under-excited, innovation/MSE curves only “look good” when the latent state has tiny variance; the network fails to track real plant dynamics regardless of learning.
- Root cause: excitability settings (`threshold`, `bias_current`, `innovation_gain`, `Ω_s` scaling, leak λ) are too low, so spikes rarely fire and the state code never leaves the origin. Learning cannot fix this until the substrate reproduces the Kalman behaviour with analytic weights.

## Next steps
1. **Tune substrate gains in teacher-forced mode**
   - Sweep `{threshold, bias_current, innovation_gain}` (Optuna/grid) to maximize decoded-vs-true variance overlap or minimize teacher-forced RMSE.
   - Adjust `Ω_s` scaling and/or leak λ (`lambda_` parameter in `SpikeOFCParams`) to ensure `r` has comparable variance to the true state.
2. **Re-run diagnostics after each sweep**
   - Use `state_traces.png`, variance ratios, and `metrics.csv` to confirm that `x̂(t)` overlays the Kalman trajectory.
3. **Re-enable learning once the substrate matches the analytic baseline**
   - Gradually turn on `η_wy`, `η_g`, `η_Ωs` and check whether innovation/MSE curves converge toward the teacher-forced reference.
4. **Future tasks (after stability)**
   - Introduce measurement delay (τ>0) via `DelayLine`, compare against Kalman with delay.
   - Add Optuna sweeps for learning-rate/bias tuning.
   - Automate plotting of decoded trajectories inside the main experiment to simplify CI checks.

