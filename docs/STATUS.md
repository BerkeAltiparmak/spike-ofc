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
  - Added `code/tools/summarize_runs.py` to aggregate per-run summaries (avg/final MSE, innovation, variance ratios).

- **Teacher-forced (analytic) mode**
  - CLI flag `--teacher-forced` inserts analytic weights (`W_y = C D`, `G = DᵀK_f`) and freezes learning rates.
  - Steady-state Kalman gain obtained via an iterative Riccati solver (no SciPy dependency).
  - Provides a ground-truth check for the SCN substrate before enabling learning.

## Current situation / observed issue
- Teacher-forced diagnostics continue to show the decoded state `x̂ = D r` carrying far less variance than the true state (we’re under-coding the latent state):
  - Baseline `tf_check2_20251129-030435`: `var(x̂)/var(x) ≈ [3e-3, 2e-3]`, RMSE `[1.7e-2, 6.4e-2]`, `‖r‖₂ ≈ 5e-3`.
  - High-gain attempt `tf_gain_sweep_20251129-031156`: raising bias, innovation gain, and scaling Ω_s lifted firing rate to ~21 Hz and variance ratios to `[1.3e-2, 5.3e-2]`, but still ≪1.
- Decoder/gain sweep (`tf_sweep_d*_g*` runs):
  - `decoder_scale=10`, `innovation_gain=40` → best balance so far (`var(x̂)/var(x) ≈ [0.61, 0.09]`, avg MSE ≈1.5e-3, `‖r‖₂ ≈1.5e-2`).
  - Increasing gain to 60 boosts the second dimension variance (≈0.34) but raises MSE.
  - `decoder_scale=15` overshoots dim0 (variance ratio ≫1) even at modest gains.
  - `decoder_scale=5` keeps variance <0.5 regardless of gain.
- We are getting closer to the desired variance ratios but still fall short of unity, especially for the second dimension; additional targeted scaling or per-dimension decoder tuning is needed.
- Row-wise scaling experiments (`tf_row_scale*`):
  - Scaling row 2 up by 3–4× while shrinking row 1 (`decoder_scale=10`, `decoder_scales=[0.5, 3]`) brought dim2 variance near 1 (`≈1.08`) but blew up dim1 (`≈6.7`), raising RMSE to ~1.7e-2.
  - Larger row scaling (`[1,4]`) drives dim2 to ≈0.95 but dim1 >11, so manual tweaks aren’t stable.
- Reference code review revealed two critical differences vs. our implementation:
  - SCN spikes have amplitude `1/dt` and only one neuron fires per dt. We now mirror this in `scn_core.spike_step` and adjusted firing-rate logging.
  - The SCN decoder is divided by a `bounding_box_factor` and thresholds are per neuron (`T = diag(DᵀD)/2`). CLI now exposes `--bounding-box` and we compute per-neuron thresholds scaled by `--threshold-scale`.
- Bounding-box experiments:
  - `tf_bbox_20251129-032815` (bounding box 10) still overshot dim0 variance (~6.7).
  - Increasing to bounding box 100 (`tf_bbox100_20251129-032837`) yielded `var(x̂)/var(x) ≈ [0.76, 0.45]`, a significant improvement without manual row tweaks.
  - With per-neuron thresholds active (`tf_thresh_bbox100_20251129-033346`), dim0 lands ~1.0 but dim1 jumps to ~9.0, so we need an automated row-scaling procedure to balance both dims.
- Conclusion: scaling the decoder helps, but we still need additional gain (likely per-dimension scaling of D or a larger innovation gain) to match the Kalman trajectory before we can trust learning runs.

## Next steps
1. **Finish aligning with SCN substrate**
   - Derive thresholds from `T = diag(DᵀD)/2` instead of a global constant so spike resets match the reference implementation.
   - Keep the bounding-box parameter (100 worked best so far) and expose it via the CLI for sweeps.
2. **Automate decoder scaling search (still teacher-forced)**
   - Implement a small optimizer (grid/Optuna) that adjusts per-row scales to drive `var(x̂)/var(x)` toward 1 (e.g., multiplicative updates using the measured ratios).
   - Work around the promising configuration (`decoder_scale=10`, `bounding_box≈100`, `innovation_gain≈40`) and refine per-row scalings to balance both dimensions.
2. **Monitor diagnostics each run**
   - `state_traces.png`, variance ratios, `r_norm`, `Ge_norm`, and firing rates help ensure we’re not saturating or quiescent.
3. **Once teacher-forced matches Kalman**
   - Re-enable learning and verify that innovation/MSE converge toward the reference.
4. **Then proceed to τ>0, robustness tests, and learning hyperopt.**

