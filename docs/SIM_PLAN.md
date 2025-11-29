# SIMULATION PLAN

## Phase 0: Skeleton
- Build `src/` modules per `DATA_STRUCTURES.md`.
- Implement LTI plants (double integrator) and a vanilla continuous-time Kalman for checks.

## Phase 1: No-delay sanity (τ=0)
- Initialize: random small `W_y`, `G`, `Ω_s`; set `Ω_f=-D^T D`.
- Run loop (Euler or RK):
  1) Plant step: `x ← x + dt*(A x) + noise_d`
  2) Sensors: `y = C x + noise_n`
  3) E dynamics: spike update; update `r`, `v`
  4) P: `ŷ = W_y r`
  5) ε: `e = y - ŷ`
  6) Correction: add `G e` to E
  7) Learning: apply `ΔW_y, ΔG, ΔΩ_s`
- Metrics: innovation power `||e||^2`, MSE `||x̂-x||^2`, firing rate, spike sparsity.

## Phase 2: Delay (τ>0)
- Add delay line on E→P path for `r(t-τ)`.
- Verify same convergence trend; innovation power decreases.

## Phase 3: Comparisons
- Compare to analytic Kalman (no learning) under same noise.
- Sweep N (overcompleteness), λ, learning rates.

## Phase 4: Robustness
- Neuron dropout in E; see re-adaptation.
- Sensor noise increase; process noise change.

## Phase 5: Ablations
- Freeze `G`, learn `W_y` only, and vice versa.
- Freeze `Ω_s`; then allow learning; compare.

**Deliverables:** plots (innovation power vs time), MSE vs time, raster plots, hist of spikes, parameter norms.
