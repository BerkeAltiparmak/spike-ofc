# Spike-OFC: Factorized Innovation Spike-Coding (Estimation Only)

**Goal:** Build a **spiking**, **fully local-learning** estimator that fuses SCN (spike coding networks) with Bio-OFC (innovation-based local learning) and **handles measurement delay** cleanly.

## 60-second summary
- SCN gives a **sparse spiking substrate**: decode `x̂ = D r`, use `Ω_f=-D^T D` for fast lateral reset, and `Ω_s` for the internal model.
- In SCN, Kalman-like correction is hard-coded as `F_k y + Ω_k r` with `F_k=D^T K_f`, `Ω_k=-D^T K_f C D`.
- Bio-OFC shows you can **learn** observation and Kalman gain **locally** using the **innovation** `e = y - C x̂` (and handle delay by delay-matching).
- **Spike-OFC factorizes** the SCN correction into **two learned maps**:
  - `W_y`: spikes → predicted sensor (`ŷ = W_y r(t-τ)`)
  - `G`: innovation → state current
  - correction = `Gy - G W_y r`, i.e., `F_k = G`, `Ω_k = -G W_y`
- Local updates:
  - `ΔW_y ∝ e r^T`, `ΔG ∝ (G e) e^T`, `ΔΩ_s ∝ (G e) r^T`

See: `THEORY.md`, `ARCHITECTURE.md`, `LEARNING_RULES.md`, `DELAY_IMPLEMENTATION.md`.
