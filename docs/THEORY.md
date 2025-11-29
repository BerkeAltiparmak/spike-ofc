# THEORY: SCN + Bio-OFC essentials for spikeOFC

## SCN essentials
- Filtered spikes: `dot r = -λ r + s`, decode `x̂ = D r`.
- Fast recurrence: `Ω_f = -D^T D` (lateral reset/competition) → sparse spikes.
- Slow recurrence: `Ω_s ≈ D^T (A + λ I) D` implements internal linear predictor.
- SCN Kalman embedding: voltage correction has `F_k y + Ω_k r`, with  
  `F_k = D^T K_f`, `Ω_k = -D^T K_f C D` (requires known `C`, `K_f`).

## Bio-OFC essentials
- Innovation: `e = y - C x̂` (with **delay**: `e_t = y_{t} - C x̂_{t-τ}`).
- Local rules (no weight transport):  
  `ΔC ∝ e x̂^T`, `ΔL ∝ (L e) e^T`, `ΔA ∝ (L e) x̂^T`.

## spikeOFC factorization
- Learn `W_y` and `G`, compute `e = y - W_y r(t-τ)`.
- Correction becomes `Gy - G W_y r` → matches SCN at convergence.
- Local rules in SCN coordinates:  
  `ΔW_y ∝ e r^T`, `ΔG ∝ (G e) e^T`, `ΔΩ_s ∝ (G e) r^T`.

**Key principle for delay:** compare **like with like**; only delay the `r → ŷ` path.
