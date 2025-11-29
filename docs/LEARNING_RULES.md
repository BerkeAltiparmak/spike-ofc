# LEARNING RULES: Local three-factor plasticity

## Objective (innovation energy)
Minimize `L = 0.5 || e ||^2`, with  
`e = y - ŷ`, `ŷ = W_y r(t-τ)`, `x̂ = D r`.

## Updates (all synapse-local)
- Predict sensors (E→P):  
  `ΔW_y = η_y * e r_τ^T` where `r_τ := r(t-τ)`
- Route error back (ε→E):  
  `ΔG = η_g * (G e) e^T`
- Refine internal model (E slow):  
  `ΔΩ_s = η_s * (G e) r^T`

> Each is **pre-trace × postsynaptic dendritic current**, optionally times a scalar gate `m(t)` (we set `m=1` by default).

## Stability notes
- Add light decay/normalization (e.g., Oja) per synapse:
  - `ΔW_y ← ΔW_y - α_y * diag(ŷ) * W_y`
  - `clip_norm` on rows of `G`
- Firing-rate homeostasis via thresholds if needed.
