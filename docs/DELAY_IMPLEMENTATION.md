# DELAY IMPLEMENTATION: Principle & Mechanisms

**Principle:** Compare like with like. If sensors have delay `τ`, then at time `t` we observe `y(t) ≈ C x(t-τ)`. Therefore form  
`e(t) = y(t) - W_y r(t-τ)` and inject `G e(t)`.

**Local mechanisms for `r(t-τ)` on E→P path:**
1) **Fixed transmission delay** on synapses (preferred if sim allows).
2) **Trace/FIFO delay line** per synapse; read `r(t-τ)` when computing `ŷ`.
3) **Filter approximation** of a pure delay (all-pass cascade).

**Learning uses aligned signals:**
- `ΔW_y ∝ e(t) r^T(t-τ)`, `ΔG ∝ (G e(t)) e^T(t)`, `ΔΩ_s ∝ (G e(t)) r^T(t)`.
