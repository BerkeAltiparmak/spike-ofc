# ARCHITECTURE: Populations & Wiring

## Populations
- **Estimator E (N spiking neurons)**
  - Two dendrites:
    - Prediction dendrite: receives `Ω_s r + Ω_f s`
    - Innovation dendrite: receives `G e`
  - Dynamics: `dot v = -λ v + Ω_s r + Ω_f s + G e`, `dot r = -λ r + s`
  - Readout: `x̂ = D r` (fixed `D`, typically column-normalized)

- **Prediction P (Q units)**
  - Computes `ŷ = W_y r(t-τ)` (delay line or trace on `r`)
  - Sends `ŷ` to Innovation

- **Innovation ε (Q units)**
  - Computes `e = y - ŷ`
  - Broadcasts `e` to E via `G`

## Correction equivalence
- With factorization: `F_k = G`, `Ω_k = -G W_y`
- Voltage correction = `Gy - G W_y r`  
  → equals `D^T K_f y - D^T K_f C D r` at convergence.

## Locality
- E↔E slow synapses: pre `r_j`, post innovation current `(G e)_i`
- ε→E synapses: pre `e_k`, post `(G e)_i`
- E→P synapses: pre `r_j(t-τ)`, post error `e_k`
- **No** use of `C^T`, `D^T` in updates; no messages from other postsyn neurons.
