# DATA STRUCTURES & APIs

## Shapes
- State: `x ∈ R^K`
- Sensors: `y ∈ R^Q`
- Spikes: `s ∈ {0,1}^N` per dt; traces `r ∈ R^N`
- Matrices: `D[K,N]`, `Ω_f[N,N]`, `Ω_s[N,N]`, `W_y[Q,N]`, `G[N,Q]`

## Modules

### spikeOFC.lti
- `step(x, dt) -> x_next`: A-only (estimation experiments)
- `make_double_integrator(dt, σd, σn) -> (A, C, proc_noise, meas_noise)`

### spikeOFC.scn_core
- `init_decoder(K,N) -> D`
- `fast_matrix(D) -> Ω_f`
- `spike_step(v, r, inputs) -> (s_new, v_new, r_new)`  # LIF + filtered spikes

### spikeOFC.spikeOFC_model
- Holds params: `D, Ω_f, Ω_s, W_y, G, τ`
- Methods:
  - `predict_sensors(r_delay) -> ŷ`
  - `innovation(y, ŷ) -> e`
  - `innovation_current(e) -> Ge`
  - `voltage_rhs(v, r, s, Ge) -> dv`
  - `step(...)` orchestrates one dt

### spikeOFC.learning
- `update_Wy(W_y, e, r_delay, η, α)`
- `update_G(G, Ge, e, η, clip=None)`
- `update_Ωs(Ω_s, Ge, r, η, sym_reg=None)`

### spikeOFC.delay
- `DelayLine(N, τ_steps)`: push/pop interface for r

### spikeOFC.loop
- `simulate(config) -> logs`  # runs the full loop

### spikeOFC.logging
- helpers to compute innovation power, MSE, firing stats
