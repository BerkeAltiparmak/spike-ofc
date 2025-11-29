# EVALUATION

## Core metrics
- Innovation energy: `E[t] = || e(t) ||^2`
- State MSE: `|| x̂(t) - x(t) ||^2` (if ground truth available)
- Firing metrics: rate, ISI CV, sparsity (#spikes / N)
- Parameter drift: norms of `W_y`, `G`, `Ω_s`

## Success criteria (τ=0)
- Monotonic downward trend in moving-average innovation energy.
- Reasonable tracking MSE vs analytic Kalman (within factor ~1–2).

## Success criteria (τ>0)
- Stable operation; innovation still decreases after alignment.
- No need for replay/history buffers.

## Robustness tests
- Randomly silence p% of E neurons → performance degrades gracefully and recovers after learning.
- Noise sweeps; hyperparam sweeps.
