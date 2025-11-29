# TODO

- [ ] Implement `DelayLine` with exact τ steps + tests *(core class lives in `src/spikeOFC/delay.py`; tests pending)*
- [ ] Implement LIF spiking & thresholds; confirm sparsity with Ω_f *(baseline LIF scaffold exists in `src/spikeOFC/scn_core.py`; need validation)*
- [ ] Implement factorized correction `Gy - G W_y r` *(wired in `src/spikeOFC/spikeOFC_model.py`; needs benchmarking)*
- [ ] Implement learning rules; add Oja/decay options *(rules coded in `src/spikeOFC/learning.py`; stability work TBD)*
- [ ] Double integrator experiment (τ=0) + plots
- [ ] Add τ>0 and verify alignment
- [ ] Compare to analytic Kalman (no learning) baseline
- [ ] Neuron dropout study
- [ ] Unit tests: shapes, invariants, innovation decreases
