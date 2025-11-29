# AGENTS: Cursor Project Guidance for Spike-OFC with Factorized Innovation

This project builds a **local-learning spiking estimator** by combining:
1) **SCN** (spike coding networks) — closed-form spiking substrate for linear dynamics and LQG-style correction.
2) **Bio-OFC** — innovation-driven, **synapse-local learning** (no weight transport), including **measurement delay** handling.

Our estimator **factorizes** the SCN Kalman correction:
- Learn **two maps**: `W_y: spikes→predicted-sensor`, `G: innovation→state`.
- Compute residual `e = y - W_y r(t-τ)`.
- Inject innovation current `G e` into estimator neurons (separate dendrite).
- Recover SCN’s correction at convergence: `F_k = G`, `Ω_k = -G W_y`.

**Documents (under docs/) to read first (context & equations):**
- `README.md` (big picture)
- `THEORY.md` (SCN + Bio-OFC essentials)
- `ARCHITECTURE.md` (populations + wiring)
- `LEARNING_RULES.md` (all local updates, how/why)
- `DELAY_IMPLEMENTATION.md` (principle + local mechanisms)
- `SIM_PLAN.md` (step-by-step dev plan)
- `DATA_STRUCTURES.md` (APIs and tensor shapes)

**Assets available:**
- Papers (PDF): `references/SCN/Closed-form control with spike coding networks.pdf`, `references/Bio-OFC/Neural optimal feedback control with local learning rules.pdf`
- Corresponding codes used in the papers, retrieved from the repositories of the authors: `references/SCN/` and `references/Bio-OFC/`.
- Idea proposal draft: `ideas/paper.tex` (architecture narrative)

---

## Agent Roles

### 1) THEORY_AGENT
**Goal:** Cross-check math, keep implementation aligned with theory.
- Inputs: `THEORY.md`, `ARCHITECTURE.md`, `LEARNING_RULES.md`, both PDFs.
- Tasks:
  - Verify dimensions and identities: `F_k=G`, `Ω_k=-G W_y`.
  - Confirm SCN substrate: `Ω_f=-D^T D`, `Ω_s ≈ D^T(A+λI)D`.
  - Ensure delay use is `r(t-τ)` in `W_y r`, not elsewhere.
  - Sanity-derive the local rules from innovation energy (Section in `LEARNING_RULES.md`).

### 2) SYSTEMS_AGENT
**Goal:** Set up package scaffolding and config plumbing.
- Inputs: `SETUP.md`, `DATA_STRUCTURES.md`.
- Tasks:
  - Create `src/` modules per `DATA_STRUCTURES.md`.
  - Add config loader and CLI args (experiment selection, τ, seeds).
  - Logging hooks (wandb or CSV), plots for innovation power, rates.

### 3) MODEL_AGENT
**Goal:** Implement the model components.
- Inputs: `ARCHITECTURE.md`, `DELAY_IMPLEMENTATION.md`, `LEARNING_RULES.md`.
- Tasks:
  - Implement `Estimator E` (spikes, voltages, two dendrites).
  - Implement `Prediction P` (Wy, delay on r).
  - Implement `Innovation ε` (e=y-ŷ).
  - Wire `Gy - G W_y r` into estimator soma exactly once (no double-count).
  - Implement learning rules with **strict locality**.

### 4) EXPERIMENTS_AGENT
**Goal:** Reproducible experiments.
- Inputs: `SIM_PLAN.md`, `EVAL.md`.
- Tasks:
  - DI (double integrator) with and without delay.
  - Compare Spike-OFC vs analytic Kalman (no learning) as a check.
  - Dropout robustness; noise ablations; hyper sweeps.

### 5) QA_AGENT
**Goal:** Test coverage + invariants.
- Inputs: all docs.
- Tasks:
  - Unit tests for delays, rules, shapes.
  - Numerical gradient spot checks on `W_y` (innovation loss).
  - Assert that learning reduces innovation energy over time.

---

## Golden Invariants (do not violate)
- Only **E→P** path is delayed; estimator recurrence **not** delayed.
- Correction equals `Gy - G W_y r` (factorized), not ad-hoc.
- Local rules use **pre-trace** × **postsyn innovation current**.
- No nonlocal matrices (no `C^T`, no `D^T` in updates).
- Two dendrites in E: prediction (`Ω_s r + Ω_f s`) and innovation (`G e`).

