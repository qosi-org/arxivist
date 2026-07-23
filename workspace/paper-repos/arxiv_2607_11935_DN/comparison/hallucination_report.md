# Hallucination Report

**Paper ID**: arxiv_2607_11935
**Comparison Date**: 2026-07-18
**Basis**: `sir.json` v1 vs `architecture_plan.json` v1 vs actual code in `src/ews_kalman/`

This report checks the generated implementation against the SIR for three kinds of
deviation: components invented that aren't in the SIR (structural), assumed
hyperparameters that may be wrong (parametric), and SIR-specified components that are
missing or stubbed (omission).

---

## Structural Hallucinations

Components in the generated code that are **not** called for by the SIR.

### 1. Dual Kalman observation mode (`mode="loglog"` / `mode="linear"`)

- **Severity**: Minor
- **Location**: `src/ews_kalman/kalman/tvp_kalman.py`, `TVPKalmanFilter.__init__` and `estimate_beta`
- **Evidence**: The SIR's `architecture.modules` describes a single "TVP-Kalman forward
  filter" using the log-log observation equation `log(q_k) = beta_k * log(T_k) + eps_k`
  (Section 2.2 of the paper). The generated code adds a second, `mode="linear"`
  observation equation (`y_k = beta_k * x_k + eps_k`) not present in the SIR.
- **Why this exists**: Not a hallucination in the usual sense — it was added during
  testing to fix a real bug. Two of the six simulated systems (`beta_step_change`,
  `beta_linear_decay`) are defined in the paper's own Section 2.4 via an explicit
  *linear* relationship, not the log-log elasticity used for the real AIRS data. Running
  those two systems through log-log mode caused the Kalman filter's initial-state
  convergence transient to be misdetected as a false "regime change" (`beta_lead`
  spuriously equal to `tipping_t` itself). This was caught empirically, not assumed.
- **Suggested fix**: None needed — this is a correctness fix, not a defect. Flagged here
  only because it is, technically, a component the SIR didn't originally specify. The
  SIR's `architecture.modules` entry for the Kalman filter should be updated in a future
  SIR revision to describe both modes explicitly, since the paper itself implies both
  are needed once Section 2.4's system definitions are read carefully.

No other structural hallucinations were found. All other classes/functions in
`src/ews_kalman/` trace directly to an SIR `architecture.modules` entry.

---

## Parametric Hallucinations

Hyperparameters marked `# ASSUMED` in `configs/config.yaml` that may be wrong,
cross-referenced against the Critical deviations found in this comparison run.

### 1. Synthetic AIRS regional-data calibration (`AIRSDataLoader.generate_synthetic_region`)

- **Severity**: Significant
- **Location**: `src/ews_kalman/data/airs_loader.py`
- **SIR reference**: `implementation_assumptions[4]` (confidence 0.5) — "exact
  spatial-averaging procedure... not specified in the paper"
- **Evidence this may be wrong**: Directly implicated in 3 of 5 Critical deviations in
  this comparison run: Tropics σ_β (-89.6%), r(β,AR1) (-594%, sign-flipped), and
  r(β,MI) (-112%, sign-flipped). The synthetic generator was calibrated to hit the
  paper's reported *mean* |β| per region (which it does, to within 0.2%), but was never
  calibrated against the paper's reported σ_β or orthogonality-with-AR(1) — those
  properties emerge (or fail to emerge) as an uncontrolled side effect of whatever
  shared noise process generates the synthetic T, q pair.
- **Suggested fix**: Either (a) use real NASA AIRS data (see `data/download.sh`), which
  sidesteps this issue entirely, or (b) if synthetic data must be used, explicitly
  decouple the noise process driving β's variability from the noise process driving
  AR(1)(T), so orthogonality is a property the generator can actually be tested against
  rather than one it was never designed to satisfy.

### 2. Simulation noise_std / forcing-rate parameters (six tipping-point systems)

- **Severity**: Significant
- **Location**: `configs/config.yaml`, `evaluation.simulation` block; consumed by
  `src/ews_kalman/simulation/tipping_systems.py`
- **SIR reference**: `implementation_assumptions[3]` and `ambiguities[1]` (confidence
  0.4-0.45) — "noise magnitudes... not given numerically in the paper"
- **Evidence this may be wrong**: Directly implicated in the Stommel AMOC (-48.7%) and
  critical slowing down (-77.1%) lead-time-margin Critical deviations, and in the
  Beta step change winner flip (paper: AR1 wins; this run: beta wins). All three
  involve exactly the parameters flagged as unspecified before this run occurred.
- **Suggested fix**: Sweep `noise_std` and `forcing_rate` for these three systems (see
  `benchmark_comparison.md` Recommended Actions); document that no combination is
  guaranteed to match, since the paper also doesn't specify the forcing-schedule
  *shape* (linear vs. sigmoid vs. exponential), only its endpoints.

### 3. Kalman process-noise `Q_diag` for simulations (dt=1.0 unitless timestep)

- **Severity**: Minor
- **Location**: `src/ews_kalman/evaluation/simulation_validation.py`,
  `SimulationValidator.__init__` default `Q_diag=(1e-4, 1e-5, 1e-6, 1e-7)`
- **SIR reference**: not directly in the SIR (the paper only gives Q for the real
  1/12-year AIRS application); this is a repo-side extrapolation.
- **Evidence this may be wrong**: Plausibly contributes to the same lead-time-margin
  deviations as hallucination #2 above, but its individual contribution cannot be
  isolated from the noise_std/forcing-rate choices without a controlled sweep.
- **Suggested fix**: Include in the same parameter sweep recommended above; report
  results as a 2D grid (Q_diag scale x noise_std) rather than a single point, since the
  two interact.

---

## Omission Hallucinations

SIR-specified components that are missing or stubbed in the generated code.

**None found.** Every module in the SIR's `architecture.modules` list has a
corresponding, non-stub implementation:

| SIR module | Code location | Status |
|---|---|---|
| TVP-Kalman forward filter | `kalman/tvp_kalman.py::TVPKalmanFilter.filter` | Implemented, tested |
| RTS backward smoother | `kalman/tvp_kalman.py::TVPKalmanFilter.smooth` | Implemented, tested |
| Classical rolling-window EWS | `ews/classical_ews.py::ClassicalEWS` | Implemented, tested (AR1, variance, PE, MI) |
| Lead-lag cross-correlation analysis | `ews/lead_lag.py::LeadLagAnalyzer` | Implemented, tested |
| Simulated tipping-point system generators | `simulation/tipping_systems.py::TippingSystemSimulator` | All 6 systems implemented, tested |

All 46 unit tests pass; none are skipped, xfailed, or assert on stub/placeholder
behavior.

---

## Overall Assessment

The Critical deviations found in this comparison run are **not attributable to missing
or incorrectly-implemented code** (no omission hallucinations, and the one structural
addition was a documented bug fix). They trace cleanly to the two parametric
hallucination categories above — both of which were already flagged as low-confidence
assumptions in the SIR *before* this comparison run was performed. This is the expected
failure mode for a paper whose numeric simulation/data-generation parameters aren't
fully specified in the text: the code is faithful to what's stated, but what's stated
underdetermines several free parameters that turn out to matter for exact reproduction.
