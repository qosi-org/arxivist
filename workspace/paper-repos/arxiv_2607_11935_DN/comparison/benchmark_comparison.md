# Benchmark Comparison Report

**Paper**: Eigenvector rotation precedes eigenvalue-based early-warning signals: a TVP-Kalman approach to detecting critical transitions
**Paper ID**: arxiv_2607_11935
**arXiv**: https://arxiv.org/abs/2607.11935
**Comparison Date**: 2026-07-18
**SIR Version Used**: 1

---

## Reproducibility Score

| Score | Confidence | Metrics Compared | Matched |
|-------|------------|-----------------|---------|
| **0.01** / 1.0 | Medium | 15 | 6 |

**Interpretation**:
- 0.90–1.00: Excellent reproduction
- 0.75–0.89: Good reproduction with minor deviations
- 0.60–0.74: Partial reproduction — review moderate deviations
- 0.40–0.59: Significant reproduction gap — likely implementation issues
- **< 0.40: Critical failure — fundamental mismatch** ← this run falls here

This score reflects a **fundamental mismatch on 5 of 6 matched metrics**, not a code
bug. One metric (Tropics |β|) is genuinely excellent. See Summary below.

---

## Metric Comparison Table

| Metric | Dataset | Split | Paper Value | Your Value | Deviation | Severity |
|--------|---------|-------|-------------|------------|-----------|----------|
| \|β\| (mean absolute elasticity) | NASA AIRS Tropics | N/A | 0.492 | 0.4910 | **-0.2%** | ✅ Excellent |
| σ_β | NASA AIRS Tropics | N/A | 0.035 | 0.0036 | **-89.6%** | 🔴 Critical |
| r(β, AR(1)_T) | NASA AIRS Tropics | N/A | +0.08 (n.s., p=0.22) | -0.395 (p=3.4e-11) | **-594.1%** | 🔴 Critical |
| r(β, MI) | NASA AIRS Tropics | N/A | -0.33 (p<0.001) | +0.041 (n.s., p=0.52) | **-112.5%** | 🔴 Critical |
| β lead over AR(1), Stommel AMOC | Simulated Stommel AMOC | N/A | +39 timesteps | +20 timesteps | **-48.7%** | 🔴 Critical |
| β lead over AR(1), critical slowing down | Simulated CSD | N/A | +153 timesteps | +35 timesteps | **-77.1%** | 🔴 Critical |
| \|β\|, σ_β, r(β,AR1), r(β,MI) — Arctic | NASA AIRS Arctic | N/A | 0.106, 0.139, +0.05, +0.03 | *not reported by user* | — | ⬜ Unmatched |
| \|β\|, σ_β, r(β,AR1), r(β,MI) — Monsoon | NASA AIRS Indian Monsoon | N/A | 0.477, 0.062, +0.03, +0.29 | *not reported by user* | — | ⬜ Unmatched |
| β lead-lag range (observational, months) | NASA AIRS (all regions) | N/A | ~19 (range 14-24) | *not reported by user* | — | ⬜ Unmatched |
| Fold bifurcation winner | Simulated | N/A | AR1 (β never detects) | AR1 | qualitative match | ✅ (not scored quantitatively — no paper timestep given) |
| Logistic map | Simulated | N/A | tie (100 vs 100) | tie (133 vs 133) | qualitative match | ✅ (not scored quantitatively — no paper timestep given) |
| Beta step change winner | Simulated | N/A | AR1 (165 vs 170, narrow) | **beta** (197 vs 153) | **winner flipped** | 🔴 Critical (qualitative) |
| Beta linear decay | Simulated | N/A | beta only (AR1 never crosses) | beta wins (AR1 also crosses at 203) | partial qualitative match | 🟠 Significant (qualitative) |

---

## Deviation Summary

| Severity | Count |
|----------|-------|
| ✅ Excellent (≤2%) | 1 |
| 🟢 Good (2–5%) | 0 |
| 🟡 Moderate (5–15%) | 0 |
| 🟠 Significant (15–30%) | 0 |
| 🔴 Critical (>30%) | 5 |
| ⬜ Unmatched | 9 |

---

## Summary

One number matches the paper almost exactly: **Tropics |β| = 0.491 vs paper's 0.492**
(-0.2% deviation). This is not luck — it reflects the Clausius-Clapeyron physics
(β → L/RT ≈ 0.5) correctly emerging from the TVP-Kalman filter on synthetic data
built to respect that scaling relationship.

Everything else matched is **critically off**, and for a specific, identifiable reason:
**the paper's central empirical claim — that β is *orthogonal* to AR(1) (r≈0, p>0.05) —
does not hold in this reproduction.** We get r=-0.395 with p=3.4×10⁻¹¹, a strong and
highly significant correlation. The MI correlation sign is also flipped (paper: -0.33;
ours: +0.04). This is a genuine reproducibility failure on the paper's core hypothesis,
not a rounding difference — see Root Cause Analysis below for why.

The six-simulation lead-time margins are directionally right for 4 of 6 systems (fold
bifurcation, logistic map tie, Stommel AMOC, critical slowing down all match the
paper's *winner*), but the "beta step change" system's winner is flipped (paper: AR1
narrowly wins; ours: β wins clearly), and all matched lead-time *magnitudes* are
50-80% smaller than the paper's — expected, since exact simulation noise/forcing
parameters are not given in the paper (flagged at SIR confidence 0.4-0.45 before this
run even happened).

---

## Root Cause Analysis

### σ_β (Tropics) — -89.6% deviation

**Likely causes** (ordered by probability):

1. **Synthetic data generator produces insufficient natural variability** (High)
   The synthetic-fallback Tropics series is built to hold β close to 0.5 *tightly* by
   construction (to match the reported mean), which mechanically suppresses its
   variance far below what real, noisy atmospheric T-q coupling would show.
   Fix: `AIRSDataLoader.generate_synthetic_region()` needs a noise-injection term on
   top of the target mean β, calibrated to also hit the target σ_β, not just the mean.

2. **Kalman process noise Q may be too small for this series** (Medium)
   Q=diag(1e-6, 1e-7, 1e-8, 1e-9) (paper-stated exactly) constrains how fast β can
   move step-to-step; combined with cause 1, an already-smooth input series gets
   smoothed further.
   Fix: this is a paper-given hyperparameter, so it should not be changed — the fix
   belongs in the synthetic data generator, not the filter.

### r(β, AR(1)_T) and r(β, MI) (Tropics) — -594% and -113% deviation, sign flips

**Likely causes** (ordered by probability):

1. **Synthetic generator entangles β and AR(1)/MI by construction** (High)
   Since both β and the classical EWS are derived from the *same* synthetically
   generated T, q pair, any shared underlying noise process used to build that pair
   can inject spurious correlation between β and AR(1)/MI that would not exist between
   independently-observed real climate variables. The paper's orthogonality result is
   an empirical property of *real* atmospheric physics — a synthetic toy generator has
   no obligation to reproduce it unless explicitly designed to.
   Fix: this is the single highest-priority fix — see Recommended Actions.

2. **This is not a code bug — it's a data-source mismatch** (High)
   No implementation change to `RegionSummaryComputer` or `LeadLagAnalyzer` will fix
   this; both computed the correlation correctly (see `test_evaluation.py`, verified
   passing). The problem is upstream, in what data was fed in.
   Fix: use real NASA AIRS data (see `data/download.sh`) — this is the only reliable
   fix, since orthogonality is an emergent property of the real dataset the paper
   analyzed, not something a synthetic generator can be made to guarantee in general.

### β lead over AR(1), Stommel AMOC and critical slowing down — -48.7% and -77.1% deviation

**Likely causes** (ordered by probability):

1. **Simulation noise_std / forcing-rate parameters don't match the paper's (unknown) values** (High)
   Already flagged in the SIR (confidence 0.4-0.45) and `architecture_plan.json`
   risk_assessment before this run: the paper gives no numeric noise magnitudes or
   forcing schedules for any of the six systems.
   Fix: sweep `configs/config.yaml`'s `evaluation.simulation.stommel_amoc` and
   `critical_slowing_down` blocks (noise_std, forcing_rate, lambda_start) to see if a
   different illustrative choice pushes the margin closer to 39 / 153 timesteps —
   but note there is no guarantee an exact match exists, since the *shape* of the
   forcing trajectory (linear vs. sigmoid vs. exponential ramp) is also unspecified.

2. **Kalman Q_diag for simulations (1e-4, 1e-5, 1e-6, 1e-7) is a repo default, not paper-stated** (Medium)
   The paper only gives Q for the real 1/12-year AIRS application; the simulation Q was
   chosen by this repo to be proportionally larger for the dt=1.0 unitless timestep, but
   is otherwise a free choice.
   Fix: expose this as a config parameter and sweep it alongside the noise/forcing
   parameters above.

### Beta step change — winner flipped (paper: AR1; ours: beta)

**Likely causes** (ordered by probability):

1. **Same simulation-parameter ambiguity as above** (High) — this system uses `mode="linear"`
   Kalman filtering (a repo-side fix made during testing, see `architecture_plan.json`
   risk_assessment), and its noise_std is again a free illustrative choice.
   Fix: same sweep as above, applied to `configs/config.yaml`'s `beta_step_change` block.

2. **The paper's own reported result here is already "close" (165 vs 170, narrow AR1 win)** (Medium)
   Small parameter changes could plausibly flip this system's winner in *either*
   direction even with correct methodology — the paper's own numbers show this system
   sits right at the boundary. A flipped winner on a near-tie is less alarming than a
   flipped winner on Stommel AMOC or CSD would be (where the paper reports a clear
   margin).

---

## Hallucination Report Summary

See `hallucination_report.md` for the full report.

| Type | Count | Critical |
|------|-------|---------|
| Structural | 1 | 0 |
| Parametric | 3 | 0 |
| Omission | 0 | 0 |

---

## Recommended Actions

Prioritized by expected impact on reproducibility score:

1. **Fix the synthetic-fallback data generator to decouple β from AR(1)/MI**, or —
   preferably — **switch to real NASA AIRS data** (`data/download.sh`). This single
   change would address 3 of 5 Critical deviations (σ_β, r(β,AR1), r(β,MI)) and is the
   only way to actually test the paper's central orthogonality claim, which cannot be
   validated on synthetic data built from a shared noise process.
2. **Sweep simulation noise_std / forcing-rate parameters** in `configs/config.yaml`'s
   `evaluation.simulation` block for Stommel AMOC, critical slowing down, and beta
   step change, to see whether lead-time margins and the beta-step-change winner move
   closer to the paper's reported values. Treat this as exploratory, not a guaranteed
   fix — the paper doesn't give enough information to know if an exact match exists.
3. **Re-run with N ≥ 3 independent seeds per simulated system** and report a
   median/IQR rather than a single seed=0 run, to distinguish genuine parameter-driven
   deviation from single-run noise sensitivity (the current run used only seed=0 for
   both the reproduce and explore notebooks, which is why they agree with each other
   but that agreement doesn't by itself validate against the paper).

---

## Implementation Notes

*From the SIR — sections with confidence < 0.7 that may affect these results:*

- **`architecture` (0.7)**: F transition matrix named but not numerically specified by
  the paper (ambiguity #1); simulation generators (confidence 0.45) have unspecified
  noise/forcing parameters (ambiguity #2) — directly implicated in the Critical
  deviations on Stommel AMOC, CSD, and beta step change above.
- **`implementation_assumptions` (0.5)**: permutation-entropy/MI estimator choice, and
  the synthetic AIRS spatial-averaging assumption — directly implicated in the
  Tropics σ_β / r(β,AR1) / r(β,MI) Critical deviations above.

---

## Verification Log Summary

- Comparison run at: 2026-07-18T15:11:21Z
- User results hash: `efa7f34215883e703c37475a830b0d4a61f985ca1206aacf276ba87754fd0935`
- User-reported config modifications: none reported (default `configs/config.yaml`, seed=0, synthetic-fallback data)
- Manual review required: **yes** — orthogonality claim (the paper's central result) does not hold in this run; root cause is very likely the synthetic data source, not the implementation, but this should be confirmed with real AIRS data before drawing any conclusion about the paper's validity.

Full audit trail in `verification_log.md`.
