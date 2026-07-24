# Benchmark Comparison Report

**Paper**: Quantum Kernels and the Cross-Section of Stock Returns: Anatomy of a Vanishing Advantage
**Paper ID**: arxiv_2607_20168
**arXiv**: https://arxiv.org/abs/2607.20168
**Comparison Date**: 2026-07-24
**SIR Version Used**: 1

---

## ⚠️ Read this before the score below

The only results available come from `python run_study.py --study main --debug`, evaluated
with `compare_models.py`. That run used:

- **A synthetic panel** (`data/synthetic.py`): 20-60 fake tickers, randomly-generated
  characteristics with a small artificially-injected signal — not real China A-share data
  (which is proprietary and not released by the paper's author; see `data/README_data.md`).
- **3 windows**, not the paper's 170 (main study). Statistically meaningless sample size.
- **N=32 training subsample**, not the paper's N=1,536. Tiny universe (20-60 tickers vs.
  the paper's top-450 pool).
- A synthetic signal deliberately injected into returns (`0.02*char[0] + 0.01*latent_quality`)
  so the pipeline has *something* learnable to smoke-test against — this makes the
  synthetic data far *more* predictable than real, noisy financial returns.

**The reproducibility score below is therefore not informative about whether the
implementation correctly captures the paper's empirical claims.** All three models score
dramatically *higher* than their real-paper counterparts — expected, since synthetic
returns are far easier to predict than real 20-day-ahead A-share returns, and the small
window/sample counts make every summary statistic unstable. This is a **pipeline
sanity-check**, not a reproduction attempt.

---

## Reproducibility Score

| Score | Confidence | Metrics Compared | Matched |
|-------|------------|-----------------|---------|
| **0.16** / 1.0 | low | 12 | 12 |

---

## Metric Comparison Table

| Metric | Model | Paper Value | Your Value | Deviation | Severity |
|--------|-------|-------------|------------|-----------|----------|
| mean_ic | qkrr-fid | 0.0254 | 0.1271 | +400.4% | 🔴 Critical |
| mean_ic | krr-rbf | 0.0208 | 0.1183 | +468.8% | 🔴 Critical |
| mean_ic | ridge (top-8) | 0.0494 | 0.1816 | +267.6% | 🔴 Critical |
| icir | qkrr-fid | 0.171 | 0.490 | +186.5% | 🔴 Critical |
| icir | krr-rbf | 0.161 | 0.795 | +393.8% | 🔴 Critical |
| icir | ridge (top-8) | 0.247 | 0.958 | +287.9% | 🔴 Critical |
| t_stat | qkrr-fid | 2.23 | 0.85 | −62.0% | 🔴 Critical |
| t_stat | krr-rbf | 2.10 | 1.38 | −34.4% | 🔴 Critical |
| t_stat | ridge (top-8) | 3.21 | 1.66 | −48.3% | 🔴 Critical |
| hit_rate | qkrr-fid | 0.582 | 0.667 | +14.6% | 🟡 Moderate |
| hit_rate | krr-rbf | 0.582 | 0.667 | +14.6% | 🟡 Moderate |
| hit_rate | ridge (top-8) | 0.671 | 0.667 | −0.6% | ✅ Excellent |

---

## Deviation Summary

| Severity | Count |
|----------|-------|
| ✅ Excellent (≤2%) | 1 |
| 🟢 Good (2–5%) | 0 |
| 🟡 Moderate (5–15%) | 2 |
| 🟠 Significant (15–30%) | 0 |
| 🔴 Critical (>30%) | 9 |
| ⬜ Unmatched | 0 |

The hit-rate near-matches (moderate/excellent) are almost certainly coincidental — with
only 3 windows, hit rate can only take values in {0, 1/3, 2/3, 1}, so a collision with the
paper's ~0.58-0.67 range is expected by chance, not evidence of a correctly-calibrated
pipeline on this metric specifically.

---

## Root Cause Analysis

All twelve comparisons point to the same handful of causes; analyzed together rather than
metric-by-metric, since they don't have independent explanations here.

**mean_ic, icir — dramatically higher than paper (Critical, all 3 models):**

1. **Synthetic data has artificially injected, cleaner signal than real returns** (High)
   `data/synthetic.py` injects `0.02*char[0] + 0.01*latent_quality` directly into forward
   returns — by construction easier to learn than the real cross-section's genuinely weak,
   noisy signal-to-noise ratio (which is the paper's whole point: "low signal-to-noise,
   approximately linear").
   Fix: Use real data (see `data/README_data.md`).

2. **Tiny universe (20-60 synthetic tickers vs. real top-450 pool) inflates rank
   correlations** (High) — Spearman IC on a small cross-section is a noisier, higher-variance
   estimator that can spike much higher than its true-population value by chance.
   Fix: Real universe size, or at minimum a much larger synthetic universe for smoke-testing.

**t_stat — dramatically lower than paper (Critical, all 3 models):**

3. **Only 3 windows vs. the paper's 170** (High) — `t_stat = icir * sqrt(n_windows)`; even
   with a comparable ICIR, `sqrt(3)=1.73` vs. `sqrt(170)=13.0` mechanically produces a much
   smaller t-statistic regardless of any real predictive skill.
   Fix: Run the full 170-window walk-forward (`--study main` without `--debug`), which
   requires real data.

**Implementation correctness (not a cause of the above, checked separately):**

4. **No evidence of an implementation bug** — see `hallucination_report.md`. The pipeline
   (bandwidth scaling → quantum circuit → both quantum kernels → classical RBF control →
   closed-form KRR → walk-forward windowing → IC/ICIR/t-stat/hit-rate → paired
   significance → Holm correction) ran successfully end-to-end, including a real bug found
   and fixed during testing (a pandas-version-dependent `groupby().apply()` column-dropping
   issue — see `verification_log.md`), not papered over.

---

## Hallucination Report Summary

See `hallucination_report.md` for the full report.

| Type | Count | Critical |
|------|-------|---------|
| Structural | 0 | 0 |
| Parametric | 5 | 0 |
| Omission | 2 | 0 |

---

## Recommended Actions

1. **Source real data** — this single change would invalidate essentially every deviation
   above; the entire mismatch traces to synthetic-vs-real data, not model/pipeline logic.
2. **Run the full 170-window main study** once real data is wired in — 3 windows cannot
   produce a meaningful t-stat or stable ICIR regardless of data quality.
3. **Resolve the `H_q` gate ambiguity** before trusting any real-data run's quantum-kernel
   numbers specifically — this is the one assumption that could silently produce a
   different (wrong) quantum feature map even with perfect data (see `hallucination_report.md`).
4. **Verify the Nyström extension** (Table 4's 2×2 design) against a hand-computed small
   example if attempting the full-budget comparison — its formula is assumed, not
   paper-stated (SIR confidence 0.55).
5. **Once real results exist, re-run Stage 6** — this report exists to validate the
   reporting pipeline itself and surface known risk factors, not to judge this
   implementation's correctness.

---

## Implementation Notes

*From the SIR — sections with confidence < 0.7:*

- **`mathematical_spec` (confidence 0.72)**: driven down specifically by Eq. (1)'s
  undefined `H_q` term (confidence 0.6 on that entry alone).
- **`training_pipeline` (confidence 0.55)**: closed-form KRR itself is well-specified, but
  MLP/NN3 optimizer/schedule and the exact stratified-subsampling allocation scheme are not
  stated in the paper.
- **`implementation_assumptions` (confidence 0.55, aggregate)**: the `H_q` gate, projected-
  kernel γ tuning, Nyström formula, and subsampling scheme are all individually flagged in
  `sir.json → implementation_assumptions[]` / `→ ambiguities[]`.

---

## Verification Log Summary

- Comparison run at: 2026-07-24T05:31:52Z
- User results directory: `/tmp/qkernel_out/` (3 CSVs: `qkrr-fid_ic_series.csv`,
  `krr-rbf_ic_series.csv`, `ridge_ic_series.csv`, plus `comparison.csv`/`comparison_pairwise.csv`)
- Manual review required: **yes** — synthetic smoke-test results, not a real reproduction
  attempt; do not treat the 0.16 score as evidence of implementation quality either way.

Full audit trail in `verification_log.md`.
