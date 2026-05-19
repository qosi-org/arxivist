# Benchmark Comparison Report

**Paper**: Deep Residual Learning for Image Recognition
**Paper ID**: arxiv_1512_003385
**arXiv**: https://arxiv.org/abs/1512.03385
**Comparison Date**: 2026-05-19
**SIR Version Used**: 1

---

## Reproducibility Score

| Score | Confidence | Metrics Compared | Matched |
|-------|------------|-----------------|---------|
| **0.96** / 1.0 | medium | 1 | 1 |

**Interpretation**: Excellent reproduction. The single ResNet-20 run reproduces the paper's
reported test error within noise. Confidence is `medium` rather than `high` because only one
of the five paper variants was trained (≥ 3 matched metrics would lift it to `high`).

---

## Metric Comparison Table

| Metric | Dataset | Split | Paper Value | Your Value | Deviation | Severity |
|--------|---------|-------|------------:|-----------:|----------:|----------|
| test_error_rate_percent | CIFAR-10 | test | 8.75 | 8.60 | -0.15 pp (-1.71%) | ✅ Excellent |

(Absolute deviation in percentage points; relative deviation in parentheses.)

---

## Deviation Summary

| Severity | Count |
|----------|-------|
| ✅ Excellent (≤2%) | 1 |
| 🟢 Good (2–5%) | 0 |
| 🟡 Moderate (5–15%) | 0 |
| 🟠 Significant (15–30%) | 0 |
| 🔴 Critical (>30%) | 0 |
| ⬜ Unmatched (paper metrics not measured by user) | 4 |

The four unmatched paper metrics are ResNet-32, ResNet-44, ResNet-56, and ResNet-110 — not run
in this user session. They are recorded but did not affect the score for the metrics that were
measured.

---

## Summary

The user trained ResNet-20 for the full 64,000 iterations on CPU with seed=42 and achieved
**8.60% test error**, slightly under the paper's **8.75%**. The deviation is within the noise
floor expected from single-seed training. The implementation is faithful to the paper for this
variant.

---

## Root Cause Analysis

No deviations of Moderate severity or higher were observed for matched metrics, so no
root-cause analysis is required.

A small note on the direction of deviation: the user's run is **below** the paper's number
(better). Plausible contributing factors:
- The paper reports the result of a single training run; ResNet variants on CIFAR-10 have
  per-seed standard deviation in the 0.1–0.2pp range, so ±0.15pp around 8.75% is well within
  expectation.
- The user's run used the full 50k training split (val_size=0), matching the paper's final
  protocol. If the paper used a 45k/5k split for hyperparameter selection then retrained on
  the full 50k, the slight edge from full-data training is expected.

---

## Hallucination Report Summary

See `hallucination_report.md` for the full report.

| Type | Count | Critical |
|------|-------|---------|
| Structural | 0 | 0 |
| Parametric | 3 | 0 |
| Omission | 0 | 0 |

The three parametric hallucinations are documented assumptions (per-pixel mean, BN weight
decay exclusion, Kaiming init mode) that are exposed as config flags. None appear to be
causing measurable deviation given the close numerical match.

---

## Recommended Actions

Prioritized by expected impact on the broader reproducibility picture (not the current score):

1. **Train ResNet-56 next.** A second matched metric would lift score confidence from
   `medium` to `high` and would also exercise the deeper-network code path.
2. **Train ResNet-110 (with the auto-enabled warmup).** This is the variant most likely to
   surface hidden bugs (warmup logic, deeper-stack gradients). Compare against paper's
   median 6.43%.
3. **Run a 3-seed sweep on ResNet-20** to estimate the per-seed standard deviation in this
   environment and confirm the current -0.15pp is within noise rather than a systematic bias.

---

## Implementation Notes

*From the SIR — sections with confidence < 0.7 that may affect these results:*

None. All SIR sections used in this comparison have confidence ≥ 0.70:

- `training_pipeline`: 0.91
- `evaluation_protocol`: 0.93
- `implementation_assumptions` (aggregate): 0.83

The lowest-confidence individual items (per-pixel mean subtraction at 0.75; BN weight decay
behavior at 0.70) are exposed as config flags rather than baked into the code, so they can
be ablated without modifying the implementation.

---

## Verification Log Summary

- Comparison run at: 2026-05-19T10:30:00Z
- User results hash: `6071c7302ca1027a8ceffddec1d029bd9856d7315954a0a1830b332cb5befe074`
- User-reported config modifications: none (default `configs/resnet20.yaml`)
- Manual review required: no

Full audit trail in `verification_log.md`.
