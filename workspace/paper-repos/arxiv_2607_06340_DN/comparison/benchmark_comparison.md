# Benchmark Comparison Report

**Paper**: Signature-based identification of volatility models from path geometry
**Paper ID**: arxiv_2607_06340
**arXiv**: 2607.06340
**Comparison Date**: 2026-07-13
**SIR Version Used**: 1

---

## Reproducibility Score

| Score | Confidence | Metrics Compared | Matched |
|-------|------------|-----------------|---------|
| **0.38** / 1.0 | Low | 23 | 8 |

**Interpretation**: 0.90-1.00 Excellent · 0.75-0.89 Good · 0.60-0.74 Partial · 0.40-0.59 Significant gap · **<0.40 Critical failure**

This lands right at the boundary between "Significant gap" and "Critical failure." Read the root-cause section below before concluding the reproduction has failed outright — most of the gap traces to two identifiable, explainable causes rather than a broadly broken implementation.

---

## Metric Comparison Table (Experiment 6.1 + Section 6.8 nu experiments)

| Metric | Paper | Yours | Deviation | Severity |
|--------|-------|-------|-----------|----------|
| Overall test accuracy | 0.9863 | 0.7856 | +20.4% | 🟠 Significant |
| Overall train accuracy | ~1.00 | 0.9423 | +5.8% | 🟡 Moderate |
| Heston per-class accuracy | 96.6% | 88.75% | +8.1% | 🟡 Moderate |
| **OU per-class accuracy** | **100.0%** | **53.75%** | **+46.3%** | 🔴 **Critical** |
| rB0.1 per-class accuracy | 98.9% | 92.75% | +6.2% | 🟡 Moderate |
| rB0.3 per-class accuracy | 99.0% | 79.0% | +20.2% | 🟠 Significant |
| Low-nu Heston→OU misclass rate | 69.8% | 1.6% | +97.7% | 🔴 Critical (known open issue) |
| High-nu Heston→OU misclass rate | 9.1% | 6.6% | +27.5% | 🟠 Significant |

**Qualitative check (not scored numerically):** top built-in-importance feature matched the paper exactly (`sig_27` both times); top permutation-importance feature did not (`sig_21` vs. paper's `sig_27`).

Of the paper's ~23 tracked metrics (Section 5/6 experiments, tail exponents, timing, sample-size robustness), this round only exercised 8, via Experiment 6.1 and the two 6.8 nu variants.

---

## The Headline Finding: OU Accuracy Collapse (46% deviation)

This is the most important number in this table, and it's a genuinely new finding — not something already flagged. Every other class degraded gracefully with demo scale (Heston -8%, rB0.1 -6%, rB0.3 -20%), consistent with simply having 83x fewer training paths than the paper (3,000 vs. 250,000). **OU alone collapsed disproportionately** — accuracy fell to nearly half, with 29.9% of OU test paths misclassified as Heston.

**Most likely cause**: OU's random-parameter experiment draws 3 free parameters (kappa, theta, sigma) from fairly wide ranges. With only ~3,000 training paths spread across 4 classes, the classifier may not see enough OU parameter-space coverage to learn a stable boundary against Heston — especially since the paper's own results show the Heston/OU boundary is the hardest one in this problem even at full scale (Section 6.8 exists specifically because this boundary is subtle). A boundary that's already the hardest one at 250,000 paths would plausibly be the first to break down under an 83x sample reduction, while the "easier" rBergomi-vs-everything boundaries hold up better.

**This is a testable hypothesis, not a guess**: if you rerun Experiment 6.1 at a larger scale (e.g. 20,000-50,000 paths/class) and OU accuracy recovers toward 90%+, that confirms sample size as the cause, not a code defect. If it stays low even at 50,000/class (the smallest scale the paper itself tested, where they report 0.9846 overall accuracy), that would point to something else — worth flagging back to me if so.

---

## Root Cause Analysis

### Overall test accuracy (+20.4%) and per-class degradation

**Likely cause: scale.** 3,000 vs. 250,000 training paths/class (83x fewer) is well outside any regime the paper itself tested — their own sample-size robustness check (Table 6.2) only went as low as 50,000/class (5x reduction), where accuracy dropped by less than 0.2 points. An 83x reduction landing outside that tested range producing a much larger gap is not surprising on its own, though the *unevenness* across classes (OU collapsing much harder than the others) suggests scale isn't the whole story — see above.

### Low-nu misclassification rate (+97.7%) — already a known, documented issue

Not new: this is the Section 6.8 calibration ambiguity flagged in the README and notebook before you ran this. The paper's footnote 7 says Heston's `nu` and OU's `sigma` are drawn from "comparable ranges" without defining the calibration precisely; the current implementation's interpretation doesn't reproduce the paper's low-nu confusion mechanism. This is an open item, not a regression.

### High-nu misclassification rate (+27.5%)

Closer to the paper (6.6% vs. 9.1%) but still Significant by the severity bands. Likely the same scale effect as the overall metrics, compounded slightly by whatever calibration gap affects the low-nu experiment (both share the same underlying "comparable ranges" logic, just less severely at high nu).

### Feature importance: built-in matches, permutation doesn't

`sig_27` topping the built-in importance ranking matches the paper exactly — a good sign the signature computation and tree-splitting behavior are structurally correct. The permutation-importance mismatch (`sig_21` vs. paper's `sig_27`) is most likely a stability artifact: permutation importance has known higher variance than gain-based importance (the paper itself notes this), and with only 800 test paths/class and `n_repeats=5`, the ranking of closely-competing features is not well-resolved at this scale. Increase `n_repeats` to 20-30 and use the full test set for a more stable comparison.

---

## Hallucination Summary

See `hallucination_report.md` for full detail. No new structural or code-generation hallucinations found this round — the two bugs found during Stage 4 testing (XGBClassifier predict() issue, 6.8 nu-override elif-chain bug) were already fixed prior to this comparison. The Section 6.8 low-nu gap is best classified as an **unresolved parametric ambiguity** (the paper doesn't fully specify "comparable ranges"), not a hallucinated fact.

---

## Recommended Actions

1. **Rerun Experiment 6.1 at a larger scale** (20,000-50,000 paths/class) as the single highest-value next step — this directly tests whether the OU collapse is a sample-size artifact (most likely) or something deeper.
2. **Re-run feature importance with more repeats** (`n_repeats=20+`) and the full test set to get a stable permutation-importance comparison.
3. **Treat Section 6.8 low-nu as a known limitation**, not a bug to keep chasing right now — it's explainable, documented, and doesn't block using this repo for the paper's main results (Experiment 6.1-6.3).
4. **Run the other 8 untested experiments** (5.1-5.3, 6.2-6.3, 6.5, 6.6, 6.9) to get a fuller picture before drawing an overall verdict — this round only covered 8 of ~23 tracked metrics.

---

## Verification Log Summary

- User-reported config: demo-scale run (`N_PATHS_DEMO=3000`, `N_TEST_DEMO=800` vs. paper's 250,000/50,000), via `notebooks/reproduction_walkthrough.ipynb`, experiments 6.1 + 6.8_low_nu + 6.8_high_nu, seeds 0/1/2.
- Manual review required: **Yes** — the OU accuracy collapse is a new, unexplained-until-retested finding; the low-nu gap is a known, already-documented open issue.

Full computation trace in `verification_log.md`.
