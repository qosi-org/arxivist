# Benchmark Comparison Report

**Paper**: Approximating SPR Distance Between Phylogenetic Trees with Graph Neural Networks
**Paper ID**: arxiv_2607_18311
**arXiv**: https://arxiv.org/abs/2607.18311
**Comparison Date**: 2026-07-22
**SIR Version Used**: 1

---

## ⚠️ Read this before the score below

The only results available for this comparison come from `python train.py --config
configs/config.yaml --debug --data-dir data/toy`, evaluated with `evaluate.py`. That run
used:
- **`data/toy/master_pairs.csv`**: 4 synthetic tree pairs with **randomly-generated SPR
  labels** — not the paper's real 388-pair Zenodo dataset, and not real
  `phangorn::SPR.dist` labels.
- **`--debug` overrides**: `max_epochs=2`, `batch_size=4` (vs. the paper's early-stopping
  regime over the full 272-pair training split).
- **A test split of 1 pair**, so R² is mathematically undefined (needs ≥2 unique target
  values) and MAE/RMSE are single-sample point estimates, not population statistics.

**The reproducibility score below is therefore not a judgment on whether the
implementation is correct.** It reflects "wrong dataset + 2 epochs," which is expected to
score badly regardless of implementation quality. No genuine reproduction attempt against
the paper's actual dataset has been run yet. Treat this report as a **template exercise
and pipeline sanity-check**, not a verdict — re-run Stage 6 once real training on the
downloaded Zenodo dataset is complete.

---

## Reproducibility Score

| Score | Confidence | Metrics Compared | Matched |
|-------|------------|-----------------|---------|
| **0.0** / 1.0 | low | 4 | 2 |

**Interpretation** (per the standard rubric — **does not apply meaningfully here**, see caveat above):
- 0.90–1.00: Excellent reproduction
- 0.75–0.89: Good reproduction with minor deviations
- 0.60–0.74: Partial reproduction — review moderate deviations
- 0.40–0.59: Significant reproduction gap — likely implementation issues
- < 0.40: Critical failure — fundamental mismatch

---

## Metric Comparison Table

| Metric | Dataset | Split | Paper Value | Your Value | Deviation | Severity |
|--------|---------|-------|-------------|------------|-----------|----------|
| MAE | in-distribution | test | 127.13 | 1.00 | −99.21% | 🔴 Critical |
| RMSE | in-distribution | test | 202.24 | 1.00 | −99.51% | 🔴 Critical |
| R² | in-distribution | test | 0.873 | undefined (null) | n/a | ⬜ Unmatched |
| MAPE | in-distribution | test | *(paper does not report in-distribution MAPE — only stratified-CV MAPE = 2.16%)* | 100.00 | n/a | ⬜ Unmatched |

---

## Deviation Summary

| Severity | Count |
|----------|-------|
| ✅ Excellent (≤2%) | 0 |
| 🟢 Good (2–5%) | 0 |
| 🟡 Moderate (5–15%) | 0 |
| 🟠 Significant (15–30%) | 0 |
| 🔴 Critical (>30%) | 2 |
| ⬜ Unmatched | 2 |

---

## Root Cause Analysis

### MAE on in-distribution — −99.21% deviation
### RMSE on in-distribution — −99.51% deviation

Both metrics fail for the same underlying reasons; analyzed together.

**Likely causes** (ordered by probability):

1. **Wrong dataset entirely — synthetic random-label toy fixture, not the paper's real data** (High)
   Fix: Run `python data/download.py` to fetch the real Zenodo dataset (DOI
   10.5281/zenodo.20476872), then `python train.py --config configs/config.yaml` (no
   `--debug`, no `--data-dir data/toy`).

2. **Severely truncated training — 2 epochs vs. an early-stopping regime with LR-plateau
   patience of 10 and stopping patience of 25** (High)
   Fix: Remove `--debug`; let training run to convergence under the real early-stopping
   config.

3. **Test-set size of 1 pair — no statistical meaning at this scale** (High)
   Fix: Use the real 388-pair dataset's 15% test split (~58 pairs) as the paper does
   (Sec 4.3, seed 42).

4. **GIN layer-count ambiguity (SIR confidence 0.6): 2 layers used vs. Figure 5's apparent
   3** (Medium — cannot be assessed from this run; would only become visible with real
   training)
   Fix: Once real training is running, try `model.num_gin_layers: 3` in a second run and
   compare R² against Table 2's 0.873 to disambiguate empirically.

5. **Implementation bug** (Low)
   No evidence of one — see `hallucination_report.md`. All architecture modules trace
   cleanly to the SIR, and the pipeline (parsing → features → GIN → pooling → MLP head →
   loss → optimizer step) has been exercised end-to-end without errors, including a real
   bug (missing `--data-dir` flag in `evaluate.py`) that was found and fixed during
   testing rather than papered over.

### R² — Unmatched

**Likely causes:**
1. **Single-pair test split makes R² mathematically undefined** (High) — `sklearn.r2_score`
   requires ≥2 unique target values; `RegressionMetrics.compute()` correctly returns `NaN`
   rather than a misleading number.
   Fix: Same as above — needs the real dataset's full test split.

### MAPE — Unmatched

**Likely cause:**
1. **The paper does not report an in-distribution single-split MAPE at all** (High) — it
   only reports MAPE for the aggregated stratified cross-validation regime (2.16% ± 0.37%),
   which this repo's config also supports (`evaluation.cross_validation`) but which
   requires multiple training runs across folds, not yet performed.
   Fix: No implementation fix needed; this is a dataset-granularity mismatch, not an error.

---

## Hallucination Report Summary

See `hallucination_report.md` for the full report.

| Type | Count | Critical |
|------|-------|---------|
| Structural | 0 | 0 |
| Parametric | 6 | 0 |
| Omission | 1 | 0 |

---

## Recommended Actions

Prioritized by expected impact on a *real* reproducibility score:

1. **Download and use the real dataset** (`python data/download.py`) — this single change
   would invalidate essentially all of the "Critical" deviations above, since they stem
   from comparing against synthetic noise, not a training failure.
2. **Run full training without `--debug`** — let early stopping (patience 25) and
   LR-plateau (patience 10) govern the run, per Sec 4.3.
3. **Once real results exist, re-run Stage 6** and specifically watch the R² metric —
   it's the paper's primary reported statistic (0.873 in-distribution) and the most
   sensitive to the `num_gin_layers` ambiguity.
4. **If real R² lands meaningfully below ~0.85**, try `num_gin_layers: 3` as the single
   highest-leverage architecture change per the SIR's flagged ambiguity.
5. **Tune `huber_delta`** if the loss looks unstable during real training — SPR distances
   range roughly 0–2000, and the assumed default of 1.0 may be too small for that scale.

---

## Implementation Notes

*From the SIR — sections with confidence < 0.7 that may affect results once real training runs:*

- **`training_pipeline` (confidence 0.55)**: batch size, max epochs, and Adam β1/β2 are
  not stated in the paper at all; current defaults (32, 200, 0.9/0.999) are literature
  conventions, not paper-derived.
- **`implementation_assumptions` (confidence 0.6, aggregate)**: GIN layer count (0.6),
  batch size (0.3), max epochs (0.4), Huber delta (0.5), and taxonomic-id semantics (0.55)
  are all flagged individually in `sir.json → implementation_assumptions[]`.
- **`architecture` (confidence 0.78)**: driven down specifically by the 2-vs-3 GIN layer
  ambiguity (Sec 4.2 text vs. Fig. 5 diagram).

---

## Verification Log Summary

- Comparison run at: 2026-07-22T18:34:20Z
- User results hash: `d7ea9ccd78dea8fe0f6d6ebcd7e0693f3c1d1691bd294b5e866a78ce8b5fbb72`
- User-reported config modifications: `--debug` (max_epochs 200→2, batch_size 32→4),
  `--data-dir data/toy` (synthetic 4-pair fixture with random labels, not the real dataset)
- Manual review required: **yes** — results are from a synthetic smoke test, not a real
  training run; do not treat the 0.0 score as evidence of an implementation problem.

Full audit trail in `verification_log.md`.
