# Benchmark Comparison Report

**Paper**: LATTICE: Graph Self-Supervised Learning for Multimodal Spatial Omics Integration
**Paper ID**: arxiv_2607_14410
**arXiv**: https://arxiv.org/abs/2607.14410
**Comparison Date**: 2026-07-22
**SIR Version Used**: 1

---

## Reproducibility Score

| Score | Confidence | Metrics Compared | Matched |
|-------|------------|-------------------|---------|
| **0.070** / 1.0 | low | 5 | 4 |

**Interpretation**: < 0.40 = Critical failure — fundamental mismatch.

**Read this before the number alarms you**: this is not primarily a verdict on
the *implementation*. It's the expected, honest result of comparing a run on
**synthetic placeholder data** (4 samples × 800 spots, 25 epochs) against the
paper's numbers from its **real, private 11-sample × 4,992-spot melanoma
cohort** trained for up to 100 epochs. See "Root Cause Analysis" below — data
mismatch dominates every deviation here, not code bugs.

---

## Metric Comparison Table

| Metric | Dataset | Paper Value | Your Value | Deviation | Severity |
|--------|---------|-------------|------------|-----------|----------|
| ARI | melanoma (paper) / synthetic (repro), M5 | 0.329 | 0.052 | -84.2% | 🔴 Critical |
| NMI | melanoma (paper) / synthetic (repro), M5 | 0.450 | 0.078 | -82.7% | 🔴 Critical |
| Spatial contiguity | melanoma (paper) / synthetic (repro), M5 | 0.850 | 0.533 | -37.3% | 🔴 Critical |
| Silhouette | melanoma (paper) / synthetic (repro), M5 | 0.417 | 0.276 | -33.8% | 🔴 Critical |
| MUS | melanoma (paper) / synthetic (repro), M5 | 0.803 | 0.388 | — | ⬜ Unmatched* |

\* MUS is normalized jointly across all rows in an evaluation pool (Eq. 11).
The paper's pool is 5 baselines + 5 LATTICE ladder levels on the *real*
cohort; this repro's pool is only its own 5 synthetic ladder levels. The two
0.0–1.0 scales are not directly comparable, so this is marked unmatched
rather than given a fabricated deviation number.

---

## Deviation Summary

| Severity | Count |
|----------|-------|
| ✅ Excellent (≤2%) | 0 |
| 🟢 Good (2–5%) | 0 |
| 🟡 Moderate (5–15%) | 0 |
| 🟠 Significant (15–30%) | 0 |
| 🔴 Critical (>30%) | 4 |
| ⬜ Unmatched | 1 |

---

## Root Cause Analysis

### ARI — -84.2% deviation, NMI — -82.7% deviation

**Likely causes** (ordered by probability):

1. **Data mismatch: synthetic vs. real cohort** (High)
   The synthetic data generator produces spot signal from a simple Gaussian
   mixture over synthetic "domains" — it has none of the real biological
   structure (cell-type gradients, true regulatory programs) that the
   paper's Space-Ranger-derived reference clusters are built from. ARI/NMI
   specifically measure agreement with an RNA-derived reference, and there
   is no reason a synthetic reference clustering should behave the same way.
   *Fix*: Only a real (or realistically simulated, e.g. scRNA-simulator-based)
   multimodal spatial dataset can close this gap — see `data/README_data.md`.
2. **Undertrained relative to the paper** (High)
   25 epochs vs. up to 100 with early-stopping patience 20 (paper's cohort
   averaged 45.2 epochs to converge, Appendix G.2); 4 samples vs. 11.
   *Fix*: Re-run with the full `configs/config.yaml` (11 samples, 4,992
   spots/sample, 100 epochs) once you have real or better-simulated data.
3. **Ambiguous loss/alignment choices** (Medium)
   The Huber-vs-MSE reconstruction loss and single-pair alignment choice
   (SIR ambiguities[0], [1]) could shift these numbers by some margin, but
   are very unlikely to explain an 80%+ gap on their own.
   *Fix*: Sweep `training.recon_loss` and `model.projection_head.aligned_modality_pair` once data mismatch is addressed.

### Spatial contiguity — -37.3% deviation, Silhouette — -33.8% deviation

**Likely causes** (ordered by probability):

1. **Small, low-density synthetic graph** (High)
   800 spots on a small synthetic grid produces a sparser, less spatially
   coherent structure than the real 4,992-spot Visium arrays, directly
   suppressing spatial-contiguity and silhouette scores regardless of model
   correctness.
2. **Gaussian-kernel edge weight left off** (Medium)
   `data.edge_weight_mode: uniform` (SIR ambiguities[2], confidence 0.5) may
   under-weight true spatial proximity relative to whatever the paper
   actually used.
   *Fix*: Try `edge_weight_mode: gaussian` and compare.
3. **Undertraining** (Medium) — same as above.

---

## Hallucination Report Summary

See `hallucination_report.md` for the full report.

| Type | Count | Critical |
|------|-------|---------|
| Structural | 1 | 0 |
| Parametric | 4 | 0 |
| Omission | 1 | 0 |

No Critical hallucinations were found — all flagged items were already
documented as assumptions in the SIR/architecture plan with confidence
scores, not undisclosed deviations.

---

## Recommended Actions

Prioritized by expected impact on reproducibility score:

1. **Get real (or realistically simulated) multimodal spatial data.** This
   single factor almost certainly accounts for the bulk of the ARI/NMI gap;
   no hyperparameter change will close an 80%+ deviation caused by
   comparing against a fundamentally different data distribution.
2. **Run the full-scale config** (11 samples, 4,992 spots/sample, up to 100
   epochs with early stopping) rather than the scaled-down demo config used
   here, once data is addressed.
3. **Resolve the Eq.6-vs-Appendix-H reconstruction-loss ambiguity** by
   trying both `huber` and `mse` and comparing — low effort, plausible
   secondary lever.
4. **Try Gaussian edge weighting** (`data.edge_weight_mode: gaussian`) to
   see if it meaningfully changes spatial contiguity / silhouette.

---

## Implementation Notes

*From the SIR — sections with confidence < 0.7 that may affect these results:*

- **implementation_assumptions** (SIR confidence 0.58, lowest of all sections): reconstruction loss form (Huber vs MSE), alignment modality-pair scope, edge-weight mode, missing-modality imputation, and Leiden target-K are all assumption-driven — see `sir.json → ambiguities` for the full list with alternatives.
- **training_pipeline** (confidence 0.72): AdamW betas and LR schedule are unstated in the paper and were assumed.
- **architecture** (confidence 0.78): modality-aware fusion weighting formula and whether the encoder maintains per-modality branches (relevant to the alignment heads) are not fully specified.

---

## Verification Log Summary

- Comparison run at: 2026-07-22T09:00:00Z
- User results hash: see `verification_log.md`
- User-reported config modifications: dataset scaled from 11→4 samples, 4,992→800 spots/sample; training epochs 100→25; early-stopping patience 20→6; **data itself synthetic throughout** (dominant factor)
- Manual review required: **Yes** — comparison is against synthetic data at reduced scale, not a full-scale run against the real cohort (which is not publicly available to compare against in the first place)

Full audit trail in `verification_log.md`.
