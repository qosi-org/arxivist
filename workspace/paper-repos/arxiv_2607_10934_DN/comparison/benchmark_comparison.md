# Benchmark Comparison Report

**Paper**: Multidimensional stochastic liquidity in Kyle's model of informed trading
**Paper ID**: arxiv_2607_10934
**arXiv**: https://arxiv.org/abs/2607.10934
**Comparison Date**: 2026-07-15T18:38:54Z
**SIR Version Used**: 1

---

## Important scope note

This paper reports **no numerical results table** — it is a pure theory paper whose
"evaluation" is the analytical recovery of five known closed-form Kyle-Back equilibria
as special cases (Section 5). There is therefore no external experimental run to compare
against. The "comparison" below is a **self-verification**: does the generated code's
simulated output match the paper's own closed-form formulas, to the precision expected
of an exact mathematical identity (not a statistical benchmark)?

## Reproducibility Score

| Score | Confidence | Metrics Compared | Matched |
|-------|------------|-------------------|---------|
| **0.8703** / 1.0 | medium | 6 | 4 |

**Interpretation**: 0.75-0.89 = good reproduction with minor deviations. The score is
held below "excellent" (>0.90) primarily by the SIR-confidence penalty (0.58 overall
SIR confidence, since this is a theory paper whose SIR sections were mostly
reinterpreted from an ML-shaped schema) and the unmatched-by-scope penalty (2 of 6
paper claims -- Sections 5.5 and 5.7 -- are out of scope by design, not implementation
failures; see below).

---

## Metric Comparison Table

| Metric | Case | Paper Value | Your Value | Deviation | Severity |
|--------|------|-------------|------------|-----------|----------|
| terminal_covariance_identity (eq. 4.6) | kyle1985 (synthetic, C=1.0, sigma=1.0, T=1.0) | 1 | 0.999998 | 2.00e-04% | excellent |
| terminal_covariance_identity (eq. 4.6) | back_pedersen1998 (synthetic, C=1.0, sigma(t)=1+0.5sin, T=1.0) | 1 | 0.999998 | 1.78e-04% | excellent |
| terminal_covariance_identity, Monte Carlo mean (eq. 4.6) | collin_dufresne_fos2016 (synthetic stochastic sigma_t, 50 MC paths) | 1 | 1 | 8.08e-06% | excellent |
| M*_0 agreement: eigenbasis-route vs direct-matrix-closed-form (Section 5.3 vs 5.6) | common_eigenbasis_bcel2020 (synthetic, n_assets=3, random SPD C) | 1 | 1 | 1.91e-13% | excellent |

All four matched checks are **excellent** (well under 2% -- in fact at or near
machine/quadrature precision), consistent with these being exact closed-form identities
rather than noisy empirical benchmarks.

---

## Deviation Summary

| Severity | Count |
|----------|-------|
| Excellent (<=2%) | 4 |
| Good (2-5%) | 0 |
| Moderate (5-15%) | 0 |
| Significant (15-30%) | 0 |
| Critical (>30%) | 0 |
| Unmatched (out of scope) | 2 (Section 5.5, Section 5.7 -- see Hallucination Report) |

---

## Root Cause Analysis

No Moderate/Significant/Critical deviations were found among the four implemented and
matched checks -- all residuals are consistent with floating-point/quadrature precision,
not an implementation error. The two **unmatched** paper claims are not deviations but
scope decisions:

### Section 5.5 (scalar stochastic liquidity via stochastic maximum principle) -- unmatched

1. **No numerical BSDE solver specified in the paper** (High probability, by design)
   Fix: implement a quadratic BSDE solver (deep BSDE / Han-Jentzen-E or LSMC) for eq. (5.44)-(5.45) if this case is needed.

### Section 5.7 (general, non-commuting matrix FBSDE) -- unmatched

1. **Explicitly an open problem in the source paper** (High probability, by design)
   Fix: none available from this pipeline; `fbsde_stub.py` documents the gap and raises on use rather than guessing.

---

## Hallucination Report Summary

See `hallucination_report.md` for the full report.

| Type | Count | Critical |
|------|-------|----------|
| Structural | 0 | 0 |
| Parametric | 2 | 0 |
| Omission | 2 | 0 |

No structural hallucinations (no components were added beyond the SIR's architecture
graph) or critical parametric/omission hallucinations were found.

---

## Recommended Actions

1. If Section 5.5 coverage matters for your use case, add a quadratic-BSDE solver (biggest expected reproducibility-score gain).
2. Increase `training.n_steps` in `configs/config.yaml` if tighter terminal-price agreement (not just identity-residual agreement) is desired.
3. Leave `fbsde_stub.py` (Section 5.7) untouched unless you are specifically researching that open problem -- it is correctly non-functional by design.

---

## Implementation Notes

*From the SIR -- sections with confidence < 0.7 that may affect these results:*

- `architecture` (0.62): reinterpreted from an ML-module schema onto stochastic-control components.
- `tensor_semantics` (0.55): reinterpreted onto matrix/vector-valued stochastic processes rather than NN tensors.
- `training_pipeline` (0.15): not applicable -- no training occurs in this repo.
- `evaluation_protocol` (0.55): no empirical benchmark table exists in the paper.
- `implementation_assumptions` (0.55): several numerical-scheme choices (discretization, BSDE solver, dimension) were necessarily assumed since the paper is continuous-time and abstract throughout.

---

## Verification Log Summary

- Comparison run at: 2026-07-15T18:38:54Z
- Results hash: `8a0e41285b4b147cb79c573e94c0f988ac6f1eb801f8588287ac7e7d4b2c5cf3`
- User-reported config modifications: none
- Manual review required: yes -- overall_sir_confidence (0.58) is below the pipeline's 0.65 threshold; Section 5.5 and 5.7 paper claims are unmatched by design (out of scope / explicitly open problem), not because of an implementation failure; There is no independent human-run experiment to compare against -- these are self-verification checks of the generated code against the paper's own closed-form formulas, since the paper reports no numerical results table

Full audit trail in `verification_log.md`.
