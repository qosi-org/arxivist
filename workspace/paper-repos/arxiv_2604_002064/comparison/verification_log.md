# Verification Log

**Comparison run timestamp**: 2026-07-22T00:45:00Z
**Paper ID**: arxiv_2604_002064
**ArXivist SIR version used**: 1 (`sir-registry/arxiv_2604_002064/versions/sir_v1.json`)
**Architecture plan version used**: 1 (`sir-registry/arxiv_2604_002064/architecture_plan.json`)

## Input Provenance

No external user-supplied results were provided to compare against. Per the Results
Comparator's role of rigorously comparing *actual* results (never fabricated ones), this
comparison run instead executed the repository's own generated code as a self-check:

| Run | Script/method | Config | Scale |
|---|---|---|---|
| 1 | `training/trainer.py::fit_method_a_lbfgsb` (equivalent to `train.py --method A`) | `configs/config.yaml` | `n_accuracy_blocks=8`, 1600-point Black-Scholes training grid, seed=42 |
| 2 | Evaluation via `training/trainer.py::predict` (closed-form) on `data/dataset.py::sample_eval_grid` | `configs/config.yaml` | Full 40x40 (1600-point) evaluation grid, matching Section 4.3 exactly |
| 3 | `training/trainer.py::fit_method_a_lbfgsb` on `data/dataset.py::GaussianDensityDataset` | sigma=1.0, 100-point grid, x in [-4,4], n_accuracy_blocks=8, seed=42 | Matches Section 2.3.1/4.2 exactly (data scale), optimizer differs (see hallucination report) |
| 4 | `models/noise_channels.py::HardwareNoiseCalibrator` direct calculation (no circuit execution) | ibm_fez params from `configs/config.yaml` (Appendix A Table 1) | Pure calculation, no live hardware |

Raw self-check output files (intermediate, not final deliverables):
`comparison/_self_check_eval_results.json`, `comparison/_self_check_gaussian_results.json`,
`comparison/_self_check_alpha_results.json`.

**SHA256 of concatenated self-check result files** (for traceability):
`8e9c463aa2563904e613efd075915b1bb6cc7bcd1ab47ce1216b8b24e1cb8d14`

## Paper Metrics vs. Matched Results

- Paper metrics available (from `sir.json -> evaluation_protocol.reported_results`): **9**
- Metrics matched to a self-check result: **6**
- Metrics left `UNMATCHED`: **3** (all three require live `ibm_fez` hardware access:
  hardware MAE, hardware total_error_bound, hardware Pearson correlation)

### All metric names compared

1. `RMSE` — Gaussian density (sigma=1, n=8) — MATCHED
2. `MAE` — Gaussian density (sigma=1, n=8) — MATCHED
3. `max_error` — Gaussian density (sigma=1, n=8) — MATCHED
4. `MAE` — Black-Scholes Put, Method A, noiseless, n=8 — MATCHED
5. `max_error` — Black-Scholes Put, Method A, noiseless, n=8 — MATCHED
6. `MAE` — Black-Scholes Put on ibm_fez hardware — UNMATCHED (no hardware access)
7. `total_error_bound` — Black-Scholes Put on ibm_fez hardware — UNMATCHED (no hardware access)
8. `pearson_correlation` — ibm_fez vs. comprehensive noise model — UNMATCHED (no hardware access)
9. `alpha_fidelity_factor` — ibm_fez hardware run — MATCHED (via direct calculation, not live hardware)

## Config Modifications From Paper-Stated Protocol

The following deviations from the paper's stated experimental protocol were made in this
self-check run, and are the primary suspects identified in the root cause analysis
(`benchmark_comparison.md`):

1. Gaussian-density experiment used L-BFGS-B (Method A), not differential evolution as
   Section 4.2 states was used for that specific experiment.
2. Both Black-Scholes and Gaussian evaluations used the closed-form (infinite-shot)
   expectation rather than shot-sampled circuit output (`N_shots=8192` per Section 4.1).
3. Single-seed, single-restart optimisation for all L-BFGS-B fits (paper's restart count,
   if any, is unspecified — SIR `training_pipeline` confidence 0.45).
4. `alpha` was computed from the paper's own stated formulas (Section 3.6) using an ASSUMED
   `n0=0`, rather than sourced from a live IBM Quantum calibration snapshot.
5. No live `ibm_fez` (or any other) hardware was accessed; `IBM_QUANTUM_TOKEN` was unset for
   this run, so `scripts/run_hardware.py`'s fallback path (AerSimulator) would have been used
   had it been invoked for this comparison (it was not — the closed-form path was used
   instead, for speed).

## SIR Confidence Scores Used in Score Computation

From `sir.json -> confidence_annotations`:
`architecture=0.80, mathematical_spec=0.90, tensor_semantics=0.75, training_pipeline=0.45,
evaluation_protocol=0.85, implementation_assumptions=0.45` → mean = 0.70

## Reproducibility Score Computation Trace

```
base_score = mean(1 - min(|pct_dev_i| / 50, 1.0)) over the 6 matched metrics
           = mean(0, 0, 0, 0.088, 0.5502, 0)   [Gaussian RMSE/MAE/max_error clip to 0,
                                                  BS MAE=0.088, BS max_error=0.5502, alpha clips to 0]
           = 0.10637

sir_confidence_penalty = (1 - 0.70) * 0.15 = 0.045
unmatched_penalty      = (3 / 9) * 0.2     = 0.0667

reproducibility_score = max(0, 0.10637 - 0.045 - 0.0667) = max(0, -0.00533) = 0.0
```

## Audit Conclusion

This comparison should be treated as a **partial, self-generated sanity check**, not a
faithful reproduction of the paper's experimental protocol. The 0.0 reproducibility score
reflects genuine, non-trivial deviations in both directions (over- and under-performing the
paper depending on the metric) rather than a uniform failure — see `benchmark_comparison.md`
for the full root-cause breakdown. `requires_manual_review: true` is set in
`reproducibility_score.json` accordingly. A trustworthy reproduction attempt should, at
minimum, address the two Significant/Critical omission and parametric hallucinations
identified (differential-evolution optimizer; two-qubit gate-count formula for `alpha`)
before this score should be expected to improve materially.
