# Benchmark Comparison Report

**Paper**: Quantitative Universal Approximation for Noisy Quantum Neural Networks
**Paper ID**: arxiv_2604_002064
**Comparison Date**: 2026-07-22
**Reproducibility Score**: 0.00 (medium confidence)

> **Source of "your" results**: no external user-supplied results were provided for this
> comparison. To produce a genuine (non-fabricated) comparison, ArXivist ran this repository's
> own generated code as a self-check: `train.py`-equivalent Method A (L-BFGS-B) fitting against
> the closed-form QNN expectation (`training/trainer.py`), on the full `configs/config.yaml`
> scale (`n_accuracy_blocks=8`, 1600-point Black-Scholes training grid; 100-point Gaussian
> grid), evaluated on the paper's exact 40x40 Black-Scholes grid (Section 4.3) and the
> `sigma=1.0` Gaussian grid (Section 2.3.1). The hardware fidelity factor `alpha` was computed
> directly from the ibm_fez calibration constants in `configs/config.yaml` (Appendix A Table 1),
> with no live hardware access. No results were run through the live ibm_fez backend.

## Metric Comparison

| Metric | Dataset | Paper Value | Your Value | Deviation | Severity |
|---|---|---|---|---|---|
| RMSE | Gaussian density (sigma=1, n=8) | 0.070027 | 0.002218 | -96.83% | Critical |
| MAE | Gaussian density (sigma=1, n=8) | 0.048624 | 0.001973 | -95.94% | Critical |
| max_error | Gaussian density (sigma=1, n=8) | 0.17234 | 0.003715 | -97.84% | Critical |
| MAE | Black-Scholes Put, Method A, noiseless, n=8 (40x40 eval grid) | 5.0326 | 7.3274 | +45.60% | Critical |
| max_error | Black-Scholes Put, Method A, noiseless, n=8 (40x40 eval grid) | 17.2535 | 21.1356 | +22.49% | Significant |
| alpha_fidelity_factor | ibm_fez hardware calibration (n=8, n_qubits=5) | 0.3650 | 0.9909 | +171.48% | Critical |
| MAE | Black-Scholes Put on ibm_fez hardware | 2.345 | -- | UNMATCHED | -- |
| total_error_bound | Black-Scholes Put on ibm_fez hardware (epsilon_total) | 18.578 | -- | UNMATCHED | -- |
| pearson_correlation | ibm_fez vs. comprehensive noise model | 0.9973 | -- | UNMATCHED | -- |

**Metrics compared**: 9 | **Matched**: 6 | **Unmatched**: 3 (all three require live ibm_fez
hardware access, which was not available for this self-check run)

## Summary

This self-check reproduction does **not** closely match the paper's reported numbers, in
either direction. On the Gaussian density problem, our closed-form fit substantially
**outperforms** the paper (errors ~25-45x smaller) -- almost certainly because the paper used
a different optimizer for this specific experiment (differential evolution, Section 4.2) and
its reported numbers reflect real shot-noise-sampled circuit output, while ours is a
noise-free closed-form expectation fit with L-BFGS-B. On the Black-Scholes pricing problem,
our reproduction **underperforms** the paper (MAE 45.6% higher, max error 22.5% higher),
consistent with a single-restart L-BFGS-B run landing in a worse local optimum than whatever
protocol the paper used, compounded by an unresolved ambiguity in how to recover
individual `(T, sigma)` values from the paper's `(K, sigma*sqrt(T))` evaluation-grid
parametrisation. Most strikingly, our computed hardware fidelity factor `alpha=0.99` is wildly
inconsistent with the paper's reported `alpha=0.365` -- see the Hallucination Report below,
which traces this to the paper's own approximate two-qubit gate-count formula producing an
implausibly low gate count when applied literally.

## Root Cause Analysis

### Gaussian density metrics (Critical, all in the "better than paper" direction)

1. **Optimizer mismatch (High probability).** Section 4.2 of the paper states the Gaussian
   experiment's parameters were optimised "by differential evolution," not L-BFGS-B/Adam
   (those are documented in Section 4.1 specifically for the Black-Scholes Methods A/B/C).
   `training/trainer.py` does not implement differential evolution at all (see Hallucination
   Report, omission #1). *Fix*: add `QNNTrainer.fit_differential_evolution` wrapping
   `scipy.optimize.differential_evolution`, and re-run this specific comparison with it.
2. **No shot noise in our evaluation (High probability).** The paper's Gaussian numbers come
   from real (or shot-simulated) circuit execution, subject to Eq. (4.2) statistical error;
   our comparison used the closed-form expectation directly (`trainer.predict`), which is the
   $N_{\\text{shots}} \\to \\infty$ limit and has no sampling noise. This alone would make the
   paper's numbers noisier (larger) than ours, in the observed direction. *Fix*: re-run using
   `MeasurementProcessor` + `AerSimulator` sampled outputs (as `scripts/run_hardware.py`
   already does for Black-Scholes) instead of the closed-form shortcut.
3. **Random initialization / restart count (Medium probability).** SIR ambiguities do not
   specify DE population size or generation count, or whether the paper used multiple random
   restarts; we ran a single seeded L-BFGS-B call. *Fix*: run multiple seeds and report a
   distribution rather than a point estimate.

### Black-Scholes Put MAE / max_error (Critical / Significant, "worse than paper" direction)

1. **Single-restart optimisation (High probability).** SIR `training_pipeline` confidence is
   0.45 precisely because restart count / initialization strategy for L-BFGS-B is unspecified.
   The 56-dimensional (`8 blocks x 7 params`) objective is a highly oscillatory sum of cosines
   -- a classic multi-modal landscape where a single L-BFGS-B run from one random start is
   prone to a poor local optimum. *Fix*: wrap `fit_method_a_lbfgsb` in a multi-restart loop
   (10-20 random inits, keep the best) and re-run.
2. **Evaluation-grid parametrisation ambiguity (Medium probability).** `data/dataset.py`'s
   `sample_eval_grid` docstring already flags that recovering individual `(T, sigma)` values
   from the paper's `(K, sigma*sqrt(T))` axis choice requires an assumed convention (we fix
   `T=1`); a different convention could shift the evaluated function meaningfully. *Fix*: sweep
   `T` within its stated training range instead of fixing it, and check result sensitivity.
3. **Payoff/truncation consistency (Low-Medium probability).** Section 2.3.3's sharp-bound
   worked example uses a *truncated* Put payoff for the Fourier-transform derivation, while our
   evaluation prices the actual (untruncated) Put via the standard closed-form formula -- the
   right target per Section 4.3, but worth double-checking end-to-end consistency between the
   training target and the theoretical-bound constant used.

### alpha_fidelity_factor (Critical, 171% deviation)

See **Hallucination Report** below (parametric hallucination #3) -- root cause is almost
certainly the paper's own approximate two-qubit gate-count formula (`N2Q ~= n*n_qubits/15`)
producing an implausibly small gate count (2.67) when applied literally to this configuration.

## Recommended Actions

1. **Highest impact**: implement and use differential evolution for the Gaussian experiment
   (addresses the largest, most systematic deviations).
2. **High impact**: replace the closed-form gate-count approximation for `lambda_U` with a
   count measured directly from the actual transpiled circuit (`circuit.count_ops()`), and
   re-validate `alpha` against the paper's reported 0.365.
3. **Medium impact**: add multi-restart optimisation to `fit_method_a_lbfgsb` / `fit_method_b_two_stage`.
4. **Medium impact**: re-run all comparisons using actual shot-sampled circuit output
   (`MeasurementProcessor` + `AerSimulator`/hardware) instead of the closed-form shortcut, to
   remove that confound entirely.
5. **Lower impact, still needed for full parity**: obtain live `ibm_fez` (or equivalent)
   hardware access to fill in the 3 UNMATCHED hardware metrics.
