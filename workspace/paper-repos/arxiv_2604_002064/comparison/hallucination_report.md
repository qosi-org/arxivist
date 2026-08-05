# Hallucination Report

**Paper**: Quantitative Universal Approximation for Noisy Quantum Neural Networks
**Paper ID**: arxiv_2604_002064
**Report Date**: 2026-07-22

This report reviews `architecture_plan.json` and the generated source code against
`sir.json` to identify components that were invented, misconfigured, or omitted relative
to what the paper actually specifies.

## Structural Hallucinations

**None found.** Every class in `architecture_plan.json → module_hierarchy` traces back to
either (a) one of the 3 SIR `architecture.modules` entries (`state_preparation_V`,
`parameterized_unitary_U`, `measurement`), or (b) a directly-named SIR
`mathematical_spec` equation (`noise_channels.py` implements Definition 3.8 / Proposition
3.12 / Section 3.6 explicitly; `postprocessing.py` implements Theorem 3.15 explicitly;
`hardware_bounds.py` implements Statement 2.1 / Theorem 3.6 / Theorem 3.17 / Proposition
3.20 explicitly). No component was added without a clear textual anchor in the paper.

## Parametric Hallucinations

| # | Parameter | Location | Severity | Evidence | Suggested Fix |
|---|---|---|---|---|---|
| 1 | Two-qubit gate count formula for `lambda_U`, `N2Q ~= n_accuracy_blocks * n_qubits / 15` | `models/noise_channels.py::HardwareNoiseCalibrator.naive_and_ucr_two_qubit_gate_counts` | **Critical** | Computed `alpha=0.9909` vs. paper's reported `alpha=0.365` (Section 4.5, Fig. 4.2 panel f) -- a 171% deviation. The formula, applied literally to `n_accuracy_blocks=8, n_qubits=5`, yields `N2Q~=2.67`, implausibly low for a real Moettoenen-style UCR gate on 3 control qubits (which typically needs O(2^3)~=8 CNOTs per UCR gate, ~16 total for the UCRZ+UCRY pair used here) -- suggesting either the paper's own "roughly 15" approximation is far looser than it reads, or additional context (e.g. it applies to a different `N2Q` baseline) is missing. Already flagged in `sir.json -> ambiguities[3]` at confidence 0.6, but the magnitude of the resulting alpha discrepancy is larger than that confidence score alone conveys. | Replace the formula-based estimate with a gate count measured directly from the actual transpiled circuit (`qiskit.transpile(circuit, basis_gates=[...]); circuit.count_ops()`), then recompute `lambda_U`/`alpha` from that real count and re-validate against the paper's reported 0.365. |
| 2 | Adam hyperparameters (`lr=0.001, beta1=0.9, beta2=0.999`) | `configs/config.yaml -> training.adam_lr/adam_beta1/adam_beta2` | Minor | Not stated in the paper (SIR `implementation_assumptions[0]`, confidence 0.3); did not directly affect this comparison run since Method A (L-BFGS-B) was used, not Method C. | Sweep a small grid of `lr` values against a held-out slice of training data before trusting Method C results specifically. |
| 3 | Readout error probability `p=0.01` | `configs/config.yaml -> hardware.readout_p` | Minor (for this run) | Not listed in the paper's own Appendix A Table 1 despite being used in Eq. (4.4) (SIR `implementation_assumptions[5]`, confidence 0.35). Did not affect this run's matched metrics (the 3 metrics that depend on it -- MAE/total_error_bound/correlation on hardware -- were UNMATCHED here). | Replace with a live-calibration value from `qiskit_ibm_runtime` backend properties before trusting any `epsilon_total` computation. |

## Omission Hallucinations

| # | Component | Present in SIR? | Present in generated code? | Severity | Evidence | Suggested Fix |
|---|---|---|---|---|---|---|
| 1 | Differential-evolution optimizer for the Gaussian-density experiment | Implied by `sir.json` provenance (Section 4.2 states "we... optimise the parameters by differential evolution"), though not captured as a distinct `mathematical_spec` or `training_pipeline` entry -- the SIR's `training_pipeline` section only captured the Black-Scholes Methods A/B/C (L-BFGS-B, two-stage, Adam) from Section 4.1 | **No.** `training/trainer.py`'s `QNNTrainer` only implements `fit_method_a_lbfgsb`, `fit_method_b_two_stage`, `fit_method_c_adam` | **Significant** | Directly implicated in the largest deviations found in this comparison: our Gaussian-density reproduction (using L-BFGS-B, the only optimizer implemented) produced RMSE/MAE/max_error 25-45x *smaller* than the paper's reported values, consistent with the paper using a different, and apparently less sample-efficient in this closed-form-equivalent setting, optimizer for that specific experiment. | Add `QNNTrainer.fit_differential_evolution(...)`, wrapping `scipy.optimize.differential_evolution`, wire it as the default for Gaussian-density experiments specifically, and re-run this comparison. |
| 2 | Shot-based (finite-`N_shots`) training loop | Implied throughout Section 4.1 ("Parameters optimisation" operates on outputs of a real/simulated circuit, which the paper's own validation section (4.1.1) treats as subject to shot noise) | **Partially.** `training/trainer.py` optimises against the closed-form expectation (`_closed_form_forward_batch`) rather than actual `MeasurementProcessor`+circuit-sampled outputs -- an explicitly documented engineering trade-off in the module's own docstring, not a silent omission, but still a functional gap relative to the paper's likely training regime | Moderate | Directly contributes to the Gaussian-density deviations (see root cause analysis in `benchmark_comparison.md`) since the closed-form path has zero shot noise by construction. | Add an alternative `QNNTrainer.fit_shot_based(...)` path that trains against `MeasurementProcessor.qnn_output` sampled via `AerSimulator`, for use when exact reproduction of the paper's shot-noise-influenced numbers is required. |

## Summary

Zero structural hallucinations were found -- the generated code faithfully mirrors the SIR's
architecture and mathematical specification with no invented components. The parametric and
omission hallucinations found are concentrated in exactly the areas the SIR itself flagged as
low-confidence (`training_pipeline`, `implementation_assumptions`, and `ambiguities[3]`'s UCR
gate-count formula) — the confidence scoring correctly anticipated where the largest
reproduction risk would materialize. The single Critical parametric hallucination
(two-qubit gate count formula → `alpha`) is the most actionable finding in this report and
should be fixed first, per `benchmark_comparison.md`'s Recommended Actions.
