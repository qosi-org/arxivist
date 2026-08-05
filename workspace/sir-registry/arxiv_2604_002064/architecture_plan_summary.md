# Architecture Plan Summary
**Paper**: Quantitative Universal Approximation for Noisy Quantum Neural Networks
**Paper ID**: arxiv_2604_002064
**Plan version**: 1

---

## 1. Framework Selection

- **Primary label**: `pytorch` (schema-forced default; see reasoning below)
- **Python**: 3.10+
- **CUDA**: not required (5-8 qubit circuits, CPU-feasible)
- **Config library**: plain YAML

**Reasoning**: This paper's real computational core is a **Qiskit** quantum circuit (state prep + parameterized unitary + measurement), executed on `qiskit-aer` (simulation) or `qiskit-ibm-runtime` (hardware). Qiskit isn't one of the schema's three allowed framework labels, so it's tracked as a required runtime dependency instead. For the classical piece — optimizing the circuit's few angles `theta`, and the two-parameter affine noise-correction layer from Theorem 3.15 — the paper compares three methods (L-BFGS-B, a two-stage method, and Adam). Per the planner's own default rule, PyTorch is nominated as the classical framework label (used for Adam and the differentiable correction layer); L-BFGS-B/two-stage use plain `scipy.optimize`.

## 2. Module Hierarchy

| File | Purpose |
|---|---|
| `models/qnn_circuit.py` | Builds V (Hadamard state prep) and U(θ,x) (UCR-decomposed block-diagonal unitary) |
| `models/measurement.py` | Groups raw counts into P0..P3, computes scalar QNN output |
| `models/noise_channels.py` | Depolarising channel, hardware-calibrated λ_V/λ_U, readout-error confusion matrix |
| `models/postprocessing.py` | Theorem 3.15 affine bias-cancellation layer |
| `data/dataset.py` | Gaussian-density and Black-Scholes-Put synthetic data generators |
| `data/transforms.py` | Input normalisation (Eq. 4.1) |
| `training/losses.py` | MSE training loss |
| `training/trainer.py` | Orchestrates Methods A/B/C |
| `evaluation/metrics.py` | RMSE, MAE, max error, bound-ratio |
| `evaluation/hardware_bounds.py` | Statement 2.1 / Theorem 3.6 / Theorem 3.17 / Prop. 3.20 bound calculators |
| `utils/config.py` | Config loading + seeding |

## 3. Tensor Flow Specification

Four forward passes are specified end-to-end: **NoiselessQNN**, **DepolarisingNoisyQNN** (adds noisy-probability mixing + readout error), **AffineCorrectedQNN** (Theorem 3.15 bias cancellation), and the **εtotal bound computation** used for the ibm_fez hardware validation (Eq. 4.4). See `architecture_plan.json → tensor_flows` for the full step-by-step pseudocode.

## 4. Configuration Schema

Config sections cover `model` (n accuracy blocks, qubit count, R-scaling rule, n0), `training` (method A/B/C selection, loss normalisation, Adam hyperparameters, seed), `data` (Gaussian and Black-Scholes parameter ranges), `evaluation` (metrics, shots), and `hardware` (ibm_fez calibration constants from Appendix A Table 1).

**Every field sourced from a SIR section with confidence < 0.7 carries an explicit `*_comment: "ASSUMED: ..."` annotation.** These are:
- `training.loss_normalisation_comment` (confidence 0.5 — PDF-extraction-corrupted formula)
- `training.adam_lr_comment`, `training.adam_hyperparams_comment` (confidence 0.3 — not stated in paper)
- `training.max_iterations_comment`, `training.seed_comment` (confidence 0.3 — not stated)
- `model.n0_comment` (confidence 0.4 — inferred from Remark 3.14)
- `hardware.readout_p_comment` (confidence 0.35 — missing from Appendix A Table 1 despite being used in Eq. 4.4)

## 5. Dependencies

**Runtime**: `qiskit`, `qiskit-aer`, `qiskit-ibm-runtime`, `numpy`, `scipy`, `torch`, `pyyaml`, `matplotlib`.
**Dev**: `pytest`, `jsonschema`, `black`, `flake8`.

## 6. Entrypoints

`train.py`, `evaluate.py`, `inference.py` (standard trio), plus two paper-specific scripts: `run_hardware.py` (ibm_fez execution + εtotal envelope, Section 4.5) and `simulate_noise.py` (depolarising-noise sweep, Section 4.4).

## 7. Docker / Runtime

Base image: `python:3.10-slim` (no CUDA needed — circuits are small and CPU-feasible). System deps: `git`, `build-essential`.

## 8. Risk Assessment

| Severity | Risk |
|---|---|
| **High** | Optimizer hyperparameters (lr, iteration budget) unspecified for all 3 methods |
| **High** | Readout error probability `p` used in the reported hardware bound is missing from the paper's own hardware table |
| **Medium** | Loss normalisation constant ambiguous (PDF extraction artifact) — low risk to optimum, but absolute loss values won't match |
| **Medium** | UCR gate decomposition delegated to external reference — wrong decomposition would silently change λ_U |
| **Medium** | `n0` padding value not confirmed for the main experiments |
| **Low** | Live ibm_fez hardware access not guaranteed to remain available/stable |
| **Low** | All datasets are synthetic/generated — no data-access risk |

Full detail (mitigations for each) is in `architecture_plan.json → risk_assessment`.

---

**Handling of low-confidence SIR sections** (per planner rule): `training_pipeline` (0.45) and the `n0`/readout-`p` assumptions (0.4/0.35) are all < 0.6, so the corresponding components (`QNNTrainer.fit_method_c_adam`, `HardwareNoiseCalibrator`, `ErrorBoundCalculator.total_bound_with_readout`) are designed to take these values as **overridable config parameters** rather than hardcoded constants, keeping them easily swappable once verified against the paper or live hardware.
