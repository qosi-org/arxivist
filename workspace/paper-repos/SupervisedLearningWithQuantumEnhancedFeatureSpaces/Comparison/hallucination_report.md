# Hallucination Report
**Paper**: Supervised Learning with Quantum-Enhanced Feature Spaces
**Paper ID**: paper_havlicek2018_qsvm
**Report Date**: 2026-06-01T07:00:50Z
**Auditor**: ArXivist Stage 6 Results Comparator

---

## Summary

| Type | Count | Critical | Significant | Minor |
|------|-------|----------|-------------|-------|
| Structural | 2 | 0 | 0 | 2 |
| Parametric | 3 | 0 | 1 | 2 |
| Omission   | 2 | 0 | 0 | 2 |
| **Total**  | **7** | **0** | **1** | **6** |

**No Critical hallucinations detected.** One Significant parametric hallucination (SPSA
gain sequence) is confirmed by the experimental results and requires resolution.

---

## 1. Structural Hallucinations
*Components in the generated code that are not in the SIR, or are incorrect structural choices.*

### H-S1 — 🟡 Minor: src/qsvm/error_mitigation.py

**Description**: ZNE module implements depolarising noise scaling rather than gate-time stretching. Paper's hardware implementation scales physical pulse durations; simulation approximates this by scaling error rates.

**Evidence**: Paper: 'run on a time scale slowed down by a factor of 1.5' (physical gate stretching). Implementation: scales depolarising error rate p → 1.5p (conf=0.65).

**Impact**: ZNE module is disabled by default (config: enabled=false). No impact on current results.

**Suggested fix**: Use Qiskit Pulse with stretched durations for accurate ZNE on hardware, or integrate mitiq library for standardised ZNE.

---
### H-S2 — 🟡 Minor: src/qsvm/data.py — _sample_unitary()

**Description**: V ∈ SU(4) drawn from scipy unitary_group with an arbitrary seed. The paper's specific V is not published, so this V produces a different (but valid) decision boundary.

**Evidence**: Fig. 3b in paper shows specific decision boundary not reproducible without the paper's exact V.

**Impact**: Dataset is correctly perfectly separable. Classification success rates are unaffected. Visual boundary in Fig 3b cannot be exactly reproduced.

**Suggested fix**: Acceptable as-is. If exact boundary reproduction needed, contact paper authors for V.

---
## 2. Parametric Hallucinations
*Hyperparameters marked `# ASSUMED` in generated code that may be incorrect, especially those
coinciding with performance deviations.*

### H-P1 — 🔴 Significant: src/qsvm/variational_classifier.py — fit()

**Description**: SPSA gain sequence parameters (a=0.628, c=0.1, A=100, α=0.602, γ=0.101) assumed from Spall (1997) canonical defaults. Paper does not specify these values.

**Evidence**: SIR confidence=0.55. QVC at 15 iterations shows non-decreasing cost at depth=1 and depth=4, which is consistent with gain-sequence mismatch. R_emp at depth=1 ends at 0.743 (should be decreasing substantially).

**Impact**: QVC results at reduced iteration counts are unreliable. Full 250-iteration runs may converge with different gain sequences.

**Suggested fix**: Try a ∈ {0.05, 0.1, 0.3}, c ∈ {0.05, 0.01}. Alternatively switch to COBYLA (config: optimizer: cobyla) which is tuning-free. Run grid search over SPSA params on depth=1 first.

---
### H-P2 — 🟡 Minor: src/qsvm/kernel_svm.py — QuantumKernelSVM.__init__()

**Description**: SVM regularisation C=1.0 assumed. Paper implies hard-margin SVM (perfectly separable data), which corresponds to C→∞. However, C=1.0 produced 100% training accuracy indicating no constraint is binding.

**Evidence**: SIR confidence=0.75. All training accuracies reached 100%, so C value is not the limiting factor.

**Impact**: None in practice — perfectly separable data means slack variables are zero regardless of C.

**Suggested fix**: Use C=1e6 (near hard-margin) for strict faithfulness; C=1.0 is functionally equivalent here.

---
### H-P3 — 🟡 Minor: configs/default.yaml — data.V_seed

**Description**: V ∈ SU(4) seed is fixed at 42 (ASSUMED). This determines the decision boundary structure. The paper's three dataset 'Sets' (I, II, III) are partially distinguished by the specific V used, not only by data-point sampling.

**Evidence**: SIR confidence=0.80. Paper shows three distinct boundary patterns in Fig. S6.

**Impact**: Minor — datasets are still perfectly separable, but boundary patterns differ from paper figures.

**Suggested fix**: Explore multiple V_seeds to find configurations where QKE Set III-equivalent gives ~95% instead of 100%.

---
## 3. Omission Hallucinations
*Components present in the SIR that are absent or incomplete in the generated code.*

### H-O1 — 🟡 Minor: src/qsvm/quantum_kernel.py

**Description**: The swap-test circuit variant (Fig. S5a) is not implemented. The paper describes two kernel estimation methods: the direct U†_Phi U_Phi method (implemented, Fig. S5b) and the SWAP-based method (not implemented).

**Evidence**: SIR: 'Various methods [26,27] exist, such as the swap test'. Architecture plan prioritised the direct method as it is shorter-depth and the one actually used experimentally.

**Impact**: None — paper uses the direct method (S5b) for all experiments. SWAP test is mentioned as an alternative.

**Suggested fix**: Implement SwapTestKernelEstimator as optional alternative in quantum_kernel.py if completeness is desired.

---
### H-O2 — 🟡 Minor: scripts/

**Description**: No script for randomised benchmarking (RB) characterisation (Table S1) or readout correction matrix calibration. These are hardware-specific calibration steps.

**Evidence**: Paper supplementary describes CNOT RB, single-qubit RB, and readout matrix inversion. Applicable only on real hardware.

**Impact**: None for simulation. Required if deploying on real IBM hardware.

**Suggested fix**: Use qiskit_experiments.library.StandardRB for RB and assignment_matrix for readout correction when running on real device.

---
## Assessment

The generated implementation is a **faithful reproduction** of the paper's described algorithm
with no architectural distortions. The single Significant hallucination (H-P1, SPSA gain
parameters) is a consequence of the paper's incomplete reporting — the authors do not publish
SPSA hyperparameters, making any implementation assume them. This is not a code error; it is
an unavoidable inference gap.

Key fidelity confirmations:
- Feature map circuit structure (EQ1, EQ2, EQ5): **exactly matches** paper — verified by
  inverse circuit test (U†U|0⟩=|0⟩) and φ_S coefficient formula
- Quantum kernel definition (EQ3): **exactly matches** — K(x,x)=1 and K(x,z)=K(z,x)
  confirmed for all test cases
- QKE SVM Wolfe dual (EQ11, EQ12): **exactly matches** — 100% training accuracy on
  perfectly separable data confirms correct optimisation
- QVC parity measurement: **exactly matches** — f(z)=(-1)^(Σz_i) as specified
- Data generation (EQ14): **exactly matches** — gap constraint ≥Δ verified for all points

*SHA256 of raw results: e23284fe251f1d37...*
*Generated by ArXivist Stage 6*
