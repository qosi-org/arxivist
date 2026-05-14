# Hallucination Report
**Paper ID**: arxiv_2605_volatility_sig  
**Generated**: 2026-05-13  
**SIR Version**: 1 (confidence 0.86)  
**Auditor**: ArXivist Stage 6 (pre-submission pass — no user results)

This report audits the generated `volsig` codebase against the SIR for structural deviations,
parametric assumptions, and omissions. It is updated when user results are submitted.

---

## Summary

| Type | Count | Critical | Significant | Minor |
|------|-------|----------|-------------|-------|
| Structural | 0 | 0 | 0 | 0 |
| Parametric | 6 | 0 | 2 | 4 |
| Omission | 1 | 0 | 1 | 0 |
| **Total** | **7** | **0** | **3** | **4** |

No critical hallucinations detected. Three significant issues require attention before claiming paper-accurate results.

---

## Structural Hallucinations

*Structural hallucinations are components present in the generated code that are absent from or contradicted by the SIR.*

**None detected.** All 9 modules in the architecture plan map directly to components described in the paper. No extra components were invented.

---

## Parametric Hallucinations

*Hyperparameters assumed in the code that were not explicitly stated in the paper.*

### P1 — Euler Time Steps (`T_steps_per_unit: 252`)
- **Severity**: Significant
- **Location**: `configs/config.yaml`, `src/volsig/models/primary_process.py`
- **Evidence**: Paper states "Euler scheme" (Section 4.3) but never specifies the number of time steps per unit time. We assumed 252 (daily).
- **Risk**: If the paper used e.g. 500 steps/year, the Volterra fBM approximation would be more accurate, potentially explaining the lower errors reported for rough Bergomi. Conversely, if the paper used fewer steps (e.g. 50), our implementation is more expensive.
- **Suggested Fix**: Run a sensitivity sweep over `T_steps_per_unit ∈ {50, 100, 252, 500}` and record the loss at convergence. Match to paper's reported 3.5e-4 for rough Bergomi.
- **SIR Confidence**: 0.55

### P2 — Box Constraint Bounds (`box_bounds: [-10.0, 10.0]`)
- **Severity**: Significant
- **Location**: `configs/config.yaml`, `src/volsig/calibration/optimizer.py`
- **Evidence**: Paper (Section 4.3) states "box constraints on ℓ to accelerate convergence" but gives no numerical bounds.
- **Risk**: If bounds are too tight (e.g. [-3, 3]) the optimizer may be cut off before reaching the paper's ℓ*. If too loose, convergence slows. The paper's reported ℓ* has components in [-0.7, +1.4]; bounds of [-2, 2] or [-5, 5] may be more appropriate.
- **Diagnostic**: After calibration, check if any ℓ* components are at ±10. If so, widen bounds.
- **Suggested Fix**: Try `box_bounds: [-5.0, 5.0]` first; widen only if components hit the bounds.
- **SIR Confidence**: 0.45

### P3 — Initialization (`l0_init: zeros`)
- **Severity**: Minor
- **Location**: `src/volsig/calibration/optimizer.py`
- **Evidence**: Paper does not state the initialization of ℓ. Zero initialization is the most common choice.
- **Risk**: L-BFGS-B may converge to different local minima from different starting points (the loss landscape is non-convex due to MC pricing). The paper reports a specific ℓ* which may correspond to a different basin than zero-initialized.
- **Suggested Fix**: Try random warm starts `l0 ~ N(0, 0.1)` if zero-init converges to a high loss.
- **SIR Confidence**: 0.60

### P4 — X0 Interpretation in Heston Primary (`x0_is_variance: true`)
- **Severity**: Minor
- **Location**: `configs/config.yaml`, `src/volsig/models/primary_process.py`
- **Evidence**: Paper (Section 5.1) states X₀=0.1 for the Heston variance SDE (Eq. 4.2). The SDE models variance σ², so X₀=0.1 means initial variance=0.1 (initial vol≈0.316), which differs from the Heston market σ₀=0.2. The paper does not clarify whether X₀=0.1 is the initial variance or something else.
- **Risk**: If X₀ should be 0.04 (= σ₀²=0.2²) the primary process starts closer to the market model.
- **Diagnostic**: Check if ℓ[0] ≈ σ₀=0.2 (paper reports 0.201). If ℓ[0] is far from 0.2, X₀ may be wrong.
- **Suggested Fix**: Try `X0: 0.04` if results diverge.
- **SIR Confidence**: 0.60

### P5 — fBM Simulation Method (`fbm_method: cholesky`)
- **Severity**: Minor
- **Location**: `src/volsig/models/primary_process.py`
- **Evidence**: Paper references Bennedsen et al. (2017) hybrid scheme for market price generation but does not specify the simulation method for the primary process. We use Cholesky (O(T²) per path).
- **Risk**: At T_steps=252 and nMC=800k, the Volterra Riemann sum loop is O(T²·nMC) ≈ 5×10¹⁰ operations. This is the main computational bottleneck; the paper's 17-19 min runtime suggests either a faster method or fewer time steps.
- **Suggested Fix**: Implement the Hybrid scheme (Bennedsen et al. 2017) for primary process simulation, or reduce T_steps. Reference: `rough_bergomi.py` already imports this method for market generation.
- **SIR Confidence**: 0.75

### P6 — Cholesky Regularisation (`cholesky_reg_eps: 1e-8`)
- **Severity**: Minor
- **Location**: `src/volsig/signatures/compute.py`, `QMatrixAssembler.cholesky()`
- **Evidence**: Paper does not mention regularisation of the Q-matrix Cholesky decomposition. We add ε·I with ε=1e-8 to avoid numerical failure (Risk R7 in architecture plan).
- **Risk**: If ε is too large it biases the Q-matrix; if too small, Cholesky fails on some paths. 1e-8 is a standard choice for double precision.
- **Diagnostic**: Monitor the Cholesky failure count logged to console. It should be 0 or near-0.
- **Suggested Fix**: No action needed unless failure count > 0.1% of paths.
- **SIR Confidence**: 0.90

---

## Omission Hallucinations

*Components present in the SIR but absent or stubbed in the generated code.*

### O1 — VIX Option Pricing (`RoughBergomiModel.vix_option_atmi()`)
- **Severity**: Significant
- **Location**: `calibrate_rbergomi_vix.py`, `src/volsig/models/rough_bergomi.py`
- **Evidence**: Section 2.2 Step 2 requires the ATM implied volatility of VIX *options* at short maturity T≈0. This requires pricing European options on the VIX index, which itself is defined as:
  $$\text{VIX}_T = \sqrt{\frac{1}{\Delta} E_T\!\left[\int_T^{T+\Delta} \sigma_s^2\,ds\right]}$$
  Pricing VIX options requires a nested MC simulation (simulate variance paths conditional on time-T information). The current implementation uses a proxy (short-T ATM equity IV) which is **not** the same quantity.
- **Risk**: Step 2 (η estimation) will be inaccurate. This propagates to Step 3 (ρ via η). The VIX analytical calibration benchmark (IVVIX) in Table 6.1 may not match the paper.
- **Evidence from code**: `calibrate_rbergomi_vix.py` line marked `# STUB: VIX option pricing requires nested simulation`.
- **Suggested Fix**:
  1. Implement `RoughBergomiModel.vix_option_atmi(T_short, nMC_inner)` using the formula above.
  2. Simulate `nMC_inner` paths of $\int_T^{T+\Delta}\sigma_s^2\,ds$ for each outer path.
  3. Compute VIX_T from the conditional expectation.
  4. Price a European call on VIX_T with strike=VIX_0 using standard MC.
  5. Invert to get the implied volatility $I^{VIX}_T(0)$.
- **Severity Justification**: This affects only the rough Bergomi analytical benchmark (VIX), not the signature model itself. The SIG calibration in Table 6.1 is not affected. However, the IVVIX comparison column in the error table will be inaccurate.
- **SIR Confidence for this component**: 0.70

---

## Not-a-Hallucination Notes

The following items were reviewed and confirmed **not** to be hallucinations:

1. **Shuffle product table** (`QMatrixAssembler.__init__`): The precomputed shuffle table is an implementation choice (not explicitly described in the paper) but is a correct and necessary optimisation for the shuffle product computation required by Eq. 4.6. Confirmed correct against Definition 3.3.

2. **Reflection scheme for Heston variance** (`np.abs(V_pos)`): Standard practice for CIR processes in Euler discretisation; consistent with ensuring positivity. Paper does not contradict this.

3. **Left-point Riemann sum for stochastic integral** (`compute_stochastic_integrals`): Consistent with Itô convention and Euler discretisation. Paper says "Euler scheme" without specifying left/right/midpoint; left is the standard Itô convention.

4. **Normalising weights** (`weights = w / w.mean()`): This normalisation is not mentioned in the paper but is a standard numerical practice that does not change the argmin of L(ℓ), only the scale of the loss value. The reported loss values may differ by a constant factor — not a hallucination.

---

## Confidence in This Report

- **High confidence**: Structural assessment (no structural hallucinations)
- **High confidence**: O1 VIX omission (clearly marked as STUB in code)
- **Medium confidence**: P1 (Euler steps), P2 (box bounds) severity assessments
- **Low confidence**: P4 (X0 interpretation) — ambiguous in paper; may be correct as-is
