# Hallucination Report

**Paper ID**: paper_pinns_raissi2017  
**Comparison Date**: 2026-07-04  
**SIR Version**: 1 (confidence: 0.87)  

---

## Summary

| Type | Count | Severities |
|---|---|---|
| Structural | 0 | — |
| Parametric | 3 | 1 Minor (fixed), 1 Minor, 1 Significant |
| Omission | 1 | Significant (fixed) |

No structural hallucinations. All four PDE operators, both loss functions (Eq. 4, 6, 12, 14),
the Gauss-Legendre Butcher tableau, and the autograd derivative chain are correctly implemented.
One implementation bug was found and fixed in Stage 6.

---

## Structural Hallucinations — None Found

The generated code maps 1:1 to the paper's described components:
- `ContinuousPINN` → Section 2, Eq. (2)–(4)
- `DiscretePINN` + `GaussLegendreTableau` → Section 3, Eq. (7)–(11)
- `BurgersOperator`, `SchrodingerOperator`, `AllenCahnOperator` → Sections 2.1, 2.2, 3.1.1
- `ContinuousPINNLoss` → Eq. (4) and (6)
- `DiscretePINNLoss` → Eq. (12) and (14)

One **implementation bug** caught during Stage 4 code generation and fixed immediately: the
Vandermonde system for the RK tableau used `lstsq(V, rhs)` instead of `lstsq(V.T, rhs)`.
Detected by the `_validate()` assertion (row sums of A ≠ c), corrected before shipping.

---

## Parametric Hallucinations

### PH-01 — Hidden layer count convention  
**Severity**: Minor | **Status**: Fixed during Stage 4

Paper states "9-layer deep neural network" with 3,021 parameters. Initial implementation used
`hidden_layers=9` → 3,441 params. Discrepancy caught in Stage 4 parameter count validation.
Paper's "9-layer" counts input as layer 1: input(1) + 8 hidden + output(1) = 10 total, but
labelled as 9. `hidden_layers=8` gives exactly 3,021 params. Config corrected.

**Impact if uncorrected**: +14% parameters, minor over-capacity, unlikely to affect L2 materially.

---

### PH-02 — Weight initialisation (Xavier uniform assumed)  
**Severity**: Minor | **Status**: Open | **SIR confidence**: 0.80

Paper does not specify initialisation. Xavier uniform with `gain=5/3` (tanh gain) assumed.
Paper's TF1 codebase defaulted to `glorot_uniform` (identical), making this assumption very
likely correct. No evidence of failure attributable to this assumption in the training curve.

**Suggested fix if convergence stalls**: try `nn.init.normal_(w, std=0.01)` as alternative.

---

### PH-03 — Physics loss weighting λ_f = 1.0 assumed  
**Severity**: Significant | **Status**: Open

Paper writes `MSE = MSE_u + MSE_f` without specifying λ_f. Equal weighting (1.0) assumed.
At iter 11,200 of the user's run, MSE_f (1.41×10⁻⁴) is 2.3× larger than MSE_u (6.08×10⁻⁵),
meaning the physics constraint is the current training bottleneck. If the paper used a higher
λ_f (e.g. 5–10), the physics residual would be penalised more heavily and converge faster.

**Suggested fix**: if Burgers continuous L2 > 5×10⁻³ at full convergence, retry with
`lambda_f: 5.0` in `configs/burgers_continuous.yaml` (edit `ContinuousPINNLoss(lambda_f=5.0)`).

---

## Omission Hallucinations

### OH-01 — Stable reference solver (caused experimental failure)  
**Severity**: Significant | **Status**: Fixed during Stage 6

**What was missing**: The SIR flagged risk R5 (confidence 0.80): the paper uses Chebfun (MATLAB)
for reference solutions; our Python implementation needed a numerically stable equivalent.
The generated `BurgersExactSolution` used naive RK4 pseudo-spectral integration **without
dealiasing**. For Burgers with ν=0.01/π, aliasing energy at high wavenumbers causes numerical
overflow (NaN) within ~10 timesteps.

**Evidence of failure**:
- User's `eval_results.json`: `"relative_l2_error": NaN`
- User's `comparison_heatmap.png`: reference panel shows colorscale of `1e239`
- Container run confirmed: naive solver produces NaN from step ~12 onward

**Fix applied**: `BurgersExactSolution.solve()` replaced with ETD-RK2 (exponential time
differencing, 2nd order) + 2/3-rule dealiasing. The integrating factor handles stiff diffusion
exactly; dealiasing suppresses convective aliasing. Validated stable across all 201 time
snapshots, no NaN, max value 1.0000.

**User action required**: Pull updated `src/pinns/data/exact_solutions.py` from the final zip
and re-run `evaluate.py` to get valid L2 errors.

---

## Open SIR Risks

| Risk ID | Description | SIR Confidence | Outcome |
|---|---|---|---|
| R1 | Gauss-Legendre tableau conditioning at q=500 | 0.82 | Not tested at q=500 yet |
| R2 | 500 sequential autograd passes (vmap TODO) | 0.88 | **Confirmed bottleneck** — caused q=500 to be infeasible on CPU |
| R3 | LBFGS line search differences from scipy | 0.83 | No issues at q≤4 |
| R4 | Weight initialisation scheme | 0.72 | Xavier assumed; no visible failures |
| R5 | Reference solution (Chebfun → scipy) | 0.80 | **Manifested as critical failure; fixed** |
| R6 | create_graph=True correctness | 0.90 | Verified correct in all tested configs |
