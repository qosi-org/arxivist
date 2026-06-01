# Scientific Intermediate Representation (SIR)
## Supervised Learning with Quantum-Enhanced Feature Spaces
**Havlicek et al., IBM / MIT, arXiv:1804.11326v2**
**paper_id:** `paper_havlicek2018_qsvm` | **SIR v1** | **Overall Confidence: 0.86**

---

## 1. Provenance

| Field | Value |
|---|---|
| Authors | Havlicek, Corcoles, Temme, Harrow, Kandala, Chow, Gambetta |
| arXiv | 1804.11326v2 (Jun 2018) |
| Domain | Quantum Machine Learning |
| Hardware | 5-qubit superconducting transmon (IBM), qubits Q0 and Q1 |

**Key Claims:**
1. Quantum feature maps embed classical data into exponentially large Hilbert spaces conjectured to be classically hard
2. Quantum Variational Classifier (QVC) achieves up to 100% success on real hardware
3. Quantum Kernel Estimator (QKE) is a direct quantum-enhanced analog of classical kernel SVM
4. The feature map is related to the hidden-shift problem for bent Boolean functions (hardness conjecture)
5. Zero-noise extrapolation (Richardson) substantially improves results for deeper circuits

---

## 2. Architecture Graph (Confidence: 0.91)

Two distinct protocols share one feature map:

```
INPUT x ∈ (0,2π]²
        │
        ▼
┌───────────────────────────────────────────────┐
│  FEATURE MAP  U_Φ(x) = U_Φ H⊗ⁿ U_Φ H⊗ⁿ      │
│                                               │
│  U_Φ(x) = exp(i Σ_{S⊆[n]} φ_S(x) Π_{i∈S} Zᵢ)│
│                                               │
│  For n=d=2:                                   │
│    φ_{i}(x) = xᵢ                             │
│    φ_{1,2}(x) = (π−x₁)(π−x₂)                │
│                                               │
│  Gate decomp: ZkZl via CNOT+Z (Fig 1c)        │
└───────────────────┬───────────────────────────┘
                    │
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
  ┌───────────────┐   ┌──────────────────────────┐
  │ PROTOCOL 1    │   │ PROTOCOL 2 (QKE)         │
  │ QVC           │   │                          │
  │               │   │ Apply U†_Φ(xⱼ) U_Φ(xᵢ) │
  │ W(θ) =        │   │ to |0ⁿ⟩; count |0...0⟩  │
  │ U^(l)_loc     │   │ outcomes → K(xᵢ,xⱼ)     │
  │ Uₑₙₜ ... l×  │   │                          │
  │               │   │ 50,000 shots/entry       │
  │ Uₑₙₜ = ΠCZ   │   │ ZNE error mitigation     │
  │               │   └──────────┬───────────────┘
  │ Meas: Z₁Z₂   │              │
  │ parity f(z)   │              ▼
  │               │   ┌──────────────────────────┐
  │ SPSA opt:     │   │ CLASSICAL SVM (precomp K)│
  │ 250 iters     │   │ Solve Wolfe dual L_D(α)  │
  │ R=200 (cost)  │   │ → support vectors α*     │
  │ R=2000 (eval) │   │ → bias b from KKT        │
  └───────┬───────┘   └──────────┬───────────────┘
          │                      │
          ▼                      ▼
  ┌───────────────┐   ┌──────────────────────────┐
  │ Decision:     │   │ Decision:                │
  │ ỹ if          │   │ sign(Σᵢ yᵢα*ᵢK(xᵢ,s)+b)│
  │ p̂_y > p̂_{-y}│   │                          │
  │ − yb          │   │                          │
  └───────────────┘   └──────────────────────────┘
```

**Hardware (n=2 qubits):**
- T1 = [55, 38] μs, T2* = [16, 17] μs, CNOT error = 0.0373
- Readout fidelity ~95%; readout correction matrix applied
- Zero-noise extrapolation: stretch gate times by 1.5×; Richardson extrapolate

---

## 3. Mathematical Specification (Confidence: 0.94)

### Core Equations

| ID | Name | Expression |
|---|---|---|
| EQ1 | Diagonal phase gate | `U_Φ(x) = exp(i Σ_{S⊆[n]} φ_S(x) Π_{i∈S} Zᵢ)` |
| EQ2 | Full feature map | `𝒰_Φ(x) = U_Φ H^⊗n U_Φ H^⊗n` |
| EQ3 | Quantum kernel | `K(x,z) = │⟨Φ(x)│Φ(z)⟩│² = │⟨0ⁿ│U†_Φ(x) U_Φ(z)│0ⁿ⟩│²` |
| EQ4 | Encoded state | `│Φ(x)⟩ = Σ_p Φ_x(p) Φ̂_x(p) │p⟩` (Walsh-Fourier) |
| EQ5 | Coefficients (n=2) | `φ_{i}(x) = xᵢ`, `φ_{1,2}(x) = (π−x₁)(π−x₂)` |
| EQ6 | Meas. probability | `p_y(x) = ⟨Φ(x)│W†(θ) M_y W(θ)│Φ(x)⟩` |
| EQ7 | Empirical risk | `R_emp(θ) = (1/│T│) Σ_{x∈T} Pr(ỹ ≠ y)` |
| EQ8 | Sigmoid approx | `Pr(ỹ ≠ y) ≈ sig(√R · ((1+yb)/2 − p̂_y) / √(2(1−p̂_y)p̂_y))` |
| EQ9 | QVC decision | `ỹ = y if p̂_y(x) > p̂_{-y}(x) − yb` |
| EQ10 | QVC–SVM equiv | `ỹ = sign(2^{-n} Σ_α w_α(θ) Φ_α(x) + b)` |
| EQ11 | Wolfe dual (QKE) | `L_D(α) = Σᵢ αᵢ − ½ Σᵢⱼ yᵢ yⱼ αᵢ αⱼ K(xᵢ,xⱼ)` |
| EQ12 | QKE classifier | `ỹ(s) = sign(Σᵢ yᵢ α*ᵢ K(xᵢ,s) + b)` |
| EQ13 | Sampling complexity | `R = O(ε⁻² │T│⁴)` total shots for ‖K−K̂‖ ≤ ε |
| EQ14 | Data generation | `m(x) = sign(⟨Φ(x)│V† (Z₁Z₂) V│Φ(x)⟩ ≷ ±Δ)`, Δ=0.3 |

---

## 4. Tensor Semantics (Confidence: 0.78)

| Tensor | Shape | Dtype | Role |
|---|---|---|---|
| `x_input` | `[2]` | float64 | Classical input ∈ (0,2π]² |
| `feature_state` | `[4]` complex | complex128 | │Φ(x)⟩ statevector (n=2) |
| `kernel_matrix_K` | `[40, 40]` | float64 | Training kernel (20+20 pts) |
| `alpha_dual` | `[40]` | float64 | SVM dual variables |
| `theta_qvc` | `[4*(l+1)]` | float64 | QVC variational params |
| `empirical_probs` | `[2]` | float64 | p̂_{+1}, p̂_{-1} per data pt |
| `shot_counts` | `[4]` | int64 | Raw bitstring counts |

---

## 5. Training Pipeline (Confidence: 0.85)

### Protocol 1 — QVC
| Parameter | Value |
|---|---|
| Optimizer | SPSA (Spall 1997/2000) |
| SPSA iterations | 250 |
| Shots for cost eval | R = 200 |
| Shots for actual experiment | 2000 |
| Shots for classification | 10,000 |
| Training set | 20 pts/label = 40 total |
| Circuit depths | l ∈ {0,1,2,3,4} |
| Training sets per depth | 3 |
| Error mitigation | ZNE (Richardson, 1.5× stretch) at every SPSA step |
| Bias | b optimized jointly; b* = 0 at inference |

### Protocol 2 — QKE
| Parameter | Value |
|---|---|
| Shots per kernel entry | 50,000 |
| Kernel size | 40 × 40 |
| Error mitigation | ZNE per kernel entry |
| Classical solver | Wolfe dual QP (sklearn precomputed kernel) |
| Training sets | 3 (Set I, II, III) |
| Test sets per training set | 10 |

---

## 6. Evaluation Protocol (Confidence: 0.96)

**Dataset:** Synthetic, generated with fixed V ∈ SU(4), gap Δ=0.3

| Method | Set | Result |
|---|---|---|
| QVC depth=0 | multiple | ~60–75% |
| QVC depth=1–4 | multiple | ~85–100% |
| QVC depth=4 | inset | **100%** |
| QKE | Set I | **100%** (all 10 test sets) |
| QKE | Set II | **100%** (all 10 test sets) |
| QKE | Set III | **94.75%** |

---

## 7. Implementation Assumptions (Confidence: 0.72)

| # | Assumption | Basis | Conf. | ⚠ |
|---|---|---|---|---|
| A1 | Qiskit for circuit construction | IBM hardware used | 0.85 | |
| A2 | SPSA defaults from Spall 1997 | Paper doesn't give gain params | 0.55 | ⚠ |
| A3 | sklearn SVC(kernel='precomputed') | Standard practice | 0.82 | |
| A4 | Statevector sim (exact) for ideal case | Standard baseline | 0.75 | |
| A5 | ZNE via depolarizing noise injection | Without real hardware | 0.65 | ⚠ |
| A6 | Fixed seed for V ∈ SU(4) | Fig 3b shows fixed boundary | 0.80 | |
| A7 | PSD enforcement: clip negative eigenvalues | Paper mentions issue | 0.70 | |

⚠ = Low confidence; flag for human review before Stage 3.

---

## 8. Confidence Summary

| Section | Confidence |
|---|---|
| Architecture | 0.91 |
| Mathematical Spec | 0.94 |
| Tensor Semantics | 0.78 |
| Training Pipeline | 0.85 |
| Evaluation Protocol | 0.96 |
| Implementation Assumptions | 0.72 |
| **Overall SIR** | **0.86** |

**human_review_required: No** (overall > 0.65; two low-confidence assumptions flagged above)

---

## 9. Key Ambiguities Requiring Resolution Before Stage 3

1. **SPSA hyperparameters** (conf 0.55): Spall canonical defaults will be used; may need tuning
2. **Exact V ∈ SU(4)** (conf 0.75): Will use fixed seed; boundary shape will differ but algorithm behavior preserved
3. **ZNE in simulation** (conf 0.65): Two options — skip (ideal sim) or inject depolarizing noise at 1×/1.5× levels
4. **QP solver** (conf 0.80): Will default to sklearn; can switch to cvxopt if numerical issues arise

---

*SIR written to: `sir-registry/paper_havlicek2018_qsvm/sir.json` (v1)*
*Pipeline state: Stage 3 (Architecture Planner) ready*
