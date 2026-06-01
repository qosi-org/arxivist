# Architecture Plan — Havlicek et al. 2018 (QSVM)
**paper_id:** `paper_havlicek2018_qsvm` | **Plan v1** | **Confidence: 0.91**

---

## Framework Selection

| Decision | Choice | Rationale |
|---|---|---|
| Quantum SDK | **Qiskit 0.45.3** | IBM hardware origin; canonical IBM SDK |
| Simulator | **Qiskit Aer 0.13.3** | Statevector (exact) + noisy backends |
| Classical ML | **scikit-learn 1.4.2** | `SVC(kernel='precomputed')` for QKE |
| Numerics | NumPy 1.26 + SciPy 1.13 | Matrix ops; `unitary_group` for random V |
| GPU | **Not required** | n=2 qubits; statevector is 4-dim |
| Python | 3.10+ |  |
| Config | YAML + dataclasses | Lightweight; no Hydra needed |

---

## Module Hierarchy

```
qsvm_paper/
├── src/qsvm/
│   ├── __init__.py
│   ├── feature_map.py            ← FeatureMap class      [EQ1,EQ2,EQ5]  conf=0.97
│   ├── quantum_kernel.py         ← QuantumKernelEstimator [EQ3,EQ13]    conf=0.95
│   ├── variational_classifier.py ← QuantumVariationalClassifier [EQ6-9] conf=0.88
│   ├── kernel_svm.py             ← QuantumKernelSVM      [EQ11,EQ12]    conf=0.92
│   ├── error_mitigation.py       ← ZeroNoiseExtrapolation               conf=0.72 ⚠
│   ├── data.py                   ← SyntheticQuantumDataset [EQ14]       conf=0.90
│   ├── metrics.py                ← ClassificationMetrics                conf=0.95
│   └── utils.py                  ← Plotting helpers (Figs 3,4)
│
├── configs/
│   └── default.yaml              ← All hyperparameters (see below)
│
├── scripts/
│   ├── train_qvc.py              ← CLI: QVC training + eval
│   ├── train_qke.py              ← CLI: QKE kernel + SVM
│   ├── run_all.py                ← Full paper reproduction
│   └── plot_results.py           ← Fig 3 + Fig 4 regeneration
│
├── tests/
│   ├── test_feature_map.py
│   ├── test_kernel.py
│   ├── test_qvc.py
│   └── test_qke.py
│
├── docker/Dockerfile
├── notebooks/                    ← Stage 5 output
└── results/
```

---

## Module Contracts

### `FeatureMap` — `feature_map.py`
Maps x ∈ (0,2π]² → |Φ(x)⟩ via 𝒰_Φ(x) = U_Φ H⊗ⁿ U_Φ H⊗ⁿ

```
__init__(n_qubits=2, reps=2, entanglement='linear')
get_circuit(x: [2]) → QuantumCircuit
phi_coefficients(x: [2]) → Dict  # {1}→x1, {2}→x2, {1,2}→(π-x1)(π-x2)
get_statevector(x: [2], backend) → [4] complex128
```

**Gate decomposition:** single-qubit phases as `RZ(2φ_i)`; two-qubit ZZ coupling via `CNOT–RZ–CNOT` (Fig 1c).

---

### `QuantumKernelEstimator` — `quantum_kernel.py`
Computes K(x,z) = |⟨Φ(x)|Φ(z)⟩|²

```
__init__(feature_map, backend, shots=1024, use_statevector=True)
evaluate(x: [2], z: [2]) → float
build_kernel_matrix(X: [N,2], Y=None) → [N,N] or [N,M]
enforce_psd(K: [N,N]) → [N,N]
```

**Two modes:**
- `use_statevector=True` (default sim): inner product of statevectors — exact, no shot noise
- `use_statevector=False` (hardware emulation): apply U†_Φ(x)U_Φ(z), count `|00⟩` outcomes

---

### `QuantumVariationalClassifier` — `variational_classifier.py`
Full QVC with SPSA training

```
__init__(feature_map, depth: int, backend, shots=1024)
build_circuit(x: [2], theta: [4*(l+1)]) → QuantumCircuit
get_probs(x, theta, b=0.0) → {+1: float, -1: float}
predict(x, theta, b=0.0) → int
cost_function(theta, X_train, y_train, b, R_cost=200) → float  # R_emp via EQ8
fit(X_train, y_train, n_iter=250) → (theta_star, b_star, cost_history)
score(X_test, y_test, theta, b=0.0) → float
```

**W(θ) circuit:** `U^(l)_loc U_ent … U^(1)_loc`
- `U^(t)_loc` = ⊗ of `exp(i/2 θ_z Z) exp(i/2 θ_y Y)` per qubit
- `U_ent` = `CZ(0,1)` for linear 2-qubit chain

---

### `QuantumKernelSVM` — `kernel_svm.py`
Wraps sklearn SVC with quantum precomputed kernel

```
__init__(kernel_estimator, C=1.0)
fit(X_train: [40,2], y_train: [40]) → None
predict(X_test: [N,2]) → [N] ∈ {+1,-1}
score(X_test, y_test) → float
get_support_vectors() → (SVs, alphas, labels)
decision_function_values(X_test) → [N]  # for Fig 3b bottom panel
```

---

### `ZeroNoiseExtrapolation` — `error_mitigation.py` ⚠ conf=0.72
Optional; disabled by default in simulation

```
__init__(scale_factors=[1.0, 1.5], order=1)
mitigate_expectation(noisy_values: [2]) → float   # Richardson: (s2*E1-s1*E2)/(s2-s1)
apply_to_circuit(circuit, observable_fn, noise_model) → float
```

**Config flag:** `error_mitigation.enabled: false` — no-op passthrough in ideal sim.

---

### `SyntheticQuantumDataset` — `data.py`
Generates paper's artificial data (EQ14)

```
__init__(n_per_label=20, gap=0.3, seed=42, n_qubits=2)
generate() → (X: [40,2], y: [40])
label_point(x, V, feature_map, gap) → Optional[int]  # rejection sampling
split(test_n_per_label=20) → X_train, y_train, X_test, y_test
```

**Label rule:** m(x) = sign(⟨Φ(x)|V†(Z₁Z₂)V|Φ(x)⟩) if |val| ≥ Δ=0.3, else reject.

---

## Data Flow Diagrams

### QKE Full Pipeline
```
X_train [40,2] ──► build_kernel_matrix ──► K_train [40,40]
                                                │
                                          enforce_psd
                                                │
                                     sklearn SVC.fit(K_train, y_train)
                                                │
                                    alpha* [40], b*, support_vectors [~13,2]
                                                │
X_test [N,2] ──► evaluate(sv_i, s) ──► K_test [N,~13] ──► predict → labels [N]
```

### QVC Training Loop (one SPSA step)
```
theta [4*(l+1)], b ──► perturb by ±c_k*delta
                                │
                    For each (x_i, y_i) in X_train:
                      build_circuit(x_i, theta±delta) → QuantumCircuit
                      run 200 shots → p_hat_y
                      cost_i = sigmoid(EQ8)
                                │
                    R_emp = mean(cost_i)     [scalar]
                                │
                    SPSA update → theta_new, b_new
```

---

## Configuration (`configs/default.yaml`)

```yaml
seed: 42

feature_map:
  n_qubits: 2          # conf=0.97
  reps: 2              # double U_Phi structure

data:
  n_per_label: 20      # conf=0.96
  gap: 0.3             # Delta=0.3, conf=0.98
  V_seed: 42           # ASSUMED: conf=0.80

qvc:
  depths: [0,1,2,3,4]  # conf=0.95
  n_datasets: 3
  spsa:
    n_iter: 250
    shots_cost: 200      # R in EQ8, conf=0.95
    shots_classify: 10000
    a: 0.628             # ASSUMED: Spall default — conf=0.55 ⚠
    c: 0.1               # ASSUMED: Spall default — conf=0.55 ⚠
    A: 100               # ASSUMED
    alpha_spsa: 0.602    # ASSUMED
    gamma_spsa: 0.101    # ASSUMED

qke:
  shots_per_entry: 50000   # conf=0.97
  use_statevector: true    # exact sim mode
  svm:
    C: 1.0                 # ASSUMED: hard margin implied
    psd_epsilon: 1.0e-10

error_mitigation:
  enabled: false           # ASSUMED: ideal sim — conf=0.65 ⚠
  scale_factors: [1.0, 1.5]

backend:
  name: statevector_simulator
  device: cpu
```

---

## Dependencies

```
# requirements.txt
qiskit==0.45.3
qiskit-aer==0.13.3
scikit-learn==1.4.2
numpy==1.26.4
scipy==1.13.0
matplotlib==3.8.4
seaborn==0.13.2
pyyaml==6.0.1
tqdm==4.66.2
```

---

## Entrypoints

| Script | Purpose | Key Args |
|---|---|---|
| `scripts/train_qvc.py` | QVC training + eval | `--depth`, `--n-datasets`, `--plot` |
| `scripts/train_qke.py` | QKE kernel + SVM | `--dataset-id {I,II,III}`, `--use-statevector` |
| `scripts/run_all.py` | Full reproduction | `--config`, `--seed` |
| `scripts/plot_results.py` | Regenerate Figs 3,4 | `--results-dir`, `--format` |

---

## Risk Register

| ID | Severity | Component | Issue | Mitigation |
|---|---|---|---|---|
| **R1** | 🔴 High | SPSA params | a,c,A,α,γ not in paper; wrong values → non-convergence | All in config.yaml; swap to ADAM via flag |
| **R2** | 🟡 Medium | ZNE in sim | Noise scaling ≠ gate stretching | Disabled by default; document scope |
| **R3** | 🟡 Medium | Random V | Paper's V unknown; boundary differs visually | Fixed seed; test 100% accuracy not visual match |
| **R4** | 🟢 Low | Kernel PSD | Shot noise → negative eigenvalues | `enforce_psd()` always called before sklearn |
| **R5** | 🟢 Low | QVC convergence | Stochastic SPSA → curve shapes differ | Report mean±std over 3 datasets; depth=4 → >90% |

---

## Docker

```
Base: python:3.10-slim  (no GPU needed)
CMD:  python scripts/run_all.py --config configs/default.yaml
```

---

## Figures to Reproduce

| Figure | What | Script |
|---|---|---|
| Fig 3a | R_emp convergence over 250 SPSA steps for l=0,4 | `train_qvc.py --plot` |
| Fig 3b | Decision boundary + support vectors + test points | `train_qke.py --plot` |
| Fig 3c | Classification success vs depth (QVC + QKE) | `run_all.py --plot` |
| Fig 4a | Kernel matrix heatmap (ideal vs experimental) | `train_qke.py --plot` |
| Fig 4b | Row cut through kernel matrix | `train_qke.py --plot` |

---

*Written to: `sir-registry/paper_havlicek2018_qsvm/architecture_plan.json`*
*Pipeline state → Stage 4 (Code Generator) ready*
