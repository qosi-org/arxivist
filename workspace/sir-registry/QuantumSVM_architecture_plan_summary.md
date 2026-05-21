# Architecture Plan — QSVM Fraud Detection
**Paper:** Quantum Support Vector Machine for Fraud Detection (Ren & Zhang, IEEE CCPQT 2025)  
**Paper ID:** `paper_qsvm_fraud_detection`  
**Plan Version:** 1 | **Generated:** 2026-05-14

---

## 1. Framework Selection

| Decision | Choice | Reasoning |
|---|---|---|
| Primary framework | **Qiskit + scikit-learn** | Paper explicitly names both |
| Python version | 3.10+ | Modern type hints, match statements |
| CUDA required | **No** | All quantum sim is CPU-based |
| Config library | Plain YAML | Modest hyperparameter count |
| HuggingFace | No | Tabular/quantum domain |

> **Key insight:** This is NOT a PyTorch model. The compute stack is:
> `qiskit-machine-learning` (ZZFeatureMap, QuantumKernel) +
> `qiskit-aer` (statevector_simulator) +
> `scikit-learn` (SVC, KMeans, SelectKBest, metrics).

---

## 2. Repository Structure

```
paper-repos/paper_qsvm_fraud_detection/
├── README.md
├── setup.py
├── .gitignore
├── .env.example                        # KAGGLE_USERNAME, KAGGLE_KEY
│
├── src/qsvm_fraud/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── quantum_smote.py            ← BaseSMOTE, QuantumSMOTE, ClassicalSMOTE
│   │   ├── feature_map.py              ← QSVMFeatureMap (wraps ZZFeatureMap)
│   │   ├── quantum_kernel.py           ← QSVMKernelComputer
│   │   └── qsvm.py                     ← QSVM (primary model class)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py                  ← FraudDataset (load, feature_select, split)
│   │   └── transforms.py               ← FraudPreprocessor (StandardScaler wrapper)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                  ← QSVMTrainer (orchestrates full pipeline)
│   │   └── losses.py                   ← SVMObjective (dual objective, docs only)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py                  ← FraudMetrics (accuracy, F1, recall, AUC, plots)
│   └── utils/
│       ├── __init__.py
│       └── config.py                   ← Config loader, seed utilities
│
├── configs/
│   ├── config.yaml                     ← Primary config (10-qubit, Quantum-SMOTE)
│   ├── config_4qubit.yaml              ← Ablation: 4-qubit variant
│   ├── config_8qubit.yaml              ← Ablation: 8-qubit variant
│   └── config_debug.yaml              ← n_samples=500 for rapid local testing
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── data/
│   ├── README_data.md                  ← How to get creditcard.csv from Kaggle
│   └── download.sh                     ← Kaggle CLI download script
│
├── notebooks/
│   ├── reproduce_paper_qsvm_fraud_detection.ipynb
│   └── explore_paper_qsvm_fraud_detection.ipynb
│
├── scripts/
│   ├── download_data.py
│   ├── run_ablation.py                 ← Reproduces Table I + Table II
│   └── tune_C.py                       ← Grid search over SVM C parameter
│
├── checkpoints/  (.gitkeep)
├── results/      (.gitkeep)
├── comparison/   (.gitkeep)
│
├── train.py
├── evaluate.py
├── inference.py
├── requirements.txt
├── requirements-dev.txt
└── environment.yaml
```

---

## 3. Module Responsibilities

### `models/quantum_smote.py` ⚠ Highest risk module
Implements the Quantum-SMOTE pipeline:
1. **KMeans clustering** of minority (fraud) samples into K=5 clusters
2. **Amplitude encoding** of each sample into `log2(D)` qubits
3. **Swap test** to compute angular distance between sample and cluster centroid
4. **Quantum rotation** (Ry gate parameterized by rotation_angle) to synthesize new samples

Uses abstract base class `BaseSMOTE` so the quantum implementation can be swapped for `ClassicalSMOTE` (imbalanced-learn fallback) via config flag.

### `models/feature_map.py`
Thin wrapper around `qiskit_machine_learning.circuit.library.ZZFeatureMap`. Supports configurable `n_qubits` (4, 8, or 10), `reps`, and `entanglement` pattern. This is the quantum encoder used by QSVM.

### `models/quantum_kernel.py`
Implements kernel matrix computation:
```
K_ij = |<phi(x_i)|phi(x_j)>|^2
      = |<0^n | U_phi(x_j)† U_phi(x_i) | 0^n>|^2
```
Uses Qiskit's `QuantumKernel` or manual circuit composition + statevector evaluation. **Caches computed matrices to `.npy` files** to avoid redundant O(N²) circuit evaluations.

### `models/qsvm.py`
Top-level orchestrator:
- `fit(X_train, y_train)` → builds ZZFeatureMap → computes K_train → fits `SVC(kernel='precomputed')`
- `predict(X_test)` → computes K_test (N_test × N_sv) → applies decision function: `sign(Σ α_i y_i K(x_sv_i, x) + b)`

---

## 4. Tensor Flows

### Training (QSVM + Quantum-SMOTE)
```
X_raw [284807, 30]
  ↓ SelectKBest(k=10)
X_reduced [284807, 10]
  ↓ train_test_split(0.8/0.2, stratified)
X_train [~227845, 10]  |  X_test [~56962, 10]
  ↓ StandardScaler (fit on train only)
X_train_scaled [~227845, 10]  |  X_test_scaled [~56962, 10]
  ↓ QuantumSMOTE (minority class only: ~393 fraud samples)
X_balanced [N_balanced, 10]  y_balanced [N_balanced]
  ↓ ZZFeatureMap(n_qubits=10) + QuantumKernel
K_train [N_balanced, N_balanced]
  ↓ SVC(kernel='precomputed', C=1.0).fit()
→ alphas [N_sv], bias b, support_vectors [N_sv, 10]
```

### Inference
```
X_test_scaled [N_test, 10]
  ↓ ZZFeatureMap → kernel vs support vectors
K_test [N_test, N_sv]
  ↓ f(x) = sign(Σ α_i y_i K_ij + b)
y_pred [N_test]  {0=legit, 1=fraud}
```

---

## 5. Config Schema (config.yaml excerpt)

```yaml
model:
  n_qubits: 10           # Paper: 4/8/10 tested; primary=10 (confidence 0.97)
  reps: 2                # ASSUMED: Qiskit ZZFeatureMap default (confidence 0.60)
  entanglement: "full"   # ASSUMED: Qiskit ZZFeatureMap default (confidence 0.60)
  C: 1.0                 # ASSUMED: sklearn SVC default (confidence 0.65)
  backend: "statevector_simulator"  # ASSUMED: ideal sim (confidence 0.85)
  cache_kernel: true

quantum_smote:
  n_clusters: 5          # ASSUMED: common KMeans default (confidence 0.45)
  rotation_angle: 0.5    # ASSUMED: not specified in paper (confidence 0.45)
  minority_ratio: 0.5    # ASSUMED: target 50/50 balance (confidence 0.45)
  segmentation_factor: 1.0

data:
  csv_path: "data/raw/creditcard.csv"
  n_features: 10         # Explicitly stated in paper (confidence 0.90)
  score_func: "f_classif" # ASSUMED (confidence 0.60)
  test_size: 0.2         # ASSUMED 80/20 (confidence 0.55)
  random_state: 42

evaluation:
  metrics: [accuracy, f1, recall, auc]
  ablation_qubits: [4, 8, 10]
  run_classical_baseline: true
```

---

## 6. Dependencies (key packages)

| Package | Version | Purpose |
|---|---|---|
| `qiskit` | >=1.0,<2.0 | Core quantum circuits |
| `qiskit-machine-learning` | >=0.7.0 | ZZFeatureMap, QuantumKernel |
| `qiskit-aer` | >=0.13 | statevector_simulator backend |
| `scikit-learn` | >=1.3 | SVC, KMeans, SelectKBest, metrics |
| `numpy` | >=1.24 | Array operations |
| `pandas` | >=2.0 | CSV loading |
| `matplotlib` / `seaborn` | >=3.7 / >=0.12 | Plots |
| `imbalanced-learn` | >=0.11 | ClassicalSMOTE fallback |
| `joblib` | >=1.3 | Model + kernel caching |

---

## 7. Docker Spec

- **Base image:** `python:3.10-slim` (no CUDA needed)
- **System deps:** gcc, g++, libgomp1, git, curl
- **No GPU mount required**
- **Volumes:** `/app/data`, `/app/checkpoints`, `/app/results`
- **Env vars:** `KAGGLE_USERNAME`, `KAGGLE_KEY`
- **Default CMD:** `python train.py --config configs/config.yaml`

---

## 8. Risk Assessment Summary

| Severity | Risk | Mitigation |
|---|---|---|
| 🔴 High | Quantum-SMOTE circuit underspecified | Abstract base class + ClassicalSMOTE fallback |
| 🔴 High | Kernel matrix O(N²) is computationally intractable at full scale | Caching + --max-train-samples flag |
| 🔴 High | SVM C parameter unspecified | Default C=1.0; expose tune_C.py script |
| 🟡 Medium | Train/test split unspecified | Default 80/20 stratified; document as assumption |
| 🟡 Medium | Exact KBest features unspecified | Log selected features; test score_func variants |
| 🟡 Medium | ZZFeatureMap reps/entanglement unspecified | Use Qiskit defaults; expose as config |
| 🟢 Low | Kaggle dataset requires account | Kaggle CLI download script + manual instructions |
| 🟢 Low | Qiskit API version changes | Pin exact versions in requirements.txt |
