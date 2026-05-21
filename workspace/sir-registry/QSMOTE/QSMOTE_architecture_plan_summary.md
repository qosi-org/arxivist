# Architecture Plan — Quantum-SMOTE
**Paper:** arXiv:2402.17398 | **Plan v1** | Generated: 2026-05-21

---

## 1. Framework Selection

| Concern | Choice | Reason |
|---|---|---|
| Quantum circuits | **Qiskit 0.45.3 + Qiskit-Aer 0.13.3** | Explicitly named in paper; pinned to 0.x for API compatibility |
| Classical ML | **scikit-learn 1.4.1** | K-Means, Random Forest, Logistic Regression |
| Data | **pandas + numpy** | Tabular dataset handling |
| Config | **YAML (PyYAML)** | Lightweight; no DL framework needed |
| Python | **3.10+** | |
| GPU | **Not required** | CPU-only; statevector simulation |

---

## 2. Project Structure

```
paper-repos/arxiv_2402_17398/
├── run_experiment.py          ← Main entrypoint (full pipeline)
├── generate_synthetic.py      ← SMOTE-only generation
├── evaluate.py                ← Classifier evaluation
├── visualize.py               ← Paper figure reproduction
├── configs/
│   └── config.yaml            ← All hyperparameters
├── src/
│   └── quantum_smote/
│       ├── data/
│       │   ├── preprocessor.py     ← TelcoChurnPreprocessor
│       │   └── dataset.py          ← TelcoChurnDataset
│       ├── clustering/
│       │   └── kmeans_clusterer.py ← KMeansClusterer
│       ├── quantum/
│       │   ├── state_preparation.py  ← StatePreparation (Algorithm 3)
│       │   ├── swap_test.py          ← CompactSwapTest (Algorithm 4)
│       │   ├── angle_calculator.py   ← AngleCalculator (Algorithm 2)
│       │   └── rotator.py            ← QuantumRotator (Algorithm 6)
│       ├── smote/
│       │   └── quantum_smote.py      ← QuantumSMOTE orchestrator (Algorithm 7)
│       └── evaluation/
│           ├── metrics.py            ← All paper metrics
│           └── classifier.py         ← RF + LR wrappers
├── data/
│   └── README.md              ← Download instructions
├── notebooks/
│   └── quantum_smote_demo.ipynb
├── docker/
│   └── Dockerfile
├── results/
│   └── figures/
└── README.md
```

---

## 3. Data Flow (Full Pipeline)

```
raw CSV [N_raw, 21]
    ↓  TelcoChurnPreprocessor
X [N, 32] float64   y [N] int
    ↓  KMeansClusterer (K=3)
labels [N]  centroids [3, 32]
    ↓  FOR each cluster:
       minority_X [N_min_c, 32]   centroid_dp [32]
         ↓  FOR each minority sample [32]:
            StatePreparation.prepare()
              phi [2]   psi [64]   (amplitude-encoded)
            CompactSwapTest.run(psi, phi)
              swap_test_probability scalar
              angular_distance scalar (radians)
            AngleCalculator.compute(ad, split_factor, loop)
              angle scalar (radians)
            QuantumRotator.rotate(minority_sample, angle)
              synthetic_point [32]
    ↓  Accumulate → syn_dataframe [N_syn, 32]
X_aug [N+N_syn, 32]   y_aug [N+N_syn]
    ↓  train_test_split (80/20 stratified)
ClassifierFactory.train_evaluate(RF / LR)
    ↓
metrics {accuracy, f1, pr_auc, roc_auc, confusion_matrix}
```

---

## 4. Key Config Parameters

```yaml
quantum_smote:
  target_pct: 50           # Tested: 30, 40, 50
  split_factor: 5          # ASSUMED (confidence 0.55) — paper tests 2/5/10/100
  rotation_axis: X         # Paper uses X for main experiment
  angle_increment: 0.0174533  # 1 degree in radians (per loop iteration)
  use_statevector: true    # ASSUMED (confidence 0.65) — exact probs vs shot sampling
  statevector_extraction_strategy: first_F  # ASSUMED (confidence 0.55)
```

---

## 5. Risk Summary

| ID | Severity | Issue |
|---|---|---|
| R1 | 🔴 High | Statevector → feature vector extraction ambiguous |
| R2 | 🔴 High | split_factor for main experiment not specified |
| R3 | 🟡 Medium | Shots vs statevector mode in swap test |
| R4 | 🟡 Medium | Feature selection threshold not given |
| R5 | 🟡 Medium | Train/test split ratio not stated |
| R6 | 🟢 Low | Simulation loop performance |
| R7 | 🟢 Low | Kaggle data download requires credentials |
| R8 | 🟢 Low | Qiskit 1.0 API breaking changes |

---

## 6. Entrypoints

| Script | Purpose |
|---|---|
| `run_experiment.py` | Full pipeline end-to-end |
| `generate_synthetic.py` | SMOTE generation only |
| `evaluate.py` | Classifier evaluation on saved data |
| `visualize.py` | Reproduce paper figures |

---

## 7. Test Plan Summary

**Unit tests (8):** StatePrep encoding correctness, SwapTest boundary cases (identical/orthogonal vectors), AngleCalculator three branches, Rotator shape/identity, NormalizeArray, Preprocessor shape.

**Integration tests (3):** Baseline pipeline accuracy, 30% SMOTE pipeline count check, Metric comparison to Table 1 (±0.05 tolerance).
