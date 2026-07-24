# Architecture Plan Summary — arxiv_2607_20168

**Paper:** Quantum Kernels and the Cross-Section of Stock Returns: Anatomy of a Vanishing Advantage
**Framework:** PyTorch (MLP/NN3 baselines) + PennyLane (quantum circuit sim) + scikit-learn/XGBoost (classical baselines), YAML config.

## Pipeline
```
Raw exchange snapshots + financials
        │
CharacteristicBuilder ─▶ 27/31 standardized characteristics (Table 1)
        │
   ┌────┴────┐
PointInTime  StaticScreen   (Sec 3.2 -- the paper's central methodological contrast)
Universe     Universe
        │
FactorRotationSelector ─▶ top-8 (shared by all top-8 models) + active set (classical full-set models)
        │
BandwidthScaler (x~ = λx, grid-searched per window)
        │
   ┌────┴─────────────┬─────────────┐
QuantumFeatureMap   QuantumFeatureMap   (same x~)
(fidelity kernel)   (projected kernel)   ClassicalRBFKernel (control)
        │                  │                    │
        └────────┬─────────┴────────────────────┘
        KernelRidgeRegression.fit (closed-form, α grid)  ── kernel-swap control: only the kernel differs
                 │
        ┌────────┴─────────┐
   N=1536 subsample    NystromKRR (full ~37,800-obs budget, 2×2 design)
                 │
     WalkForwardEngine (170 main / 60 diagnostic windows)
                 │
     PerformanceMetrics (IC/ICIR/t-stat/hit-rate/Sharpe/drawdown, paired t/Wilcoxon, Holm correction)
                 │
     GeometricDifference (Eq. 2 diagnostic, Sec 4.5/8)
```

## Repo layout
```
src/qkernel_finance/
├── data/          characteristics.py, universes.py, interactions.py, synthetic.py
├── features/       top8_selector.py, bandwidth.py
├── quantum/         feature_map.py (Eq.1 circuit), kernels.py (fidelity + projected)
├── classical/        rbf_kernel.py, baselines.py (ridge/xgb/mlp/nn3/poly2ridge)
├── models/           krr.py (closed-form solve + Nystrom)
├── evaluation/        walkforward.py, metrics.py, geometry.py (Eq.2)
└── utils/             config.py
run_study.py · compare_models.py · run_bandwidth_geometry_diagnostic.py
configs/config.yaml · requirements.txt · Dockerfile
```

## Key config decisions (with SIR confidence)
| Setting | Value | SIR confidence | Note |
|---|---|---|---|
| Quantum circuit: n qubits, R reps | 8, 2 | 0.9 | explicit |
| **H_q gate semantics** | Hadamard-conjugated RZ | **0.6** | **paper never defines H_q — highest-risk assumption in this repo** |
| Fidelity kernel | \|⟨ψ\|ψ'⟩\|² | 0.92 | explicit |
| Projected kernel γ tuning | per-window grid search (assumed) | 0.5 | not stated in paper |
| Bandwidth grid (production) | {0.05, 0.1, 0.2, 0.4} | 0.85 | explicit |
| KRR α grid | {1e-3, 1e-2, 1e-1, 1} | 0.92 | explicit |
| Subsample size N | 1536, stratified by date | 0.55 (allocation scheme) | size explicit, scheme assumed |
| Nyström formula | standard Williams & Seeger (2001) | 0.55 | cited, not restated |
| MLP/NN3 optimizer | Adam, lr=1e-3 (assumed) | 0.5 | not stated |

## Risks (full detail in `architecture_plan.json.risk_assessment`)
- **High** — `H_q` gate undefined in paper; wrong interpretation invalidates the entire quantum branch
- **High** — underlying China A-share dataset is proprietary, not publicly released (only "available from author on request")
- **Medium** — projected-kernel γ tuning, Nyström formula, and stratified-subsampling scheme all under-specified
- **Low** — MLP/NN3 training hyperparameters unstated; simulation compute budget nontrivial but tractable

## Entrypoints
- `run_study.py --study main --config configs/config.yaml` (170-window point-in-time evaluation, Table 3/4)
- `run_study.py --study diagnostic` (60-window static-screen evaluation, Table 5)
- `compare_models.py --results-dir results/` (paired significance + Holm correction, Sec 5.2)
- `run_bandwidth_geometry_diagnostic.py` (Section 8 / Figure 2 reproduction)
