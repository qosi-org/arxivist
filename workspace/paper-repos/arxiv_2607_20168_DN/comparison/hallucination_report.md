# Hallucination Report — arxiv_2607_20168

**Comparison date**: 2026-07-24
**SIR version**: 1 | **Architecture plan version**: 1

---

## Structural Hallucinations

**None found.**

Every module in `src/qkernel_finance/` traces to a named SIR `architecture.modules[]` entry:

| Code component | SIR module |
|---|---|
| `CharacteristicBuilder` | `CharacteristicConstructionAndStandardization` |
| `PointInTimeUniverse` / `StaticScreenUniverse` | `UniverseConstruction_PointInTime` / `_StaticScreen` |
| `FactorRotationSelector` | `FactorRotationAndTop8Selection` |
| `BandwidthScaler` | `BandwidthScaling` |
| `QuantumFeatureMap` | `QuantumFeatureMapCircuit` |
| `FidelityKernel` | `FidelityKernelComputation` |
| `ProjectedQuantumKernel` | `ProjectedKernelComputation` |
| `ClassicalRBFKernel` | `ClassicalRBFKernelControl` |
| `KernelRidgeRegression` | `KernelRidgeRegressionSolver` |
| `NystromKRR` | `NystromExtension` |
| `RidgeBaseline`, `XGBoostBaseline`, `MLPBaseline`, `NN3Ensemble`, `Poly2RidgeBaseline` | `ClassicalBenchmarkSuite` |

`WalkForwardEngine`, `PerformanceMetrics`, `GeometricDifference`, `Config`, and
`InteractionCatalog` are evaluation/utility infrastructure directly cited in the paper's
methodology (Sec 4.1, 4.5, 5.2, 3.3) rather than invented additions.

---

## Parametric Hallucinations

Five hyperparameters were set to assumed values. None are Critical — no real training run
has occurred, so these are pre-identified risk factors from Stage 3, not confirmed causes
of any observed deviation.

| Hyperparameter | Assumed value | Severity | Evidence |
|---|---|---|---|
| `quantum.hq_gate_interpretation` | Hadamard-conjugated RZ | **Significant** | SIR `ambiguities[0]`, confidence 0.6 — Eq. (1)'s `H_q` term is undefined in the paper |
| `quantum.gamma_projected_kernel_grid` | reuses the λ grid | Minor | SIR `ambiguities[1]`, confidence 0.5 |
| `krr.nystrom_formula` | standard Williams & Seeger (2001) | Minor | SIR `ambiguities[2]`, confidence 0.55 |
| `krr.subsample_stratification` | proportional by date | Minor | SIR `ambiguities[3]`, confidence 0.55 |
| `classical_baselines.nn3_optimizer/lr` | Adam, 1e-3 | Minor | SIR `implementation_assumptions[4]`, confidence 0.5 |

`hq_gate_interpretation` is the one **Significant** item — it's the single assumption most
likely to silently produce a *different* (and possibly wrong) quantum feature map even
with perfect real data, since it changes the circuit itself rather than a downstream
hyperparameter.

---

## Omission Hallucinations

| Missing component | Severity | Why |
|---|---|---|
| Real data ingestion (`CharacteristicBuilder.build_market_characteristics` / `.build_fundamental_characteristics` are `NotImplementedError` stubs) | Significant | **Deliberate, documented** (architecture_plan.json risk_assessment: "High" severity — the paper's China A-share data is proprietary, "available from the author on request," not publicly released). Not an oversight. |
| Full-scale walk-forward runs (170 + 60 windows) | Significant | Only a 3-window synthetic smoke test has been executed in this session. `run_study.py --study main` (without `--debug`) supports the full run once real data is available; not run here because (a) no real data, (b) the compute budget (170+60 windows × 8-qubit statevector sim × N up to 1536) is nontrivial. |

No other omissions found. All fourteen `architecture_plan.json → module_hierarchy[]`
entries have a corresponding, exercised code path — confirmed via the executed
`notebooks/reproduce_arxiv_2607_20168.ipynb` and the `run_study.py` /
`compare_models.py` / `run_bandwidth_geometry_diagnostic.py` smoke tests.

---

## Summary

| Type | Count | Critical | Significant | Minor |
|---|---|---|---|---|
| Structural | 0 | 0 | 0 | 0 |
| Parametric | 5 | 0 | 1 | 4 |
| Omission | 2 | 0 | 2 | 0 |

The implementation faithfully maps the SIR's methodology. Both Significant omissions are
deliberate, documented scope boundaries (no public data, no real-scale run performed) —
not hidden gaps. The one Significant parametric item (`H_q` gate) is the paper's own fault
for being underspecified, not an implementation shortcut; it's surfaced prominently in the
README, config comments, and code docstring rather than buried.
