# Hallucination Report
**Paper**: Quantifying the Risk–Return Tradeoff in Forecasting  
**Paper ID**: arxiv_2605_09712  
**SIR Version**: 1 (confidence: 0.87)  
**Architecture Plan Version**: 1  
**Audited**: 2026-05-12

---

## Overview

This report documents deviations between the paper's described methods and the generated
implementation. Three categories are audited: structural hallucinations (code components
not in the SIR), parametric hallucinations (assumed hyperparameters that may be wrong),
and omission hallucinations (paper components absent or stubbed in the code).

**Summary**: 1 structural, 4 parametric, 3 omission hallucinations identified.

---

## 1. Structural Hallucinations
*(Components in generated code NOT present in the SIR)*

### S1 — LGB+ OOB Winner Selection via `linear_tree=True`
- **Location**: `src/forecast_risk/models/tree_models.py`, `LGBPlusForecaster.fit()`
- **Severity**: Critical
- **Evidence**: The paper (Sec 3) states LGB+ "at each boosting step, a tree-based and a
  linear update compete; the winner is selected via out-of-bag validation." The generated
  code uses LightGBM's `linear_tree=True` flag, which simply adds linear features to tree
  splits. This is architecturally different: the paper describes a competitive boosting
  procedure with OOB model selection, not a single model with linear leaves.
- **Impact**: LGB+ and LGBA+ results will not match the paper. These are key models in
  the unemployment and inflation results.
- **Suggested Fix**: Obtain the actual LGB+ implementation from Goulet Coulombe (2026).
  The working paper "LGB+: A macroeconomic forecasting road test" likely contains or
  references the code.

---

## 2. Parametric Hallucinations
*(Hyperparameters marked `# ASSUMED` that may differ from paper values)*

### P1 — KRR Bandwidth Grid
- **Location**: `configs/default_config.yaml`, `kernel_ridge.sigma_grid`
- **Severity**: Moderate
- **SIR Confidence**: 0.55 (explicitly flagged as ambiguous)
- **Assumed Value**: `[0.01, 0.1, 1.0, 10.0, 100.0]`
- **Evidence**: Paper (Sec 3) states "bandwidth σ and λ cross-validated" with no grid
  specification. The grid range and density significantly affect which kernel width is
  selected, particularly for macro data with high dimensionality after MARX transformation.
- **Impact on Results**: KRR is a top performer for GDP growth (Sortino 2.85 at h=1). Wrong
  bandwidth could explain deviations of ±30–50% in risk-adjusted metrics.
- **Suggested Fix**: Run grid search over `[0.001, 0.01, 0.1, 1, 10, 100, 1000]` and compare
  OOS performance. Contact author for exact specification.

### P2 — LGB Early Stopping Rounds
- **Location**: `configs/default_config.yaml`, `lgb.early_stopping_rounds`
- **Severity**: Moderate
- **SIR Confidence**: 0.55 (ambiguity flagged)
- **Assumed Value**: 50 rounds
- **Evidence**: Paper says "early stopping" but gives no patience specification. Patience
  affects how many sub-optimal boosting rounds are allowed, influencing model complexity.
- **Impact on Results**: LGB results could deviate 5–20% in RMSE ratios. Affects Sharpe,
  Sortino rankings.
- **Suggested Fix**: Try patience ∈ {20, 50, 100}; report sensitivity.

### P3 — Neural Network Learning Rate
- **Location**: `configs/default_config.yaml`, `neural_net.learning_rate`
- **Severity**: Minor
- **SIR Confidence**: 0.65 (assumed as Adam default)
- **Assumed Value**: 0.001
- **Evidence**: Paper specifies "Adam optimizer" but not learning rate. 0.001 is the
  canonical Adam default and unlikely to be far off.
- **Suggested Fix**: Sweep `[0.0001, 0.001, 0.01]` if NN results are significantly off.

### P4 — Max Training Epochs / Early Stopping Patience (NN)
- **Location**: `configs/default_config.yaml`, `neural_net.max_epochs` and `early_stopping_patience`
- **Severity**: Minor
- **SIR Confidence**: 0.65 (assumed)
- **Assumed Values**: max_epochs=200, patience=20
- **Evidence**: Paper states "early stopping" for NN but no epoch count. The expanding
  window re-estimation every 8 quarters means models are retrained frequently; very
  long training may overfit.
- **Suggested Fix**: Check if NN loss curves converge before 200 epochs on macro data;
  adjust if models appear undertrained or overtrained.

---

## 3. Omission Hallucinations
*(Paper components absent or stubbed in the generated code)*

### O1 — HNN: Full Hemisphere Architecture (Critical Omission)
- **Location**: `src/forecast_risk/models/neural.py`, `HemisphereNeuralNetwork`
- **Severity**: Critical
- **What Is Missing**: 
  - Four dedicated hemispheres (long-run expectations, short-run expectations,
    output gap, commodities) with shared common input core
  - Volatility emphasis constraint (breaks mean/variance indeterminacy)
  - Blocked subsampling with B=1000 bootstrap samples
  - Out-of-bag reality check for variance recalibration
  - Softplus activation for variance hemisphere
  - Maximum likelihood training objective (not MSE)
- **Evidence**: Paper (Sec 3, Appendix A) describes all of these. The stub uses a plain
  2-layer FFN trained on MSE.
- **Impact**: HNN results will be completely unreproducible. This model is responsible for
  the paper's most important finding (post-2021 inflation robustness, Sortino 2.0 for
  pre-2020 inflation, Edge 1.83 post-2021).
- **Suggested Fix**: Consult Goulet Coulombe (2025a) "A Neural Phillips Curve and a Deep
  Output Gap" (JBES 2025) and Goulet Coulombe et al. (2026) "From Reactive to Proactive
  Volatility Modeling" (JAE forthcoming) for full HNN code.

### O2 — TabPFN Wrapper (Moderate Omission)
- **Location**: `src/forecast_risk/models/tabpfn_wrapper.py` (referenced but not generated)
- **Severity**: Moderate
- **What Is Missing**: The TabPFN wrapper module was described in the architecture plan but
  the code file was not generated in Stage 4. The notebook references it but no `.py` file
  exists for it.
- **Evidence**: Architecture plan lists `tabpfn_wrapper.py`; no such file in repo.
- **Suggested Fix**: Create the wrapper using the `tabpfn` Python package:
  ```python
  from tabpfn import TabPFNClassifier  # or regression variant
  ```
  Note: TabPFN v0.1.9 may require adaptation for regression tasks.

### O3 — SPF Data Loader (`data/spf_loader.py`)
- **Location**: Architecture plan lists `src/forecast_risk/data/spf_loader.py`
- **Severity**: Minor
- **What Is Missing**: The SPF loader module was planned but not generated. The main pipeline
  references SPF as a competitor but has no automated download/parsing logic.
- **Impact**: SPF must be loaded manually; pipeline will emit a warning and exclude it.
- **Suggested Fix**: Implement SPF CSV parser for Philadelphia Fed data format. Key columns:
  RGDP, CPI, UNEMP, HOUSING; horizons 1, 2, 4; quarterly frequency.

---

## 4. Confirmed Correct Components

For completeness, the following components are correctly implemented with high confidence:

| Component | SIR Confidence | Implementation Status |
|-----------|---------------|----------------------|
| Forecast Sharpe Ratio formula | 0.99 | ✅ Exact — verified vs paper Eq |
| Forecast Sortino Ratio formula | 0.99 | ✅ Exact — verified vs paper Eq |
| Forecast Omega Ratio formula | 0.99 | ✅ Exact — verified vs paper Eq |
| Maximum Drawdown algorithm | 0.98 | ✅ Exact — R_0=0 convention correct |
| Edge Ratio formula incl. (M-1) scaling | 0.97 | ✅ Exact — null expectation =1 |
| Meta-analysis percentage return | 0.97 | ✅ Exact |
| Diebold-Mariano / Sharpe link | 0.99 | ✅ Correct per Sec 2.3 |
| AR(4) specification | 0.99 | ✅ 4 lags, OLS |
| FAAR (4 PCs, AR lags) | 0.95 | ✅ Stock-Watson spec correct |
| Random Forest (n=500, sub=0.75, min_leaf=5) | 0.98 | ✅ Paper-specified params |
| Expanding window (refit every 8 quarters) | 0.97 | ✅ Correct |
| HAC NOT applied to Sharpe/Sortino/Omega | 0.97 | ✅ Deliberate design choice preserved |
| MARX transformation (lags 4, MA 2/4/8) | 0.95 | ✅ Correct per paper Sec 3 |
| Standardization per training fold | 0.90 | ✅ Implemented in run_evaluation.py |
