# Benchmark Comparison Report
**Paper**: Quantifying the Risk–Return Tradeoff in Forecasting  
**Paper ID**: arxiv_2605_09712  
**ArXiv**: https://arxiv.org/abs/2605.09712  
**Comparison Date**: 2026-05-12  
**Reproducibility Score**: N/A — Pre-run scaffold (no user results submitted yet)  
**SIR Version**: 1  
**SIR Overall Confidence**: 0.87

---

## Status

No user experimental results have been submitted yet. This report provides the **pre-run
scaffold** — the paper's reported results, the full set of comparison targets, and the
root causes to investigate when deviations arise.

To trigger a live comparison, re-run the ArXivist comparator and supply your output results.

---

## Paper Reported Results — Comparison Targets

The following are the ground-truth metrics extracted from the SIR
(`evaluation_protocol.reported_results`). All comparisons should be against these values.

### Application 1 — GDP Growth, Pre-COVID (2007Q2–2019Q4)

| Metric | Model | Horizon | Paper Value | Target Tolerance |
|--------|-------|---------|-------------|-----------------|
| Sortino (SE) | SPF | h=1 | 1.80 | ± 0.15 (±8%) |
| Sortino (SE) | KRR | h=1 | 2.85 | ± 0.25 (±9%) |
| Sortino (SE) | NN  | h=1 | 1.09 | ± 0.15 (±14%) |
| Edge Ratio   | FAAR | h=1 | 0.85 | ± 0.15 |
| Edge Ratio   | SPF | h=1 | 0.08 | ± 0.05 |
| Edge Ratio   | SPF | h=4 | 1.10 | ± 0.20 |
| Return (%)   | KRR | h=1 | 0.29 | ± 0.05 |

### Application 1 — Unemployment Rate, Pre-COVID

| Metric | Model | Horizon | Paper Value | Target Tolerance |
|--------|-------|---------|-------------|-----------------|
| Return (%) | LGB+ | h=1 | 0.55 | ± 0.05 |
| Sortino    | LGB+ | h=1 | 7.7  | ± 1.0  |
| Edge Ratio | TPFN | h=4 | 4.45 | ± 0.5  |
| Sortino    | RF   | h=4 | 4.8  | ± 0.5  |
| Sortino    | NN   | h=4 | 0.5  | ± 0.1  |

### Application 1 — Inflation, Pre-2020, h=1

| Metric | Model | Paper Value | Target Tolerance |
|--------|-------|-------------|-----------------|
| Sortino (SE) | HNN  | 2.0  | ± 0.3 |
| Sortino (SE) | SPF  | 1.2  | ± 0.2 |
| Edge Ratio   | TPFN | 0.86 | ± 0.15 |
| Edge Ratio   | SPF  | 0.52 | ± 0.10 |

### Application 1 — Inflation, Post-2021, h=1

| Metric | Model | Paper Value | Target Tolerance |
|--------|-------|-------------|-----------------|
| Edge Ratio | HNN | 1.83 | ± 0.30 |
| Return (%) | HNN | >0   | Must be positive |
| Return (%) | All others | <0 | Must be negative |

### Application 2a — Meta-Analysis RMSE (GCFK Table 1)

| Metric | Model | Paper Value | Target Tolerance |
|--------|-------|-------------|-----------------|
| Sharpe  | HNN  | 0.85 | ± 0.10 |
| Sharpe  | BART | 0.97 | ± 0.10 |
| Sortino | HNN  | 1.62 | ± 0.20 |
| Sortino | BART | 1.62 | ± 0.20 |
| Edge    | HNN  | 1.97 | ± 0.30 |
| Edge    | BART | 0.81 | ± 0.15 |

### Application 2a — Meta-Analysis Log Score (GCFK Table 1)

| Metric | Model | Paper Value | Target Tolerance |
|--------|-------|-------------|-----------------|
| Sharpe  | HNN    |  0.48 | ± 0.10 |
| Sharpe  | BART   | -0.41 | ± 0.10 |
| Sortino | HNN    |  0.92 | ± 0.15 |
| Sortino | BART   | -0.30 | ± 0.10 |
| Edge    | HNN    |  2.59 | ± 0.40 |

### Application 2b — M4 Monthly (Table 2)

| Metric | Model | Paper Value | Target Tolerance |
|--------|-------|-------------|-----------------|
| MASE Sharpe  | Model 118 (ES-RNN) | 0.16 | ± 0.03 |
| OWA  Sharpe  | Model 237           | 0.42 | ± 0.05 |
| OWA  Sharpe  | Model 118           | 0.37 | ± 0.05 |

---

## Pre-Run Root Cause Analysis

For each result category, the most likely sources of deviation are pre-identified:

### GDP / Unemployment Results

**High-risk sources of deviation:**

1. **LGB+ / LGBA+ are STUBs** (Critical): The generated `LGBPlusForecaster` approximates
   LGB+ using `linear_tree=True` in LightGBM. The actual algorithm (Goulet Coulombe 2026)
   alternates tree and linear updates with OOB selection. Results for LGB+ will almost
   certainly deviate from the paper.  
   *Fix*: Obtain the actual LGB+ implementation from the author.

2. **HNN is a STUB** (Critical): The generated `HemisphereNeuralNetwork` delegates to a
   plain feed-forward NN. The actual HNN has 4 dedicated hemispheres, volatility emphasis
   constraints, and B=1000 bootstrap recalibration.  
   *Fix*: Obtain the HNN implementation from Goulet Coulombe (2025a, 2026).

3. **SPF excluded from estimation models** (Significant): SPF results depend on correctly
   downloading and aligning the Philadelphia Fed SPF medians. Misaligned vintages or
   incorrect horizon mapping can cause large deviations.  
   *Fix*: Verify SPF series dates match FRED-QD evaluation periods exactly.

4. **KRR bandwidth grid** (Moderate): The grid `[0.01, 0.1, 1.0, 10.0, 100.0]` is assumed
   (SIR confidence 0.55). If the paper used a different grid, KRR results may differ.  
   *Fix*: Run sensitivity analysis over bandwidth grids.

5. **Evaluation window boundary** (Moderate): Post-COVID end date is ambiguous (2024Q2 vs
   2025Q1 across tables). The main text uses 2024Q2; some appendix tables use 2025Q1.  
   *Fix*: Match exactly to the table being reproduced.

### Inflation Results (HNN dominance post-2021)

The paper's most striking result — HNN uniquely retaining positive returns post-2021 — 
depends critically on the actual HNN implementation with its proactive volatility hemisphere.
The stub implementation will **not reproduce this result**.

### M4 Competition Results

The M4 results use publicly available competition submissions. Model IDs (118, 237, etc.) 
refer to specific competition entries, not models generated by this codebase. To reproduce
Table 2, use the original M4 competition data and submissions from:
https://github.com/Mcompetitions/M4-methods

---

## Recommended Actions (Priority Order)

1. **[Critical]** Obtain actual LGB+ / LGBA+ code from the author (Goulet Coulombe 2026 WP)
2. **[Critical]** Obtain actual HNN code from author repos (Goulet Coulombe 2025a, 2026)
3. **[High]** Download and verify SPF data alignment with FRED-QD quarters
4. **[High]** For M4 results: use original competition submissions, not this codebase
5. **[Medium]** Run sensitivity analysis on KRR bandwidth grid
6. **[Medium]** Confirm post-COVID evaluation end date (2024Q2 vs 2025Q1) per table
7. **[Low]** Fix random seed and deterministic mode before final comparison runs

---

## Comparison Table Template (Fill in your results)

When you have results, fill in the "Your Value" column:

| Setting | Metric | Model | Paper | Your Value | Deviation | Severity |
|---------|--------|-------|-------|------------|-----------|----------|
| GDP h=1 Pre-COVID | Sortino | SPF | 1.80 | ___ | ___ | ___ |
| GDP h=1 Pre-COVID | Sortino | KRR | 2.85 | ___ | ___ | ___ |
| GDP h=1 Pre-COVID | Edge | FAAR | 0.85 | ___ | ___ | ___ |
| GDP h=4 Pre-COVID | Edge | SPF | 1.10 | ___ | ___ | ___ |
| UNEMP h=1 Pre-COVID | Sortino | LGB+ | 7.7 | ___ | ___ | ___ |
| UNEMP h=4 Pre-COVID | Edge | TPFN | 4.45 | ___ | ___ | ___ |
| CPI h=1 Pre-2020 | Sortino | HNN | 2.0 | ___ | ___ | ___ |
| CPI h=1 Post-2021 | Edge | HNN | 1.83 | ___ | ___ | ___ |
| Meta RMSE | Sharpe | HNN | 0.85 | ___ | ___ | ___ |
| Meta RMSE | Sharpe | BART | 0.97 | ___ | ___ | ___ |
| Meta LogScore | Sharpe | HNN | 0.48 | ___ | ___ | ___ |
