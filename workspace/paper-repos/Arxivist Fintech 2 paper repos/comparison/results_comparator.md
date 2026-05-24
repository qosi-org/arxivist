# Results Comparator — Gu, Kelly, Xiu (RFS 2020)
## ArXivist Stage 6

---

## Table 1 Targets — Monthly R²_oos (% per month, full panel)

| Model | Paper R²_oos | Tolerance (±%) | Notes |
|-------|-------------|----------------|-------|
| OLS (all 920) | −3.46 | ±1.0 | Expected strongly negative |
| OLS-3 | +0.16 | ±0.10 | Lewellen (2015) benchmark |
| ENet | +0.11 | ±0.10 | |
| PCR | +0.26 | ±0.10 | ~20-40 components selected |
| PLS | +0.27 | ±0.10 | ~3-6 components selected |
| GLM | +0.19 | ±0.10 | Note: fails to beat linear (no interactions) |
| RF | +0.33 | ±0.10 | ~1-5 layers avg |
| GBRT | +0.34 | ±0.10 | ~50-100 chars used |
| NN1 | +0.29 | ±0.15 | |
| NN2 | +0.31 | ±0.15 | |
| **NN3** | **+0.40** | ±0.15 | **Best performer** |
| NN4 | +0.35 | ±0.15 | Performance drops after NN3 |
| NN5 | +0.35 | ±0.15 | |

---

## Table 3 Targets — Diebold-Mariano Statistics

Key statistically significant differences (positive = column better than row):

| Row → Col | NN3 vs ENet | NN3 vs GLM | NN3 vs OLS-3 |
|-----------|-------------|------------|--------------|
| DM statistic | ~2.07 | ~2.17 | ~2.13 |
| Significant? | Yes (5%) | Yes (5%) | Yes (5%) |

**Pattern to verify**: Neural networks significantly outperform all linear methods.
Trees outperform linear methods but difference is marginally significant.
NN vs tree difference is **not** statistically significant.

---

## Table 5 — S&P 500 Monthly R²_oos (%)

| Model | Paper R²_oos |
|-------|-------------|
| OLS-3 | −0.22 |
| GLM | +0.71 |
| RF | +1.37 |
| GBRT | +1.40 |
| NN1 | +1.08 |
| **NN3** | **+1.80** |
| NN5 | +1.63 |

---

## Table 7 — Long-Short Decile Portfolio (Value-Weighted)

| Model | Ann. Sharpe | Mean Ret (% mo) | Notes |
|-------|-------------|-----------------|-------|
| OLS-3 | 0.61 | 0.94 | Benchmark |
| ENet | 0.39 | 0.60 | |
| RF | 0.77 | 1.62 | |
| GBRT | 0.81 | 0.99 | |
| NN1 | 1.17 | 1.81 | |
| NN2 | 1.20 | 1.92 | |
| **NN3** | **1.35** | **2.12** | **Annualized: 27.1% return** |
| NN4 | 1.35 | 2.26 | |
| NN5 | 1.07 | 1.97 | |

Equal-weighted NN3: Sharpe = **2.45**, mean return = **3.27% per month**.

---

## Table 8 — Risk-Adjusted Performance

| Model | FF5+Mom α (% mo) | t-stat | Information Ratio |
|-------|-----------------|--------|-------------------|
| OLS-3 | 0.24 | 1.09 | 0.21 |
| RF | 1.20 | 3.95 | 0.77 |
| GBRT | 0.66 | 3.11 | 0.61 |
| NN1 | 1.20 | 4.68 | 0.92 |
| NN2 | 1.33 | 4.74 | 0.93 |
| **NN3** | **1.52** | **4.92** | **0.96** |
| NN4 | 1.72 | 6.05 | 1.18 |

---

## Qualitative Checks (data-independent)

These structural results should hold regardless of data vintage:

1. **OLS overfit**: R² with all 920 predictors must be strongly negative (overfit)
2. **Regularization helps**: ENet/PCR/PLS must all beat full OLS
3. **Nonlinearity matters**: RF/GBRT/NN must beat ENet/PCR/PLS
4. **NN3 > NN5**: Shallow beats deep for financial data (small N, low SNR)
5. **GLM ≈ linear**: GLM without interactions should not beat PCR/PLS significantly
6. **Dominant signals**: Top predictors must cluster around momentum, liquidity, volatility
7. **Model agreement**: All methods should agree on which characteristics are most important

---

## Comparison Script

```python
import json

with open("results/evaluation_summary.json") as f:
    results = json.load(f)

r2 = results["r2_oos_pct"]
portfolio = results["portfolio"]

targets = {
    "OLS": -3.46, "OLS3": 0.16, "ENet": 0.11, "PCR": 0.26, "PLS": 0.27,
    "GLM": 0.19, "RF": 0.33, "GBRT": 0.34,
    "NN1": 0.29, "NN2": 0.31, "NN3": 0.40, "NN4": 0.35, "NN5": 0.35,
}

print("R²_oos comparison:")
for model, target in targets.items():
    if model in r2:
        actual = r2[model]
        delta = actual - target
        status = "✓" if abs(delta) < 0.15 else "⚠"
        print(f"  {status} {model:<8}: actual={actual:+.3f}%  target={target:+.3f}%  Δ={delta:+.3f}%")

# Check key qualitative ordering
nn3_r2 = r2.get("NN3", 0)
ols_r2 = r2.get("OLS", 0)
assert ols_r2 < 0, "FAIL: OLS should overfit (negative R²)"
assert r2.get("PCR", 0) > r2.get("ENet", 0) - 0.1, "WARN: PCR should be ~competitive with ENet"
assert nn3_r2 > r2.get("ENet", 0), "FAIL: NN3 should beat ENet"
print("\nQualitative ordering checks passed!")
```

---

## Known Gaps vs Paper

| Gap | Severity | Impact on Results |
|-----|----------|-------------------|
| CRSP/WRDS data proprietary | 🔴 High | Cannot reproduce exact numbers without subscription |
| Mini-batch size assumed (0.45) | 🟡 Medium | Affects NN training dynamics |
| Ensemble seed count assumed (0.50) | 🟡 Medium | Affects NN prediction variance |
| Early stopping patience assumed (0.55) | 🟡 Medium | Affects NN generalization |
| Group lasso approximated by standard ENet | 🟡 Medium | GLM results may differ |
| Spline knot placement assumed (0.62) | 🟡 Medium | Minor GLM effect |
| Exact SIMPLS implementation | 🟢 Low | sklearn PLSRegression is standard |
