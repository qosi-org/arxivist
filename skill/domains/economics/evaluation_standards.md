# Domain: Economics — Evaluation Standards (Stage 6 Enrichment)

Load alongside `agents/06_results_comparator.md` when domain is **Economics**.

---

## What "reproduction" means in economics

Economics reproduction is coefficient-level, not metric-level. The comparison is:
- Coefficient estimate (β) — should match to 2–3 decimal places
- Standard error — should match to 2 decimal places
- t-statistic and p-value — should produce the same significance conclusion
- R² — should match to 3 decimal places

## Deviation thresholds

| Quantity | Excellent | Good | Moderate | Notes |
|----------|-----------|------|----------|-------|
| Coefficient | ≤ 0.5% | ≤ 2% | ≤ 5% | Relative to magnitude |
| Standard error | ≤ 1% | ≤ 3% | ≤ 8% | |
| R² | ≤ 0.001 | ≤ 0.003 | ≤ 0.01 | Absolute |
| F-statistic (first stage) | ≤ 5% | ≤ 15% | ≤ 30% | Relative |

**If significance conclusion differs (p < 0.05 vs p > 0.05), always classify as Significant
regardless of coefficient deviation.** The inference is wrong even if the point estimate is close.

---

## Root causes specific to economics

1. **Different data vintage** — GDP and macro data are revised; using revised vs real-time
   data changes regression coefficients, sometimes substantially.
2. **Sample period mismatch** — one missing year or quarter changes all fixed effects estimates.
3. **Standard error formula mismatch** — HC2 vs HC3 vs clustered produces different SEs.
4. **Missing control variables** — omitted variables change coefficient estimates through OVB.
5. **Proprietary data** — many economics papers use data that cannot be obtained freely. If
   the dataset is proprietary, reproducibility requires the exact same data access.
