# Domain: Finance — Evaluation Standards (Stage 6 Enrichment)

Load alongside `agents/06_results_comparator.md` when domain is **Finance**.

---

## Deviation thresholds for finance metrics

Finance results have high variance due to market regime changes. These thresholds apply to
**out-of-sample** results only. In-sample comparison is not meaningful.

| Metric | Excellent | Good | Moderate | Significant | Notes |
|--------|-----------|------|----------|-------------|-------|
| Sharpe ratio | ≤ 0.05 | ≤ 0.15 | ≤ 0.30 | ≤ 0.50 | Absolute difference |
| Max drawdown | ≤ 1% | ≤ 3% | ≤ 6% | ≤ 12% | Absolute percentage points |
| Annual return | ≤ 0.5% | ≤ 1.5% | ≤ 3% | ≤ 6% | Absolute percentage points |
| Sortino ratio | ≤ 0.08 | ≤ 0.20 | ≤ 0.40 | ≤ 0.70 | Absolute |
| Turnover | ≤ 5% | ≤ 15% | ≤ 30% | > 30% | Relative — affects cost sensitivity |

**A Sharpe ratio within 0.15 of the paper's reported value is Good reproduction** — finance
results vary across market regimes and the same strategy can produce different Sharpe ratios
on the same data depending on exact cost assumptions.

---

## Root causes specific to finance

1. **Survivorship bias in universe construction** — if the paper uses a stock universe without
   delisting data, results are optimistic. Check whether the paper acknowledges this.

2. **Price adjustment methodology** — split-adjusted vs dividend-adjusted vs total return.
   Even the same data provider produces different series depending on adjustment method.

3. **Transaction cost assumption mismatch** — 1 basis point difference in commission can
   change Sharpe ratio by 0.2–0.5 on high-turnover strategies. Always flag if costs differ.

4. **Rebalancing frequency implementation** — "monthly rebalancing" can mean first trading
   day of month, last trading day, or calendar month-end. Verify exact implementation.

5. **Different data vendor** — prices from Bloomberg vs Compustat vs Yahoo Finance differ
   due to different corporate action handling. Flag if data source is not the same.

6. **Look-ahead bias** — the most dangerous and most common source of Critical deviation in
   finance. If reproduction Sharpe is dramatically higher than paper, suspect look-ahead bias
   in the generated code first.

---

## Finance-specific hallucination checks

**Always check for look-ahead bias in generated code:**
- Rolling features computed on full dataset before train/test split
- `shift(-1)` used to compute forward returns in the training signal
- StandardScaler or similar fitted on full dataset including test period

**Parametric hallucinations specific to finance:**
- Annualisation factor wrong (252 vs 260 vs 365)
- Risk-free rate not subtracted from returns before Sharpe computation
- Transaction costs applied to gross positions rather than changes in position
