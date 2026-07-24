# Data — arXiv:2607.20168

## The real dataset is not available

Unlike some ArXivist reproductions, this paper's underlying data is **not
publicly released**. The paper's own "Data and code availability" statement
says only: *"Code and intermediate data for full replication are available
from the author."* There is no public dataset URL, DOI, or vendor named.

The data required (Sec 3.1-3.2) is:
- Daily China A-share exchange snapshots: price, valuation ratios, market
  capitalization, turnover (Jan 2010 - Mar 2026).
- Quarterly financial statements for the same universe, with **announcement
  dates** (not just fiscal period end dates) -- required for the paper's
  point-in-time alignment rules (Sec 3.1).
- Sufficient trading history to compute 252-day trailing price coverage back
  to 2011 (for the point-in-time universe) and full-sample 2020-2025 coverage
  (for the static-screen universe).

This is the **single largest reproducibility barrier** in this paper (see
`architecture_plan.json → risk_assessment`, "High" severity). Likely sources
if you want to attempt a real reproduction: a commercial China A-share data
vendor (e.g. Wind, Tushare, JoinQuant), or a university/institutional data
subscription with equivalent coverage.

## What this repo does instead

`src/qkernel_finance/data/synthetic.py` generates a schema-compatible
synthetic panel (random characteristics, weak injected signal, synthetic
prices/market caps) so that **every other module** — universe construction,
feature selection, the quantum feature map and both quantum kernels, the
classical RBF control, closed-form KRR, the Nyström extension, walk-forward
evaluation, and all statistical tests — can be exercised and smoke-tested
end-to-end without the real data.

```bash
python run_study.py --config configs/config.yaml --study main --debug
```

**This does not reproduce the paper's actual numbers.** It only confirms the
pipeline is wired correctly. Do not compare its output against Table 3/4/5 —
see `comparison/benchmark_comparison.md` for why that comparison would be
misleading if attempted here.

## If you have real data: expected schema

`CharacteristicBuilder.build_market_characteristics()` and
`.build_fundamental_characteristics()` are stubbed with `NotImplementedError`
and clear docstrings describing exactly what each needs (Sec 3.1's merge
rules: TTM aggregation only if 4 quarters span ≤380 days, YoY growth requires
a 330–400 day span, forward-fill ≤250 trading days). Implement those two
methods against your data source, and the rest of the pipeline (standardize →
universe → top-8 selection → kernels → KRR → walk-forward → metrics) will run
unmodified.

Expected panel shape after `CharacteristicBuilder.standardize()`:

| column | description |
|---|---|
| `date` | trading date |
| `ticker` | stock identifier |
| `book_to_price`, `sales_to_price`, ... | one column per characteristic (27 main-study / 31 diagnostic-study, Table 1) |
| `price` | daily close (or similar), for universe coverage screens |
| `float_mktcap` | float-adjusted market capitalization |
| `fwd_return_20d` | 20-trading-day forward return (the prediction target) |

## Interaction catalog data requirements

`InteractionCatalog.build()` (Table 2) additionally needs `analyst_revisions`
and `eps_vol` columns, which per Sec 3.1 are only available in the
31-characteristic diagnostic-study set (available from 2020), not the
27-characteristic main-study set.
