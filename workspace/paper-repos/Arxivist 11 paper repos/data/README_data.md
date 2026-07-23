# Data Requirements

This paper's exact 420-stock large-cap U.S. universe and the exact construction of the
182 sorted-portfolio factors are **not enumerated** in the paper text (see
`sir-registry/arxiv_2505_01575/sir.json` → `ambiguities[2]`, confidence 0.3). To run this
repo against the real dataset, you need to supply two CSV files yourself:

## 1. `data/raw/factors_182.csv`
- Rows: monthly dates, `1957-01` through `2022-12`.
- Columns: up to 182 firm-characteristic sorted-portfolio factors.
- Source described in the paper: **Andrew Y. Chen & Tom Zimmermann (2020), "Open Source
  Cross-Sectional Asset Pricing"**, Centre for Financial Research (CFR) working paper.
  Public factor library: https://www.openassetpricing.com/
- Any factor with more than 40% missing values across the sample is automatically dropped
  by `FactorPreprocessor` (Section 3), so you do not need to pre-filter.

## 2. `data/raw/excess_returns_420stocks.csv`
- Rows: same monthly date index as the factors file.
- Columns: excess returns (return minus risk-free rate) for 420 large-capitalization
  U.S. stocks.
- The paper does not list the exact ticker set. A reasonable reconstruction is the
  top-420-by-market-cap CRSP universe at each rebalance date, restricted to stocks with
  return history spanning the full 1957-2022 sample. CRSP access requires a WRDS
  subscription (not free/public).

## Quick local testing without real data

Every entrypoint (`train.py`, `evaluate.py`, `backtest.py`) falls back to a small
synthetic dataset when the CSVs above are absent, or when `--debug` is passed to
`train.py`. This lets you verify the full pipeline runs end-to-end before sourcing the
real data.

```bash
python train.py --config configs/config.yaml --debug
```

## `data/download.py`

Run `python data/download.py` for a guided check: it verifies whether the two CSVs
above are present, and if not, prints these instructions again along with the public
factor-library URL.
