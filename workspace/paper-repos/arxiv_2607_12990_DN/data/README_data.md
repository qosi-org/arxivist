# Data — arXiv:2607.12990 reproduction

The paper's market inputs come from **LSEG Workspace** (Table 8), which is a
proprietary commercial data source and cannot be redistributed with this
repository:

- Historical equity closes (13-Mar-2021 to 13-Mar-2026) for EURO STOXX 50 and SMI
- ATM implied-volatility surfaces (35 expiries for EURO STOXX 50, 18 for SMI)
- EUR OIS discount curve (pillars 7D–60Y)
- Iberdrola CDS par-spread curve (tenors 6M–30Y)
- Dividend yields

## Option 1: Use your own LSEG Workspace access

Populate `data/raw/` with the four CSVs described in `download.sh` (equity
closes, vol surface, OIS curve, CDS curve), matching the column layout
documented in that script's comments, then set
`data.use_synthetic_fallback: false` in `configs/config.yaml`.

## Option 2: Synthetic fallback (default)

With `data.use_synthetic_fallback: true` (the default), `train.py` uses the
**already-calibrated** finite-grid numbers **published in the paper's Table
4** directly (bin edges, discount factors, default increments, scaling
constants) rather than re-deriving them from raw LSEG pulls. This lets the
full quantum-circuit training and amplitude-estimation pipeline run
end-to-end without any proprietary data access, at the cost of not
independently re-validating the market-calibration step itself (CDS
bootstrap, vol-surface bucketing, GBM path simulation are still exercised
with representative synthetic curves — see `train.py::build_finite_grid`).

## Files

- `download.sh` — documents the exact LSEG Workspace pulls needed for Option 1.
- `README_data.md` — this file.

No raw data files are committed to this repository (see `.gitignore`).
