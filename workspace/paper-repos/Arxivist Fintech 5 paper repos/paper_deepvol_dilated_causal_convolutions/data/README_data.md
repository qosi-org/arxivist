# Data Requirements for DeepVol

## Dataset Description

DeepVol requires **NASDAQ-100 high-frequency intraday data** (Section 2.1):
- Time period: September 30, 2019 – September 30, 2021
- ~90 constituent tickers (listed in paper Tables A1 & A2)
- Sampling frequencies: 1, 5, 15, 30, 60 minutes
- **Optimal**: 5-minute bars (Table 2)

## Data Source

This data is **not publicly available** for free. Options:
- [Refinitiv Tick History (LSEG)](https://www.lseg.com/en/data-analytics/market-data/tick-history)
- [Bloomberg Terminal](https://www.bloomberg.com/professional/product/market-data/)
- [Polygon.io](https://polygon.io) — commercial API with US equity minute bars

## Expected Directory Structure

```
data/
├── raw/
│   └── {TICKER}/
│       └── {TICKER}_5min_2019-2021.csv  ← columns: datetime, open, high, low, close, volume
└── processed/
    ├── X_train.npy    ← [N_train, 1, 78]  float32
    ├── y_train.npy    ← [N_train, 1]       float32
    ├── X_val.npy
    ├── y_val.npy
    ├── X_test.npy
    └── y_test.npy
```

## Preprocessing

After placing raw data, run:
```bash
python data/preprocess.py --config configs/config.yaml
```

## Synthetic Data Fallback

Without real data, all scripts fall back to synthetic GARCH-simulated data for testing.
