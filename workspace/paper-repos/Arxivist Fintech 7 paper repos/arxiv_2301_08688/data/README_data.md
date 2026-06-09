# Data Setup — LOBSTER LOB Data

## Data Source

This paper uses **LOBSTER** (Limit Order Book System — The Efficient Reconstructor),
a commercial dataset of NASDAQ order book message data.

**Citation**: Huang, R. and Polak, T. (2011). LOBSTER: Limit order book reconstruction system.
Available at SSRN 1977207. [Reference 9 in paper]

## How to Obtain

1. Visit https://lobsterdata.com
2. Request access to **AAPL** Level-1 message data for:
   - Training: **2012-01-01 to 2012-05-16**
   - Test:     **2012-05-17 to 2012-06-30**
3. Download and place files under `data/lobster/`:

```
data/lobster/
├── train/
│   ├── AAPL_2012-01-03.csv
│   ├── AAPL_2012-01-04.csv
│   └── ...
└── test/
    ├── AAPL_2012-05-17.csv
    └── ...
```

## Expected CSV Format

Each file should contain 100ms-resolution LOB snapshots with columns:

```
time, bid_price, bid_volume, ask_price, ask_volume, mid_price
```

## Synthetic Fallback

If LOBSTER data is not available, the `LOBDataset` automatically generates
synthetic random-walk price data for testing purposes. Synthetic data will
NOT reproduce the paper's results but is sufficient for:
- Unit testing the pipeline
- Running the Jupyter notebook demos
- Debugging the training loop

To use synthetic data, simply run `train.py` with the default config —
the dataset loader will detect missing files and fall back automatically.

## Paper Training Details

- Only the **first hour** of each trading day (09:30–10:30) is used
- Each trading day = one RL episode
- Training: ~5.5 months × ~21 trading days/month ≈ **115 episodes**
- Test: ~1.5 months × ~21 trading days/month ≈ **31 episodes** (paper confirms 31)
