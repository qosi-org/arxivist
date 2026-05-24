# Data Requirements — Gu, Kelly, Xiu (2020)

## ⚠️ Data Availability

The paper uses three proprietary data sources. **None are freely available.**

---

## 1. CRSP Monthly Stock Returns

**Source**: Center for Research in Security Prices (CRSP)  
**Access**: WRDS subscription — https://wrds-web.wharton.upenn.edu  
**Coverage**: All NYSE/AMEX/NASDAQ common stocks, 1957-03 to 2016-12  
**Size**: ~30,000 stocks over 60 years

Required fields per stock-month:
- `permno`: CRSP stock identifier
- `date`: Year-month
- `ret`: Monthly total return
- `prc`: Closing price
- `shrout`: Shares outstanding
- `exchcd`: Exchange code (1=NYSE, 2=AMEX, 3=NASDAQ)
- `siccd`: SIC industry code (first 2 digits → 74 industry dummies)

Risk-free rate: 1-month T-bill rate from Ibbotson Associates via CRSP.

---

## 2. Stock Characteristics (94 Features)

**Source**: Green, Hand, and Zhang (2017) characteristics  
**Access**: Partially public via Jeremiah Green's website  
**Coverage**: 94 characteristics, 1957-2016

- 61 updated annually
- 13 updated quarterly
- 20 updated monthly

**Publication lags enforced** (to avoid look-ahead bias):
- Monthly: 1 month lag
- Quarterly: 4 month lag
- Annual: 6 month lag

See Internet Appendix Table A.6 of the paper for the full list of 94 characteristics.
Key signals include: size (mvel1), book-to-market (bm), momentum (mom12m, mom1m),
idiosyncratic volatility (idiovol), turnover (turn), Amihud illiquidity (ill),
bid-ask spread (baspread), and 86 more.

---

## 3. Macroeconomic Predictors (8 Variables)

**Source**: Welch and Goyal (2008)  
**Access**: **FREE** — available from Amit Goyal's website:  
https://sites.google.com/view/agoyal145

Variables: `dp` (dividend-price), `ep` (earnings-price), `bm` (book-to-market),
`ntis` (net equity issuance), `tbl` (T-bill rate), `tms` (term spread),
`dfy` (default spread), `svar` (stock variance).

---

## Expected File Format

Place files at:
```
data/raw/crsp_returns.parquet       ← CRSP monthly returns
data/raw/characteristics.parquet    ← 94 stock characteristics
data/raw/macro_predictors.csv       ← 8 Welch-Goyal macro predictors
```

### crsp_returns.parquet columns
```
year_month (str), permno (int), excess_ret (float32), mkt_cap (float32),
sic2 (int), [94 characteristic columns], exchange (int)
```

### macro_predictors.csv columns
```
date (YYYY-MM), dp, ep, bm, ntis, tbl, tms, dfy, svar
```

---

## Development Without Real Data

Use the built-in synthetic generator:
```python
from asset_pricing_ml.data.dataset import SyntheticDataGenerator
gen = SyntheticDataGenerator(n_stocks=500, n_months=720, seed=42)
data = gen.generate()
```

Or run in debug mode:
```bash
python train.py --config configs/config.yaml --debug
python train_all.py --config configs/config.yaml --debug
```

⚠️ Synthetic data results will **NOT** match the paper's reported numbers.
The generator uses a simple random factor model — it exists only to verify
the code runs end-to-end without WRDS access.
