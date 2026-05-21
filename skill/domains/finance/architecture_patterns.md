# Domain: Finance — Architecture Patterns (Stage 3 Enrichment)

Load alongside `agents/03_architecture_planner.md` when domain is **Finance**.

---

## Standard finance module patterns

**Time series feature extractor:**
```
RawPriceData → LogReturn → RollingMean/Std (multiple windows) → CrossSectionalRank → Features
```

**Factor model pipeline:**
```
UniverseFilter → FactorConstruction → Winsorisation → Neutralisation → FactorScore → Portfolio
```

**Backtesting engine structure:**
```
DataFeed → SignalGenerator → PortfolioConstructor → TransactionCostModel → PerformanceEngine
```

**Neural forecaster for returns:**
```
Input [B, T, F] → TimeSeriesEncoder → Dropout → Linear → Predicted_return [B, H]
```
Where T=lookback, F=features, H=forecast horizon.

---

## Framework and library selection

**Default to Python + pandas for factor models and classical strategies.**

**For neural finance models:** PyTorch with custom time-series dataloaders.

**Recommended libraries:**
```
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.11.0
statsmodels>=0.14.0    # for statistical tests, ARIMA, VAR
scikit-learn>=1.3.0    # for ML-based signals
pyportfolioopt>=1.5.5  # for portfolio optimisation
vectorbt>=0.26.0       # for backtesting (if applicable)
quantstats>=0.0.62     # for performance reporting
empyrical>=0.5.5       # for financial metrics
yfinance>=0.2.28       # for data download (if applicable)
```

**For tick-level or high-frequency papers:**
```
arctic>=1.82.0         # time-series database
numba>=0.58.0          # for performance-critical signal computation
```

---

## Config schema additions for Finance

```yaml
data:
  universe: "SP500"          # stock universe
  start_date: "2000-01-01"
  end_date: "2020-12-31"
  train_end: "2015-12-31"    # temporal split — never random
  val_end: "2017-12-31"
  frequency: "daily"         # tick, 1min, 5min, daily, monthly
  price_field: "adj_close"   # adjusted vs unadjusted

strategy:
  lookback_window: 252       # in periods
  rebalance_freq: "monthly"
  long_short: true
  target_volatility: 0.1     # annualised — null if not used

costs:
  commission_bps: 5          # basis points per trade
  slippage_bps: 2
  market_impact: null        # model if specified

evaluation:
  benchmark: "equal_weight"
  annualisation_factor: 252
  risk_free_rate: 0.0
```
