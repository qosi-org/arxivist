# Domain: Finance — Code Conventions (Stage 4 Enrichment)

Load alongside `agents/04_code_generator.md` when domain is **Finance**.

---

## Point-in-time correctness — non-negotiable

Every data access must use only information available at the time of the decision.
This is the single most important rule in finance code. Violations produce look-ahead bias.

```python
# CORRECT — use data available at time t to make decision at t
signal_t = features.loc[:date_t].rolling(window).mean().iloc[-1]

# WRONG — uses future data
signal_t = features.rolling(window).mean().loc[date_t]
# ↑ rolling() on the full series uses future values in the window
```

Always implement a `PointInTimeDataFeed` class that enforces temporal ordering.

## No-lookahead train/test split

```python
# ALWAYS temporal split — never random
train_data = data[data.index < config.data.train_end]
val_data   = data[(data.index >= config.data.train_end) & (data.index < config.data.val_end)]
test_data  = data[data.index >= config.data.val_end]

# NEVER this:
# train, test = train_test_split(data, test_size=0.2)  # WRONG for time series
```

## Performance metrics implementation

```python
import empyrical as ep

def compute_performance(returns: pd.Series, benchmark: pd.Series = None) -> dict:
    ann_factor = config.evaluation.annualisation_factor
    return {
        "sharpe_ratio":    ep.sharpe_ratio(returns, annualization=ann_factor),
        "sortino_ratio":   ep.sortino_ratio(returns, annualization=ann_factor),
        "max_drawdown":    ep.max_drawdown(returns),
        "calmar_ratio":    ep.calmar_ratio(returns),
        "annual_return":   ep.annual_return(returns, annualization=ann_factor),
        "annual_volatility": ep.annual_volatility(returns, annualization=ann_factor),
        "information_ratio": ep.excess_sharpe(returns, benchmark) if benchmark is not None else None,
    }
```

## Transaction cost application

Always apply transaction costs before reporting results:
```python
def apply_costs(weights: pd.DataFrame, returns: pd.DataFrame, config) -> pd.Series:
    turnover = weights.diff().abs().sum(axis=1)
    cost = turnover * (config.costs.commission_bps + config.costs.slippage_bps) / 10000
    gross_returns = (weights.shift(1) * returns).sum(axis=1)
    return gross_returns - cost
```

## What Must NOT be done

- Do NOT use `sklearn.model_selection.train_test_split` on time-series data
- Do NOT compute rolling features on the full dataset before splitting
- Do NOT report in-sample results without clearly labelling them as such
- Do NOT assume adjusted prices are always correct — document the adjustment methodology
- Do NOT hardcode the annualisation factor — always use config
