# Domain: Finance — Parsing Hints (Stage 1 Enrichment)

Load alongside `agents/01_paper_parser.md` when the detected domain is **Finance**.

---

## Architecture extraction — Finance-specific rules

Finance papers describe **strategies, factor models, and signal pipelines** not neural modules.
Reframe the architecture accordingly:

**Signal generation pipeline:** extract each step:
- Raw data inputs (OHLCV, order book, fundamentals, alternative data)
- Feature engineering steps (rolling windows, normalisation, cross-sectional ranking)
- Signal model (linear factor model, ML model, neural network)
- Position sizing rule (equal weight, volatility targeting, mean-variance optimisation)
- Transaction cost model (fixed commission, market impact, bid-ask spread)

**Factor models:** extract each factor explicitly:
- Factor name, construction method, rebalancing frequency
- Cross-sectional vs time-series construction
- Winsorisation and neutralisation steps

**Neural architectures for finance:** extract in addition to generic parsing:
- Input feature set (which market data, which frequency — tick/1min/daily/monthly)
- Look-back window length (critical — always extract)
- Prediction target (return, direction, volatility, spread)
- Whether outputs are raw signals or position weights

---

## Training pipeline — Finance-specific rules

**The train/validation/test split in finance is temporal — never random.**
Extract exact date ranges for each split. If not given, flag as a critical ambiguity.

**Walk-forward validation:** if paper uses expanding or rolling window validation:
- Window size (training period length)
- Step size (refit frequency)
- Whether lookback is fixed or expanding

**Transaction costs:** always extract the assumed cost model. If not stated, add as an
implementation assumption with `confidence: 0.4` — results change dramatically with costs.

**Rebalancing frequency:** always extract. Daily vs monthly rebalancing produces entirely
different results on the same strategy.

---

## Evaluation — Finance-specific rules

**Always extract all of:**
- Sharpe ratio (annualised) and its annualisation factor (252 for daily, 12 for monthly)
- Maximum drawdown and drawdown period
- Sortino ratio if reported
- Calmar ratio if reported
- Turnover (portfolio turnover per period)
- Transaction cost assumption used in evaluation
- Benchmark used for comparison (S&P 500, risk-free rate, equal-weight)
- Whether results are gross or net of transaction costs — this distinction is critical

**In-sample vs out-of-sample:** always note which results are which. In-sample results
without out-of-sample validation are not reproducible in a meaningful sense.
