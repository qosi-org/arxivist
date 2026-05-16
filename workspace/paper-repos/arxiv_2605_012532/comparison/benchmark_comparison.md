# Benchmark Comparison Report
**Paper**: AGENTICAITA: A Proof-of-Concept About Deliberative Multi-Agent Reasoning for Autonomous Trading Systems  
**Paper ID**: arxiv_2605_012532  
**Comparison Date**: 2026-05-16  
**Reproducibility Score**: PENDING (no user results submitted yet)

> **To fill this report**: Run the system for ≥ 500 trades / 90 days in DRY_RUN mode,
> then run `python compute_metrics.py --db data/episodic_memory.db --output results/my_results.json`
> and return to ArXivist with that file.

---

## Paper Targets (Table 5 — DRY_RUN, Apr 6–11 2026, n=157 invocations)

| Metric | Paper Value | Your Value | Deviation | Severity |
|--------|-------------|------------|-----------|----------|
| Total invocations | 157 | — | — | — |
| Assets monitored | 117 | — | — | — |
| Assets with trades | 76 | — | — | — |
| Trades executed | 139 | — | — | — |
| Win rate | 51.80% | — | — | — |
| Agentic Friction F (Eq. 8) | 11.5% | — | — | — |
| Analyst self-abstention rate | 8.3% | — | — | — |
| RM rejection rate | 3.2% | — | — | — |
| Net PnL | -$15.07 | — | — | — |
| Profit factor | 0.841 | — | — | — |
| Mean risk/reward | 3.02:1 | — | — | — |
| Max drawdown | $32.30 | — | — | — |
| Alpha vs BTC buy-hold | +$3,896 | — | — | — |

---

## Statistical Caveat (from paper)

The paper explicitly acknowledges n=139 trades is **statistically insufficient** (binomial p=0.34 for H₀: WR=0.50).  
Minimum recommended: **≥500 trades / 90 days** before drawing conclusions about edge.

A "reproduction" of this proof-of-concept therefore validates:
1. System runs without errors over 5+ days
2. Agentic friction F ≈ 10–15% (LLM behavior is stochastic but should be in this range)
3. RM hard gate rejection rate ≈ 3–5% (deterministic — should be reproducible exactly)
4. Pipeline audit trail is complete (every invocation logged with agent reasoning)

Win rate and PnL are **not reproducible** in the strict sense — they depend on live market conditions.

---

## Recommended Comparison Protocol

1. Run DRY_RUN for ≥ 7 days with the same asset universe
2. Focus on **structural metrics** (friction rate, gate rejection rate, invocation count)
3. Compare **qualitative behavior**: do Analyst outputs show coherent reasoning? Does episodic memory affect decisions?
4. Compare transaction cost sensitivity (Table 7) by running `python backtest_costs.py`
