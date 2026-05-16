# AGENTICAITA: Deliberative Multi-Agent Reasoning for Autonomous Trading

**ArXivist-generated reproduction repository**

> **Paper**: [AGENTICAITA: A Proof-of-Concept About Deliberative Multi-Agent Reasoning for Autonomous Trading Systems](https://arxiv.org/abs/2605.12532)  
> **Author**: Ivan Letteri (University of L'Aquila)  
> **arXiv**: 2605.12532 | Submitted: 2026-05-01

---

## What This Paper Does

AGENTICAITA replaces the traditional signal-then-execute algorithmic trading paradigm with a **fully autonomous deliberative loop** in which three specialized LLM agents reason, negotiate, and act in concert — without any offline training or human intervention.

The core contribution is the **Sequential Deliberative Pipeline (SDP)**: an Analyst agent produces a typed trading signal; a Risk Manager applies deterministic hard gates plus LLM validation; an Executor logs or places the order. An **Adaptive Z-Score Trigger Engine (AZTE)** gates LLM activation to statistically anomalous market events only, and an **Inference Gating Protocol (IGP)** serializes concurrent activations via a binary mutex.

Validated over 5 days: 157 autonomous invocations, 76 assets, zero human interventions, 11.5% agentic friction, and +14.94pp alpha relative to BTC buy-and-hold.

---

## Quick Start

```bash
# 1. Clone and install
git clone <repo>
cd arxiv_2605_12532
pip install -e .

# 2. Pull the LLM (requires Ollama: https://ollama.ai)
ollama pull qwen3:8b

# 3. Run a dry-run session (mock market data, no real orders)
python scripts/run_dryrun.py --ticks 100 --log-level INFO

# 4. Evaluate session metrics
python scripts/evaluate.py --db data/agenticaita.db

# 5. Open the reproduction notebook
jupyter lab notebooks/reproduce_arxiv_2605_12532.ipynb
```

---

## Installation

**pip:**
```bash
pip install -r requirements.txt
pip install -e .
```

**conda:**
```bash
conda create -n agenticaita python=3.11
conda activate agenticaita
pip install -r requirements.txt
pip install -e .
```

**Docker:**
```bash
docker-compose -f docker/docker-compose.yml up agenticaita
```

---

## Repository Structure

```
src/agenticaita/
├── trigger/azte.py        ← AZTE: rolling Z-score trigger (Eqs. 1–3)
├── pipeline/
│   ├── igp.py             ← IGP: mutex scheduler (Definition 2)
│   ├── sdp.py             ← SDP: Analyst → RM → Executor (Figure 2)
│   └── contracts.py       ← Typed JSON contracts (Pydantic)
├── agents/
│   ├── analyst.py         ← Analyst LLM agent
│   ├── risk_manager.py    ← Risk Manager: hard gates + LLM (Eqs. 4–7)
│   └── executor.py        ← Executor: DRY_RUN / LIVE
├── scoring/cbd.py         ← CBD composite score (Eqs. 9–11)
├── memory/episodic.py     ← SQLite WAL: trades, vol_history, pipeline_log
├── data/mock_feed.py      ← Synthetic market feed (offline testing)
├── evaluation/
│   ├── metrics.py         ← Agentic Friction (Eq. 8), win rate, drawdown
│   └── cost_model.py      ← Round-trip cost model (Eqs. 12–13)
└── utils/config.py        ← Config loading + seed utility
scripts/
├── run_dryrun.py          ← Main entry point
├── evaluate.py            ← Metrics report from trades DB
└── replay.py              ← Pipeline audit log replay
```

---

## Key Commands

### Dry-run session
```bash
# Default assets, 200 ticks (mock data)
python scripts/run_dryrun.py --ticks 200

# Custom asset list
python scripts/run_dryrun.py --assets BTC ETH SOL FARTCOIN --ticks 500

# Full session (runs indefinitely at 60s intervals)
python scripts/run_dryrun.py --config configs/default.yaml
```

### Evaluate results
```bash
# All scenarios
python scripts/evaluate.py --db data/agenticaita.db

# With transaction cost scenario
python scripts/evaluate.py --db data/agenticaita.db --cost-scenario realistic

# Save to JSON
python scripts/evaluate.py --db data/agenticaita.db --out results/metrics.json
```

### Audit pipeline
```bash
python scripts/replay.py --db data/agenticaita.db
python scripts/replay.py --db data/agenticaita.db --event-type pipeline_start
```

---

## Expected Results (Paper, Table 5)

| Metric | Paper Value |
|--------|-------------|
| Total trades | 139 |
| Win rate | 51.80% |
| Net PnL | –$15.07 (–0.058% on $26,079 notional) |
| Profit factor | 0.841 |
| Max drawdown | $32.30 |
| Agentic Friction F | 11.5% |
| Long signal rate | 90.4% |
| Alpha vs BTC buy-and-hold | +14.94pp |
| Binomial p-value | 0.34 (not significant at n=139) |

**Statistical scope caveat** (from paper): The 5-day, 139-trade window demonstrates operational correctness but provides insufficient power for performance inference. A minimum of 500 trades over 90 days is required before performance conclusions can be drawn.

---

## Implementation Assumptions

| Assumption | Confidence | Basis |
|-----------|-----------|-------|
| Ollama REST API at localhost:11434 | 0.95 | Stated in paper |
| SQLite WAL mode for episodic memory | 0.98 | Explicitly stated |
| DEX exchange: unknown (mock feed used) | 0.45 | Not identified in paper |
| `Agno` framework: reimplemented as asyncio | 0.55 | Not open-sourced |
| 0.05% taker fee for cost model | 0.60 | Assumed; verify for target DEX |
| Per-asset cooldown 300s, IGP cooldown 1800s | 0.99 | Table 2 |
| kappa=0.5 for CBD saturation | 0.99 | Explicit in Eq. 10 |
| Tor SOCKS5h at localhost:9050 | 0.70 | Standard Tor config |

---

## Reproducibility Notes

### Known deviations from paper
1. **DEX API**: The paper does not identify the exchange. `MockMarketFeed` generates synthetic data. For live use, subclass `MarketFeed` in `src/agenticaita/data/market_feed.py`.
2. **Agno framework**: The paper's custom orchestration framework is not publicly available. This implementation uses native `asyncio`. Behavior should be equivalent.
3. **Eq. 13 half-spread term**: A potential notational ambiguity exists in the paper's half-spread formula `(1/(2*Q_i)) * |P_ask - P_bid|`. We implement it literally; users should verify intent.
4. **LLM directional bias**: The 90.4% long signal rate is a documented paper limitation (LLM optimism bias), not a code error.

### Low-confidence sections (SIR flags)
- Exchange identity (0.45): cannot reproduce live market interaction without knowing target DEX
- Agno framework (0.55): re-implemented; interface may differ slightly from paper's system

---

## Citation

```bibtex
@article{letteri2026agenticaita,
  title={AGENTICAITA: A Proof-of-Concept About Deliberative Multi-Agent Reasoning for Autonomous Trading Systems},
  author={Letteri, Ivan},
  journal={arXiv preprint arXiv:2605.12532},
  year={2026}
}
```

---

*Repository generated by [ArXivist](https://github.com/arxivist) — ArXivist v1.0*
