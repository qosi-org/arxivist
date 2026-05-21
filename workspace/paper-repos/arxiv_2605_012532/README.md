# AGENTICAITA

**A Proof-of-Concept About Deliberative Multi-Agent Reasoning for Autonomous Trading Systems**

Ivan Letteri — University of L'Aquila, Italy
arXiv: [2605.12532](https://arxiv.org/abs/2605.12532)

---

## What this paper does

AGENTICAITA replaces the traditional signal-then-execute trading paradigm with an autonomous deliberative loop. Three specialized LLM agents — an Analyst, a Risk Manager, and an Executor — reason, negotiate, and act on perpetual futures markets without offline training or human intervention. The system is activated only on statistically anomalous market conditions (via Z-score gating), serialized through a mutex-based inference gate, and enriched with cross-episode episodic memory.

The proof-of-concept ran 157 DRY_RUN invocations across 76 assets over 5 days (April 6–11, 2026), achieving 51.8% win rate on 139 trades with 11.5% agentic friction.

---

## Quick Start

```bash
# 1. Install
git clone <this-repo>
cd agenticaita
pip install -e ".[dev]"

# 2. Configure
cp configs/config.yaml configs/config.local.yaml
# Edit: set ollama.base_url, exchange.adapter, and populate configs/assets.txt

# 3. Start Ollama with qwen3.5:9b
ollama pull qwen3.5:9b

# 4. Run (DRY_RUN — no real orders)
python run.py --config configs/config.local.yaml --mode DRY_RUN

# 5. Inspect results
python compute_metrics.py --db data/episodic_memory.db --output results/metrics.json
python inspect_trades.py  --db data/episodic_memory.db
```

---

## Installation

### pip
```bash
pip install -r requirements.txt
pip install -e .
```

### conda
```bash
conda create -n agenticaita python=3.11
conda activate agenticaita
pip install -r requirements.txt
pip install -e .
```

### Docker
```bash
docker build -t agenticaita -f docker/Dockerfile .
docker run -v $(pwd)/data:/app/data -v $(pwd)/configs:/app/configs agenticaita
```

---

## System Architecture

```
MonitoringPoller (60s) ──► AZTE (Eq. 1-3) ──► IGP (Def. 2) ──► SDP Pipeline
                                                                      │
                              CBD (Eq. 9-11) ──────────────────────────┤
                                                                      │
                                                            Analyst Agent
                                                                  ▼
                                                          Risk Manager Agent
                                                       (4 hard gates + LLM)
                                                                  ▼
                                                          Executor Agent
                                                       (DRY_RUN | LIVE)
                                                                  ▼
                                                       EpisodicMemory (SQLite)
```

**Key components:**
- **AZTE**: Z-score trigger gates (Eq. 1–3): fires only on σ≥2 anomalies or |r_t|≥0.3%
- **IGP**: Binary semaphore + 30min global cooldown — serializes all pipeline runs
- **CBD**: Composite score (Eq. 9–11) — rewards decorrelated, high-volatility assets
- **SDP**: Analyst→RiskMgr→Executor chain with typed JSON contracts
- **EpisodicMemory**: SQLite WAL-mode store — reasoning traces become future context

---

## Expected Results (Paper, Table 5)

| Metric | Paper Value |
|--------|-------------|
| Total invocations | 157 |
| Trades executed | 139 |
| Win rate | 51.80% |
| Net PnL | −$15.07 |
| Profit factor | 0.841 |
| Max drawdown | $32.30 |
| Mean win | +$1.11 |
| Mean loss | −$1.41 |
| Mean R:R | 3.02:1 |
| Agentic friction F | 11.5% |
| Binomial p-value | ~0.34 (not significant) |
| Alpha vs BTC B&H | +$3,896 (+14.94 pp) |

> **Statistical note:** 139 trades is insufficient for significance (p≈0.34). The paper explicitly recommends ≥500 trades / 90 days before drawing conclusions.

---

## Configuration

All parameters in `configs/config.yaml`. Key values (all from paper Table 2):

| Parameter | Value | Equation |
|-----------|-------|----------|
| Z-score threshold | 2.0σ | Eq. 3 |
| Rolling window | 30 bars | Eq. 2 |
| Absolute return floor | 0.3% | Eq. 3 |
| Confidence gate | 0.60 | Eq. 5 |
| Max stop-loss distance | 2% | Eq. 6 |
| Max position size | $500 | Eq. 7 |
| Per-asset cooldown | 300s | Table 2 |
| Global cooldown | 1800s | Table 2 |
| CBD α | 0.5 | Eq. 11 |
| CBD κ | 0.5 | Eq. 10 |

---

## Reproducibility Notes

### Known deviations and stubs

1. **Exchange adapter (HIGH):** The paper does not name the DEX used. `StubDEXAdapter` is provided; replace with `HyperliquidAdapter`, `DYdXAdapter`, etc. LIVE trading is blocked until a real adapter is implemented.

2. **Agno framework (HIGH):** The paper uses a custom "Agno" multi-agent orchestration framework not publicly available. This repo reimplements the orchestration using native Python `asyncio` — functionally equivalent per the paper's descriptions.

3. **Full agent prompts (MEDIUM):** Only prompt excerpts are given in the paper. The prompts in `agents/analyst.py` and `agents/risk_manager.py` follow the excerpted patterns exactly; expect some variation in LLM output quality.

4. **LLM temperature (MEDIUM, ASSUMED=0):** Not specified. Set to 0 for reproducibility; adjust in `configs/config.yaml`.

5. **Correlation method for CBD (LOW, ASSUMED=Pearson):** Eq. 9 specifies Pearson correlation by implication; configurable to Spearman via `cbd.correlation_method`.

### Confidence summary (from SIR)

| Section | Confidence |
|---------|-----------|
| Mathematical spec | 0.97 |
| Architecture | 0.94 |
| Evaluation protocol | 0.97 |
| Training pipeline | 0.99 (training-free) |
| Implementation assumptions | 0.65 ⚠ |
| **Overall SIR** | **0.90** |

---

## Citation

```bibtex
@article{letteri2026agenticaita,
  title={AGENTICAITA: A Proof-of-Concept About Deliberative Multi-Agent Reasoning for Autonomous Trading Systems},
  author={Letteri, Ivan},
  journal={arXiv preprint arXiv:2605.12532},
  year={2026},
  institution={University of L'Aquila}
}
```
