# AGENTICAITA — Architecture Plan Summary
**Paper:** arxiv_2605_012532 | **Generated:** 2026-05-16

---

## Framework Decision

**No ML framework required.** AGENTICAITA is a pure Python asyncio orchestration system:
- LLM inference via **Ollama REST API** (qwen3.5:9b)
- Persistence via **aiosqlite** (SQLite WAL mode)
- Exchange connectivity via **ccxt** (async, pluggable)
- Config via **pydantic v2** settings + YAML
- No CUDA, no PyTorch, no gradient computation

---

## Module Map

```
src/agenticaita/
├── azte.py            ← Adaptive Z-Score Trigger Engine (Eq. 1–3)
├── cbd.py             ← CBD composite score (Eq. 9–11)
├── igp.py             ← Inference Gating Protocol (asyncio.Lock + cooldowns)
├── memory.py          ← SQLite episodic memory (4 tables)
├── pipeline.py        ← SDP orchestrator: Analyst→RiskMgr→Executor
├── monitor.py         ← 60s polling loop across all assets
├── market_data.py     ← Async OHLCV / L2 / funding rate fetcher
├── exchange.py        ← Abstract DEX adapter + Tor/VPN routing
├── ollama_client.py   ← Thin async Ollama REST wrapper
├── schemas.py         ← All pydantic typed contracts (12 models)
├── config.py          ← Settings loader from config.yaml
└── agents/
    ├── analyst.py     ← Zero-shot Analyst LLM agent
    ├── risk_manager.py← Hard gates (Layer A) + LLM sizing (Layer B)
    └── executor.py    ← DRY_RUN/LIVE order routing
```

---

## Data Flow (Main Loop)

```
Every 60s per asset:
  price → AZTE → TriggerEvent?
    Yes → IGP.acquire()?
      Yes → CBD.compute_omega()
           + MarketData (OHLCV, L2, funding)
           + EpisodicMemory.get_briefing()
           → AnalystAgent → {signal, confidence, entry, SL, TP, size, reasoning}
             signal==wait → log abstention, IGP.release()
             else → RiskManager.hard_gate_check()
               FAIL → log rejection, IGP.release()
               PASS → RiskManager.llm_validate() → {approved, size_usd}
                       → ExecutorAgent.execute(DRY_RUN|LIVE)
                       → EpisodicMemory.store_trade()
                       → IGP.release()
      No → log pipeline_busy
```

---

## Key Config Parameters (from paper, all confirmed)

| Parameter | Value | Source |
|-----------|-------|--------|
| polling_interval | 60s | Table 2 |
| z_score_threshold | 2.0σ | Table 2 |
| rolling_window | 30 bars | Table 2 |
| absolute_floor | 0.003 (0.3%) | Eq. 3 |
| confidence_gate | 0.60 | Table 2 / Eq. 5 |
| max_stop_loss_pct | 2% | Table 2 / Eq. 6 |
| max_position_usd | $500 | Table 2 / Eq. 7 |
| per_asset_cooldown | 300s | Table 2 |
| igp_global_cooldown | 1800s | Table 2 |
| cbd_alpha | 0.5 | Eq. 11 |
| cbd_kappa | 0.5 | Eq. 10 |
| llm_temperature | 0 (assumed) | ⚠ ASSUMED |

---

## SQLite Schema (4 tables)

**trades**: id, asset, timestamp, signal, confidence, entry_price, stop_loss,
            take_profit, size_usd, reasoning, approved, final_size, pnl, mode

**vol_history**: id, asset, timestamp, price, z_score, r_t

**pipeline_log**: id, timestamp, asset, event_type, detail

**ollama_calls**: id, timestamp, agent, model, prompt_tokens, completion_tokens, latency_ms

---

## Implementation Risks

| Severity | Risk | Mitigation |
|----------|------|------------|
| 🔴 High | Agno framework is custom/unavailable | Replace with asyncio.Lock + direct async calls |
| 🔴 High | DEX exchange identity unknown | Abstract adapter; Hyperliquid + dYdX examples |
| 🟡 Medium | Full agent prompts not published | Use excerpts; mark prompt tuning TODOs |
| 🟡 Medium | LLM temperature unspecified | Default 0; make configurable |
| 🟡 Medium | Tor runtime dependency | Safety gate; DRY_RUN works without Tor |
| 🟢 Low | CBD correlation method unspecified | Pearson (numpy); Spearman as config option |

---

## Entrypoints

- `run.py` — main trading loop (`--config`, `--mode`, `--assets`)
- `compute_metrics.py` — reproduce Table 5 from trades DB
- `backtest_costs.py` — reproduce Table 7 cost sensitivity
- `inspect_trades.py` — query/export trade records
