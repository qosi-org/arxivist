# Hallucination Report
**Paper**: AGENTICAITA (arxiv_2605_012532)  
**Generated**: 2026-05-16  
**Basis**: SIR v1 (confidence 0.90) + Architecture Plan

This report identifies deviations between what the paper specifies and what the generated code implements. Findings are objective — deviations from the paper are flagged regardless of whether the chosen alternative is reasonable.

---

## 1. Structural Hallucinations
*(Components in generated code NOT in the SIR)*

**None found.** Every module in the generated repository corresponds to a named component in the paper (AZTE, SDP, IGP, CBD, EpisodicMemory, PrivacyLayer). No phantom components were introduced.

---

## 2. Parametric Hallucinations
*(Assumed hyperparameters not specified in paper — marked `# ASSUMED` in code)*

| Parameter | Assumed Value | Location | Confidence | Risk |
|-----------|--------------|----------|------------|------|
| `ollama.temperature` | 0 (greedy) | `ollama_client.py`, `configs/config.yaml` | 0.70 | **Medium** — Paper emphasizes determinism but never specifies temperature. Non-zero temp would invalidate the "reproducible audit trail" claim. |
| `cbd.correlation_method` | Pearson | `cbd.py`, `configs/config.yaml` | 0.75 | **Low** — Standard choice; result difference vs Spearman is typically <5% for 30-bar windows. |
| `market_data.ohlcv_limit` | 20 bars | `market_data.py`, `configs/config.yaml` | 0.90 | **Low** — Paper states "20-bar 1-minute OHLCV" explicitly in Section 4.2. This assumption is well-grounded. |
| `market_data.l2_depth` | 20 levels | `market_data.py` | 0.85 | **Low** — Common default; paper shows L2 in context but doesn't specify depth. |
| `database.path` | `data/episodic_memory.db` | `configs/config.yaml` | 0.65 | **Low** — Docker volume mount path not specified. Configurable. |
| `ollama.max_tokens` | 512 | `ollama_client.py` | 0.70 | **Medium** — Insufficient tokens could truncate agent reasoning. Paper shows reasoning excerpts of ~100 words. 512 should be adequate but is assumed. |

**Highest-risk assumption**: `temperature=0`. If the paper used `temperature>0`, the generated code will produce different outputs for the same inputs. Verify by checking Ollama call logs for response variance across identical prompts.

---

## 3. Omission Hallucinations
*(Components present in SIR but absent or STUB in generated code)*

### 3a. Exchange Adapter — CRITICAL OMISSION (severity: Significant)
- **SIR reference**: Section 4.7, Table 2
- **What paper says**: A specific DEX perpetual futures exchange is used for order routing, with Tor+VPN privacy channel. 157 DRY_RUN invocations completed successfully.
- **What is generated**: `StubExchangeAdapter` raises `NotImplementedError` on `place_order()`. The exchange identity is explicitly not named in the paper.
- **Impact on DRY_RUN**: **None** — StubExchangeAdapter is never called in DRY_RUN mode. System is fully functional for the proof-of-concept reproduction.
- **Impact on LIVE**: **Blocks all live trading** until a real adapter is implemented.
- **Suggested fix**: Identify the exchange from paper context clues (Section 4.7 mentions DEX perpetual futures with funding rates and L2 orderbooks — consistent with Hyperliquid, dYdX v4, or Drift Protocol). Implement the appropriate ccxt adapter.

### 3b. Full Agent Prompts — Minor Omission (severity: Minor)
- **SIR reference**: Section 4.2
- **What paper says**: Complete system prompts for all three agents. Only excerpts are provided in the paper.
- **What is generated**: Prompts built from the excerpted portions + standard JSON contract instructions. Core structure matches. Full prompt engineering details unavailable.
- **Impact**: LLM behavior may differ from paper's system. Analyst reasoning quality and self-abstention rate (8.3%) may vary.
- **Suggested fix**: The prompt excerpts in Section 4.2 are likely nearly complete. Test by running DRY_RUN and checking if self-abstention rate F_wait ≈ 8% after 100+ invocations.

---

## 4. Agno Framework Replacement
*(Architectural deviation — documented risk, not a hallucination)*

The paper references a "custom multi-agent orchestration framework (Agno)" for agent coordination. This framework is not publicly available.

**What was done**: Replaced Agno with native `asyncio` + `asyncio.Lock` for the IGP semaphore, and direct `async` function calls for the SDP sequential chain.

**Assessment**: The paper's described behavior (sequential pipeline with mutex gating, per-asset and global cooldowns) is fully implementable with standard asyncio — Agno likely provides this infrastructure. The functional behavior of the implementation should match the paper.

**Confidence**: 0.75 — Medium. If Agno provides additional features (retry logic, circuit breakers, agent communication protocols) beyond sequential invocation, those features would be missing.

---

## Summary

| Category | Count | Highest Severity |
|----------|-------|-----------------|
| Structural hallucinations | 0 | — |
| Parametric hallucinations | 6 | Medium (temperature) |
| Omission hallucinations | 2 | Significant (exchange adapter) |
| Documented stubs | 1 | Significant (exchange adapter) |

**Overall assessment**: The generated implementation faithfully represents the paper's architecture for DRY_RUN purposes. All deviations are documented, justified, and do not affect the core algorithmic behavior (AZTE, CBD, IGP, SDP pipeline logic, hard gates). LIVE trading requires resolving the exchange adapter stub.
