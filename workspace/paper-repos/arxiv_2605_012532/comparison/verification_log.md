# Verification Log
**Paper**: AGENTICAITA (arxiv_2605_012532)  
**Log Date**: 2026-05-16T00:00:00Z  
**ArXivist Pipeline Run**: Stage 1–6

---

## Pipeline Execution Record

| Stage | Agent | Status | Duration | Outputs |
|-------|-------|--------|----------|---------|
| 1 | Paper Parser | COMPLETE | — | sir.json (v1, confidence 0.90) |
| 2 | SIR Registry | COMPLETE | — | global_index.json, metadata.json, versions/sir_v1.json |
| 3 | Architecture Planner | COMPLETE | — | architecture_plan.json, architecture_plan_summary.md |
| 4 | Code Generator | COMPLETE | — | 27 source files, 30 repo files total |
| 5 | Notebook Generator | COMPLETE | — | reproduce_arxiv_2605_012532.ipynb (19 cells) |
| 6 | Results Comparator | PARTIAL | — | Scaffolds produced; user results pending |

---

## SIR Integrity Record

- **SIR version**: 1
- **Overall confidence**: 0.90
- **Section confidences**: architecture 0.94, mathematical_spec 0.97, training_pipeline 0.99, evaluation_protocol 0.97, implementation_assumptions 0.65
- **Equations extracted**: 13 (Eq. 1–13)
- **Implementation assumptions**: 7 (confidence range: 0.50–0.90)
- **Ambiguities logged**: 3

---

## Paper Metrics Available for Comparison

13 metrics extracted from Table 5 / Section 5.2:

1. Total invocations: 157
2. Assets monitored: 117
3. Assets with trades: 76
4. Trades executed: 139
5. Win rate: 51.80%
6. Agentic Friction F: 11.5%
7. Analyst self-abstention rate: 8.3%
8. RM rejection rate: 3.2%
9. Net PnL: -$15.07
10. Profit factor: 0.841
11. Mean risk/reward: 3.02:1
12. Max drawdown: $32.30
13. Alpha vs BTC buy-hold: +$3,896

---

## User Results Comparison Status

- **User results received**: NO
- **Comparison run**: NO
- **Reproducibility score**: PENDING
- **Action**: Submit results via `compute_metrics.py --output results/my_results.json` and return to ArXivist

---

## Hallucination Audit Summary

- Structural hallucinations: 0
- Parametric hallucinations: 6 (all documented with `# ASSUMED` comments in code)
- Omission hallucinations: 2 (exchange adapter STUB, partial agent prompts)
- Agno framework replacement: documented architectural deviation, not hallucination

---

## Reproducibility Classification

This paper falls into a special category: **non-reproducible by design for performance metrics**.

The paper's PoC results (win rate, PnL, alpha) depend on live DEX market conditions during Apr 6–11 2026 — a specific historical window. Those exact market conditions cannot be replicated.

**What CAN be reproduced** (deterministic):
- Risk Manager hard gate rejection rate (~3–4% of admitted invocations)
- IGP mutex behavior (concurrency serialization)
- AZTE trigger math (Eq. 1–3) on any price series
- CBD score computation (Eq. 9–11) on any price history
- Transaction cost formulas (Eq. 12–13)

**What CANNOT be reproduced** (market-dependent):
- Win rate (51.80%) — depends on future/past price moves
- Net PnL (-$15.07) — live market outcome
- Analyst self-abstention rate (8.3%) — partially LLM stochastic

---

## Configuration Used for Code Generation

All hyperparameters sourced from paper Table 2 and equations, except those marked `# ASSUMED`. No hyperparameters were invented without documentation.

---

*End of verification log. To complete Stage 6, submit user experimental results.*
