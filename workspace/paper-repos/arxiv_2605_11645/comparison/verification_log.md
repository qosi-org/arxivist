# Verification Log
**Paper**: GeomHerd (arXiv:2605.11645)  
**Paper ID**: `arxiv_2605_11645`  
**Audit Date**: 2026-05-13T00:00:00Z  
**ArXivist Stage**: 6 — Results Comparator  
**Auditor**: ArXivist automated pipeline

---

## 1. Input Documents

| Document | Version | SHA256 (first 16 chars) | Status |
|----------|---------|------------------------|--------|
| `sir.json` | v1 | `681f182ebf1095b0` | Loaded successfully |
| `architecture_plan.json` | v1 | `5a19ac307630ec88` | Loaded successfully |
| `metadata.json` | v1 | — | Loaded successfully |
| User experimental results | N/A | N/A | **NOT SUBMITTED** |

---

## 2. Pipeline Execution Summary

| Stage | Status | Notes |
|-------|--------|-------|
| Stage 1 — Paper Parser | ✅ Complete | SIR confidence 0.80; 4 ambiguities flagged |
| Stage 2 — SIR Registry | ✅ Complete | v1 committed; global index updated |
| Stage 3 — Architecture Planner | ✅ Complete | 14 modules, 6 risks, full config.yaml |
| Stage 4 — Code Generator | ✅ Complete | 37 files, 28 Python modules, 0 syntax errors |
| Stage 5 — Notebook Generator | ✅ Complete | 2 notebooks, 28 cells, all code cells validated |
| Stage 6 — Results Comparator | ✅ Complete (pre-run) | Awaiting user experimental results |

---

## 3. Metrics Comparison Status

| Paper Metric | SIR Location | User Result | Match Status |
|-------------|-------------|-------------|--------------|
| Median lead recall (272 steps) | `geomherd_results.kappa_plus_recall_oriented.median_lead` | PENDING | UNMATCHED |
| Median lead precision (178 steps) | `geomherd_results.kappa_plus_precision_oriented.median_lead` | PENDING | UNMATCHED |
| Recall_super precision (0.04) | `geomherd_results.kappa_plus_precision_oriented.recall_super` | PENDING | UNMATCHED |
| FAR_sub precision (0.07) | `geomherd_results.kappa_plus_precision_oriented.FAR_sub` | PENDING | UNMATCHED |
| Precision (0.45) | `geomherd_results.kappa_plus_precision_oriented.precision` | PENDING | UNMATCHED |
| β⁻ median lead (318 steps) | `geomherd_results.beta_minus_contagion.median_lead` | PENDING | UNMATCHED |
| β⁻ Recall_super (0.65) | `geomherd_results.beta_minus_contagion.recall_super` | PENDING | UNMATCHED |
| β⁻ AUROC (0.80) | `geomherd_results.beta_minus_contagion.AUROC` | PENDING | UNMATCHED |
| γ₃ median (−0.0072) | `geomherd_results.CCK_gamma3_median` | PENDING | UNMATCHED |
| Vicsek AUROC (0.99) | `geomherd_results.vicsek_transfer_AUROC` | PENDING | UNMATCHED |
| Lead diff vs Lap-CSAD (+153.8) | Table 2 in SIR | PENDING | UNMATCHED |
| Paired p-value Lap-CSAD (0.03) | Table 2 in SIR | PENDING | UNMATCHED |

**Metrics in paper**: 12  
**Matched**: 0  
**Unmatched (pending)**: 12  

---

## 4. Hallucination Audit Summary

| Type | Count | Critical | Significant | Moderate | Minor |
|------|-------|----------|-------------|----------|-------|
| Structural | 2 | 0 | 1 (H-S1) | 0 | 1 (H-S2) |
| Parametric | 7 | 2 (H-P1, H-P3) | 2 (H-P2, H-P4) | 2 (H-P5, H-P7) | 1 (H-P6) |
| Omission | 2 | 0 | 1 (H-O1) | 0 | 1 (H-O2) |
| **Total** | **11** | **2** | **4** | **2** | **3** |

All critical and significant hallucinations are documented in `hallucination_report.md` with
specific code locations, evidence, and suggested fixes.

---

## 5. Implementation Risk Register

| Risk ID | Description | Severity | Mitigation Status |
|---------|-------------|----------|-------------------|
| R1 | Ricci flow update rule ASSUMED multiplicative | High | Config flag provided (`flow_variant`); test both variants |
| R2 | Kronos head architecture absent — STUB | High | Labelled as stub; structural validation only |
| R3 | LLM persona prompts withheld | Medium | Rule-based fallback; `--llm_mode` flag for LLM |
| R4 | Kendall-τ parameters inferred | Medium | Exposed in config.yaml; calibration sweep recommended |
| R5 | CWS substrate partially reconstructed | Medium | Stylized-facts validation recommended |
| R6 | LP-W1 computational cost | Low | POT library used; batch API available |

---

## 6. Code Quality Checks

| Check | Result |
|-------|--------|
| Python syntax errors | **0** (all 28 modules) |
| Import errors (core modules) | **0** |
| Functional smoke test (AgentGraph, V_eff, CUSUM, Metrics) | **PASS** |
| Dry-run entrypoint | **PASS** — `GeomHerdPipeline(N_agents=66, snapshots=0, op=precision)` |
| Notebook JSON validity | **PASS** — 2 notebooks, 28 cells |
| Config YAML validity | **PASS** — loads with `GeomHerdConfig.from_yaml()` |
| Docker build (not executed) | NOT RUN — docker not available in current environment |
| Unit tests (not yet written) | NOT RUN — test suite not yet generated |

---

## 7. User-Reported Config Modifications

None — no user results submitted.

---

## 8. Reproducibility Score Calculation (Pre-Run Estimate)

```
base_score = N/A (no matched metrics)

SIR confidence penalty:
  mean(sir_confidence_scores) for key components:
    [0.97, 0.95, 0.90, 0.65, 0.65, 0.75, 0.80, 0.45, 0.60]
  mean = 0.75
  sir_confidence_penalty = (1 - 0.75) × 0.15 = 0.0375

Unmatched penalty:
  unmatched_count / total = 12/12 = 1.0
  unmatched_penalty = 1.0 × 0.2 = 0.20

Predicted base range (component analysis):
  6 Excellent + 2 Good + 2 Moderate + 3 Significant + 2 Critical
  base_score_predicted ∈ [0.65, 0.85]

Predicted overall ∈ [0.65 - 0.0375 - 0.00, 0.85 - 0.0375 - 0.00]
  = [0.61, 0.81] for matched-metrics-only score

Full score including unmatched penalty: REQUIRES USER RESULTS
```

---

## 9. Actions Required to Complete This Report

1. **Run full CWS pipeline** (80 seeds × 5 κ values × 2 operating points)
2. **Run Vicsek OOD** (20 seeds × 5 η values)
3. **Run CCK regression** across all supercritical seeds
4. **Submit results** to ArXivist Stage 6 as `results/eval_table.json`
5. ArXivist will automatically:
   - Compute deviations for all 12 metrics
   - Assign deviation severity classifications
   - Update reproducibility score from predicted range to actual value
   - Refine root cause analysis based on observed deviations

---

## 10. Citation

This verification log was produced by the ArXivist automated pipeline.  
SIR version: 1 | Architecture plan version: 1  
Pipeline completed: 2026-05-13  
All six stages executed: Paper Parser → SIR Registry → Architecture Planner →
Code Generator → Notebook Generator → Results Comparator
