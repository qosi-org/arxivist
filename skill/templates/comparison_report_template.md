# Benchmark Comparison Report

**Paper**: {paper_title}
**Paper ID**: {paper_id}
**arXiv**: {arxiv_link}
**Comparison Date**: {comparison_date}
**SIR Version Used**: {sir_version}

---

## Reproducibility Score

| Score | Confidence | Metrics Compared | Matched |
|-------|------------|-----------------|---------|
| **{reproducibility_score}** / 1.0 | {score_confidence} | {metrics_compared} | {metrics_matched} |

**Interpretation**:
- 0.90–1.00: Excellent reproduction
- 0.75–0.89: Good reproduction with minor deviations
- 0.60–0.74: Partial reproduction — review moderate deviations
- 0.40–0.59: Significant reproduction gap — likely implementation issues
- < 0.40: Critical failure — fundamental mismatch

---

## Metric Comparison Table

| Metric | Dataset | Split | Paper Value | Your Value | Deviation | Severity |
|--------|---------|-------|-------------|------------|-----------|----------|
{metric_table_rows}

---

## Deviation Summary

| Severity | Count |
|----------|-------|
| ✅ Excellent (≤2%) | {excellent} |
| 🟢 Good (2–5%) | {good} |
| 🟡 Moderate (5–15%) | {moderate} |
| 🟠 Significant (15–30%) | {significant} |
| 🔴 Critical (>30%) | {critical} |
| ⬜ Unmatched | {unmatched} |

---

## Root Cause Analysis

{For each metric with Moderate/Significant/Critical deviation, one section:}

### {metric_name} on {dataset} — {deviation_pct}% deviation

**Likely causes** (ordered by probability):

1. **{cause_1}** ({probability_1})
   Fix: {fix_1}

2. **{cause_2}** ({probability_2})
   Fix: {fix_2}

---

## Hallucination Report Summary

See `hallucination_report.md` for the full report.

| Type | Count | Critical |
|------|-------|---------|
| Structural | {structural_count} | {structural_critical} |
| Parametric | {parametric_count} | {parametric_critical} |
| Omission | {omission_count} | {omission_critical} |

---

## Recommended Actions

Prioritized by expected impact on reproducibility score:

1. {action_1}
2. {action_2}
3. {action_3}

---

## Implementation Notes

*From the SIR — sections with confidence < 0.7 that may affect these results:*

{list of low-confidence SIR sections with their assumptions}

---

## Verification Log Summary

- Comparison run at: {timestamp}
- User results hash: `{results_hash}`
- User-reported config modifications: {modifications}
- Manual review required: {yes/no} {reason if yes}

Full audit trail in `verification_log.md`.
