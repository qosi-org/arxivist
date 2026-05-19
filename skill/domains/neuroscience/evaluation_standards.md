# Domain: Neuroscience — Evaluation Standards (Stage 6 Enrichment)

Load alongside `agents/06_results_comparator.md` when domain is **Neuroscience**.

---

## Deviation thresholds for neuroscience metrics

Neuroscience datasets are small, noisy, and subject-variable. Thresholds are wider than
other domains to account for this inherent variance.

| Metric | Excellent | Good | Moderate | Notes |
|--------|-----------|------|----------|-------|
| Decoding accuracy | ≤ 1% | ≤ 3% | ≤ 7% | Cross-validated |
| R² (continuous decoding) | ≤ 0.02 | ≤ 0.05 | ≤ 0.10 | Absolute |
| AUC-ROC | ≤ 0.01 | ≤ 0.03 | ≤ 0.07 | |
| Bits per spike | ≤ 3% | ≤ 8% | ≤ 18% | Relative |
| Correlation (r) | ≤ 0.02 | ≤ 0.05 | ≤ 0.10 | Absolute |

**Always compare against the chance level reported in the paper, not just the raw metric.**
A 60% accuracy with 50% chance is a 10% above-chance effect; a 70% accuracy with 65%
chance is only 5% above chance. The above-chance performance is what matters.

---

## Root causes specific to neuroscience

1. **Subject variability** — neuroscience results vary dramatically across subjects.
   A paper reporting N=8 subjects has very high variance in group-level metrics.
   If the user tested on different subjects, deviation is expected and not a failure.

2. **Preprocessing parameter sensitivity** — EEG/MEG results are highly sensitive to
   filtering parameters, epoch rejection thresholds, and artefact removal methods.
   A different ICA decomposition alone can shift decoding accuracy by 3–5%.

3. **Cross-validation scheme mismatch** — leave-one-trial-out vs leave-one-session-out
   vs leave-one-subject-out produce very different accuracy estimates. Check the CV
   scheme matches the paper exactly.

4. **SNN surrogate gradient function** — different surrogate gradient functions
   (straight-through, sigmoid, fast sigmoid, arctangent) produce different training
   dynamics and final accuracy. Check which the paper used.

5. **Epoch rejection criteria** — rejecting different numbers of artefact-contaminated
   epochs changes the dataset and therefore the results. If the rejection threshold
   is not specified, flag as a high-impact ambiguity.

6. **Small N statistics** — with N < 10 subjects, results are not statistically stable.
   Deviation within ± 1 standard error of the paper's mean should always be Excellent.
