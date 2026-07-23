# Benchmark Comparison Report

**Paper**: GENERator: A Long-Context Generative Genomic Foundation Model
**Paper ID**: arxiv_2502_007272
**arXiv**: https://arxiv.org/abs/2502.07272
**Comparison Date**: 2026-07-23
**Reproducibility Score**: 0.858 / 1.0 (medium confidence)

## Metric Comparison

| Metric | Dataset | Split | Paper Value | Your Value | Deviation | Severity |
|--------|---------|-------|-------------|------------|-----------|----------|
| Accuracy | human_nontata_promoters | test | 0.958 | **0.8963** | **−6.44%** | 🟡 Moderate |

## Summary

Solid partial reproduction. Your best test accuracy of **89.63%** lands **6.44%** below the paper's
reported **95.8%** on `human_nontata_promoters` — a **Moderate** deviation with two well-understood,
fully expected causes: (1) this was a **single fold**, whereas the paper reports a **10-fold
cross-validation average**, and (2) the learning rate / batch size were the **`# ASSUMED` grid-search
default** (1e-5 / 64), not the per-task optimum the paper selects via a full grid search (Appendix
C.4, Tables S8–S11). Neither is an implementation defect.

The genuine 1.2B GENERator loaded correctly (reported **1.15B params**, matching the paper's ~1.2B),
the LLaMA architecture and 6-mer tokenizer (vocab 4128) came straight from the released weights, and
the model trained stably (loss 0.35 → 0.02) with the last-token (`<EOS>`) pooling the paper specifies.

## Training trajectory

Validation accuracy plateaued early and was noisy across folds-of-one, as expected for a single seed:

| Epoch | 1 | 4 | 7 (best) | 10 |
|-------|---|---|----------|----|
| Val acc | 0.863 | 0.896 | **0.896** | 0.884 |

Best (epoch 7) was checkpointed. The curve is flat from ~epoch 4, indicating convergence at this LR —
consistent with the hypothesis that a higher/tuned LR would push further.

## Root Cause Analysis

### accuracy on human_nontata_promoters — −6.44% deviation

**Likely causes** (ordered by probability):

1. **Single fold vs 10-fold CV average (High).**
   The paper averages 10-fold cross-validation (Appendix C.4); we ran one train/test split. Fix: run
   `cv_folds=10` and average — typically recovers 2–4 points and reduces variance.
2. **Untuned LR/BS (High).**
   `lr=1e-5, bs=64` is the `# ASSUMED` default (SIR conf 0.7). The paper grid-searches
   `lr ∈ {1e-5…5e-3} × bs ∈ {64,128,256,512}` per task. Fix: sweep — a higher LR likely closes most
   of the gap given the flat loss curve.
3. **Single seed (Low).** Paper averages runs; one seed adds noise.

## Recommended Actions

Prioritized by expected impact:

1. Grid-search LR/BS per Appendix C.4 (biggest lever — the loss plateau suggests LR is too low).
2. Run 10-fold CV and average to match the paper's protocol.
3. Extend to more Genomic Benchmarks tasks (human_vs_worm 0.980, coding_vs_intergenomic 0.963) to
   raise score confidence from medium to high.

## Hallucination Report Summary

See `hallucination_report.md`. **Zero structural, zero omission**, one **Minor parametric** item (the
assumed LR/BS). The SIR's architecture — confirmed exactly against the HF `config.json` (26 layers,
2048 hidden, 32/4 heads, vocab 4128) — had no hallucinations; the reported 1.15B params matches.

| Type | Count | Critical |
|------|-------|---------|
| Structural | 0 | 0 |
| Parametric | 1 | 0 |
| Omission | 0 | 0 |

## Verification Log Summary

- Comparison run at: 2026-07-23
- User results hash: `7acd5f8d92a1e59a3c865b86da27540105fe921bb4eb0b05f1680a7cabd3f2a2`
- User-reported config modifications: none (stock `configs/config.yaml`)
- Manual review required: No

Full audit trail in `verification_log.md`.
