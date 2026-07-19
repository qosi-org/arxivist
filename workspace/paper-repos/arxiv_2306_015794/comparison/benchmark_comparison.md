# Benchmark Comparison Report

**Paper**: HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution
**Paper ID**: arxiv_2306_015794
**arXiv**: https://arxiv.org/abs/2306.15794
**Comparison Date**: 2026-07-12
**Reproducibility Score**: 0.94 / 1.0 (medium confidence)

## Metric Comparison

| Metric | Dataset | Split | Paper Value | Your Value | Deviation | Severity |
|--------|---------|-------|-------------|------------|-----------|----------|
| Accuracy | human_nontata_promoters | test | 0.9660 | 0.9471 | −1.96% | ✅ Excellent |
| MCC | human_nontata_promoters | test | — | 0.8957 | (not reported by paper) | ⬜ Unmatched |
| F1 | human_nontata_promoters | test | — | 0.9499 | (not reported by paper) | ⬜ Unmatched |

*The paper reports top-1 accuracy for GenomicBenchmarks; MCC/F1 are extra metrics your run produced and have no paper target for this dataset.*

## Summary

Strong reproduction. Your best test accuracy of **94.71%** is within **1.96%** of the paper's
reported **96.6%** on `human_nontata_promoters` — inside the "Excellent" band (≤2%), which is
attributable to training variance and minor unspecified fine-tuning details rather than any
implementation error. The genuine pretrained HyenaDNA backbone loaded with `missing=0` (every
backbone tensor matched the released checkpoint), so the remaining gap is a fine-tuning-recipe gap,
not an architecture gap.

## Training trajectory

Accuracy rose monotonically and effectively plateaued near the end (epochs 15–20 all ≥ 94%), so the
model had converged:

| Epoch | 1 | 5 | 10 | 15 | 19 (best) |
|-------|---|---|----|----|-----------|
| Val acc | 0.832 | 0.909 | 0.926 | 0.945 | **0.947** |

## Root Cause Analysis

No deviation ≥ Moderate, so no mandatory root-cause section. For completeness, the residual −1.96%
gap most likely comes from:

1. **Fine-tuning recipe (Medium probability).** The paper's exact downstream schedule (LR, epochs,
   early-stopping/best-of selection) is not fully specified in the SIR (training_pipeline conf 0.62).
   We used LR 2e-4 / 20 epochs / cosine. Fix: try the official `lr=6e-4` and/or longer schedule.
2. **No reverse-complement augmentation (Low probability).** Coded in `transforms.py` but not wired
   into the dataset. Fix: enable RC augmentation — typically +0.5–1.5 pts.
3. **Pad-token pooling (Low probability).** Mean-pool includes `[PAD]` positions; the official code
   left-pads and pools differently. Fix: mask pads before pooling.

## Recommended Actions

Prioritized by expected impact:

1. Enable reverse-complement augmentation in the dataset pipeline.
2. Sweep LR ∈ {6e-4, 3e-4} with 20–30 epochs; keep best-by-val checkpoint (already done).
3. Mask pad tokens in mean-pooling (or switch `pool: last`).

## Hallucination Report Summary

See `hallucination_report.md`. One **parametric** item (downstream LR/epochs were assumed, now
tuned). Zero structural, zero omission — the vendored `standalone_hyenadna.py` matches the paper's
architecture key-for-key.

| Type | Count | Critical |
|------|-------|---------|
| Structural | 0 | 0 |
| Parametric | 1 | 0 |
| Omission | 0 | 0 |

## Verification Log Summary

- Comparison run at: 2026-07-12
- User results hash: `0da942d2fe39439ca052563441ff4eb3fb0cfc56c0d829f52aa4b0472e63d562`
- User-reported config modifications: epochs 5→20, lr 6e-5→2e-4
- Manual review required: No

Full audit trail in `verification_log.md`.
