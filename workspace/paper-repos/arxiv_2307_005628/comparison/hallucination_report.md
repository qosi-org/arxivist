# Hallucination Report — DNAGPT (arxiv_2307_005628)

**Comparison Date**: 2026-07-24
**SIR version**: 1
**Scope**: verified architecture (Part A). GSR accuracy pending official-weights run.

## Summary

| Type | Count | Critical | Notes |
|------|-------|----------|-------|
| Structural | 0 | 0 | Dual embeddings/heads, token language, causal GPT, combined loss all match the paper |
| Parametric | 0 | 0 | one open **assumption** (numerical-embed / regression-head MLP dims), disclosed |
| Omission | 0 | 0 | — |

## Structural (0)

Re-derived from the paper and checked mechanically:

- **Backbone** (Sec 2.1, Fig 1c): causal (unidirectional) GPT decoder, 12 layers / 768 /
  12 heads for the 0.1B variant (Fig S3). Built model = **116.3M params** ≈ paper's 0.1B. ✅
- **Dual embeddings** (Fig 1d): Sequential (k-mer) + Numerical (MLP), concatenated. ✅
- **Dual heads** (Fig 1d): Classification (CE over tokens) + Regression (MLP → scalar, MSE). ✅
- **Token language** (Sec 2.2, Fig 2, S2): non-overlap k-mer sequence tokens, number token,
  instruction (species + classification + number), connection (`+`,`=`), classification
  (True/False), reserved. ✅
- **Combined loss** (Eq 1): **L = 0.01·MSE + CrossEntropy**. ✅ (λ = 0.01 hard-checked)
- **k-mer vocab** (S1.3): 5⁶+5⁵+5⁴+5³+5²+5¹ = **19,530** sequence tokens. ✅

## Parametric (0 hallucinations; 1 open assumption)

- **Numerical-embedding / regression-head MLP shapes** are not given in the paper (it says
  "MLPs"). We use a 2-layer GELU MLP (1→D→D and D→D→1), marked `# ASSUMED` in
  `blocks.py`. **Not** a fabricated paper value — and the GSR classification reproduction
  target does not use these heads.
- All specified hyperparameters (AdamW, lr 3e-5 finetune, wd 1e-1, 10 epochs, bs 8, cosine
  +warmup, λ=0.01, flip prob 0.5) are taken verbatim from Sec 3 / Fig S3-S4.

## Omission (0)

The reproduction covers the architecture, token language, all three pre-training-loss
terms, and the GSR classification path (the Table S2 target). Pretraining on ~200B bp is
**intentionally not reproduced** (infeasible) — the repo loads the official weights and
fine-tunes instead, documented in the README, not an omitted claim. The mRNA-regression
and genome-generation tasks are represented in the metrics/heads but not fine-tuned here.

## Notes

- **Deliberate deviation (documented):** the paper overloads `A`/`N` as *both* 1-mer
  sequence tokens *and* True/False classification tokens. In a single flat vocab this
  collides (the 1-mer would overwrite the class token's id). The reproduction names the
  class tokens `<TRUE>`/`<FALSE>` (and instruction tokens `<CLS>`/`<NUMI>`) to keep ids
  distinct — the *role* matches the paper exactly. This is a correctness fix, not a
  hallucination.
- Because official weights are Google-Drive-hosted with the authors' own module names,
  the *downstream* GSR numbers are produced by the user (weights + real data). The
  *architecture* — the paper's actual contribution — is fully verified here on CPU.
