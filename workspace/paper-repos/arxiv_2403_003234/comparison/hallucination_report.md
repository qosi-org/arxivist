# Hallucination Report — Caduceus (arxiv_2403_003234)

**Comparison Date**: 2026-07-24
**SIR version**: 1
**Scope**: verified architecture (Part A). Downstream metrics pending Colab run.

## Summary

| Type | Count | Critical | Notes |
|------|-------|----------|-------|
| Structural | 0 | 0 | RC-equivariance algebra (BiMamba/MambaDNA) matches Appendix A exactly |
| Parametric | 0 | 0 | one open **assumption** (per-task LR), flagged not fabricated |
| Omission | 0 | 0 | — |

## Structural (0)

The two architectural contributions were re-derived from the paper's equations and checked
mechanically:

- **BiMamba** (Sec 3.1): `y = M(x) + flip_L(M(flip_L(x)))` with a single shared `M`. ✅ matches Fig 1.
- **MambaDNA** (Sec 3.2, Eq 8): `concat(M(x_a), RC(M(RC(x_b))))`, channel split, shared operator. ✅
- **RC operation** (Eq 6): `RC(X) = X^{D:1}_{T:1}` (reverse length + channels). ✅ involution verified.
- **RC-equivariant embedding** (Sec 4.1): `concat(Emb(X), RC(Emb(RC(X))))`. ✅ equivariant (Eq 12).
- **RC-equivariant LM head** (Sec 4.1): `LM(x_a) + flip_chan(LM(x_b))`. ✅ shape/contract verified.

**Theorem 3.1** (`RC∘MambaDNA = MambaDNA∘RC`) holds with **max-abs gap 0.0** on random inputs.

## Parametric (0 hallucinations; 1 open assumption)

- **Per-task learning rate** defaults to `1e-3`, marked `# ASSUMED` in `configs/config.yaml`. The paper
  grid-searches `{1e-3, 2e-3}` per task (Table 5); a single reproduction run fixes one point. This is a
  disclosed assumption (SIR conf 0.7), **not** a fabricated value.
- All other hyperparameters (AdamW, cosine, mean-pool head, 5-fold CV, RC aug + conjoining, MLM 15%
  masking) are taken verbatim from Sec 5 / Appendix C-D.

## Omission (0)

The reproduction covers both equivariance modes (PS parameter-sharing / Ph post-hoc conjoining), RC
data augmentation, and the mean-pool classification head. Pretraining (35B-token MLM on HG38) is
**intentionally not reproduced** — infeasible — and the repo loads the official released weights
instead; this is documented in the README and `data/README_data.md`, not an omission of a claim.

## Notes

Because the official forward pass depends on the Mamba CUDA kernels, the *downstream* numbers are
produced on GPU (Colab) by the user. The *architecture*, which is the paper's actual contribution, is
fully verified here on CPU with zero hallucinations.
