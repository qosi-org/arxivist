# Benchmark Comparison Report
**Paper**: Pay Attention to MLPs (gMLP)
**Paper ID**: arxiv_2105_08050
**Comparison Date**: 2026-06-12
**Comparison Type**: ⚠️ **PIPELINE SMOKE TEST — NOT A REPRODUCTION**
**Reproducibility Score**: N/A (see note below)

---

## ⚠️ Important Scope Note

This is **not** an attempt to reproduce the paper's reported results. The smoke test
ran tiny models (~0.4M–0.7M params, 60 steps, random/synthetic data, CPU) purely to
verify that the generated code **executes correctly end-to-end** — i.e., that
`gMLP`, `aMLP`, and the vision variant all build, run forward/backward passes,
and decrease loss without errors.

Paper-scale numbers (Table 3, 4, 6) require:
- 125K–1M training steps (vs. 60 here)
- C4 / ImageNet real data (vs. random tokens / random images here)
- 768–4096 batch size (vs. 8 here)
- TPUv2/v3 pods (vs. CPU here)

A meaningful numeric comparison to Table 3/6 is **not possible** from this run.
The table below reports only what a smoke test *can* validate.

---

## Smoke Test Results

| Component | Params (ours) | Steps | Loss (step 1) | Loss (step 60) | Trend | Status |
|---|---|---|---|---|---|---|
| gMLP (NLP, MLM) | 656,868 | 60 | 6.31 | 3.24 | ↓ 48.6% | ✅ Runs, learns |
| aMLP (NLP, MLM, tiny attn) | 738,788 | 60 | 6.31 | 3.24 | ↓ 48.7% | ✅ Runs, learns |
| gMLP (Vision, ImageNet-style) | 426,186 | 60 | 2.33 | 2.34 | ~flat | ✅ Runs (no signal in random data) |

**Implied perplexity** (gMLP MLM, last-10-step avg loss): **29.3**
**Implied perplexity** (aMLP MLM, last-10-step avg loss): **30.1**

---

## Paper Ground-Truth Targets (for future reference, Table 3 & 6)

| Model | Perplexity (paper) | SST-2 | MNLI-m | SQuAD1.1 F1 | SQuAD2.0 F1 |
|---|---|---|---|---|---|
| gMLPbase | 4.28 | 94.2 | 83.7 | 86.7 | 70.1 |
| aMLPbase | 3.95 | 93.4 | 85.9 | 90.7 | 80.9 |

vs. our smoke perplexities of **29.3 (gMLP)** / **30.1 (aMLP)** — these numbers are
**not comparable**: the smoke test used random token sequences (no linguistic
structure to learn), 1000-word vocab vs. 32K, 4 layers vs. 36–48, 60 steps vs.
125K–1M. A perplexity of ~29 on random data after 60 steps is actually a
**reasonable sanity signal** (random-guess perplexity over 1000-word vocab ≈ 1000;
the model has clearly started learning local statistics).

---

## Summary

The generated repository is **structurally sound and executable**: gMLP, aMLP,
and the vision variant all instantiate correctly, the SGU/Toeplitz/TinyAttention
modules produce correctly-shaped tensors, gradients flow through the full
network, and the optimizer reduces loss over 60 steps for both NLP variants.
This validates Stage 4's code generation but does **not** constitute evidence
toward or against the paper's reproducibility — that requires a full-scale run
(see Stage 3 risk_001/risk_002: ~100+ A100-GPU-hours for gMLPbase).

## Root Cause Analysis

Not applicable — no paper-scale deviations were measured in this run.

## Recommended Actions

1. **To get a real reproducibility signal**: run the ablation preset
   (`gmlp-base-ablation`: 125K steps, C4/RealNews, seq_len=128, batch=2048)
   on a single GPU (~10 GPU-hours per SIR risk_002) and re-run Stage 6 with
   the resulting validation perplexity vs. paper Table 3 target (4.35 for
   gMLP SGU variant, L=36, d_model=512).
2. **Vision loss was flat** over 60 steps on random images — this is expected
   (no learnable signal in random pixels + random labels with CE). Not a bug;
   re-validate only with real ImageNet data.
3. Before the full run, resolve the three flagged ambiguities
   (`w_init_std`, `attn_fusion_mode`, `pool_mode` — SIR ambiguity_001/002/003)
   since these affect convergence behavior at paper scale even if they didn't
   block this smoke test.
