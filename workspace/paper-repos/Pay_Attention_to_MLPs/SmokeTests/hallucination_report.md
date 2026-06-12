# Hallucination Report
**Paper ID**: arxiv_2105_08050 | **Date**: 2026-06-12
**Scope**: Audit of generated code (Stage 4) against SIR (Stage 1/2) and architecture plan (Stage 3)

---

## Structural Hallucinations
*(Components in generated code NOT present in the SIR)*

**None found.** All implemented modules trace directly to SIR `architecture.modules`:
- `ToeplitzLinear` → SIR `toeplitz_constraint` + Eq.6
- `TinyAttention` → SIR `architecture.modules[2]` + Eq.7
- `SpatialGatingUnit` / `aMLP_SGU` → SIR `architecture.modules[1]/[3]` + Eq.3/4
- `gMLPBlock` (+ `DropPath`) → SIR `architecture.modules[0]` + Eq.1; stochastic depth from `training_pipeline.vision_pretraining.stochastic_depth`
- `PatchEmbedding` → SIR `architecture.input_protocols.Vision`

Two additions not explicitly itemized in SIR but standard/necessary, flagged as minor:
- `gMLP._init_weights()` trunc_normal(std=0.02) for Linear/Embedding — standard Transformer init, not paper-specified but necessary and doesn't override ToeplitzLinear's own init (verified in code comment).
- `MixupCutmixCollator` batch-level augmentation — directly implements SIR `training_pipeline.vision_pretraining.augmentation`, just organized as a separate collator class rather than inline.

**Severity**: Minor (both are standard, documented, non-paper-contradicting choices)

---

## Parametric Hallucinations
*(Assumed hyperparameters — flagged `# ASSUMED` — that could be wrong)*

| # | Parameter | Assumed Value | SIR Confidence | Status after smoke test |
|---|-----------|---------------|-----------------|--------------------------|
| 1 | `w_init_std` | 0.002 | 0.65 (ambiguity_002) | **Untested at scale.** Smoke test used 4-layer model with this value — no divergence in 60 steps, but this is far too short to validate stability claims paper makes about near-zero init mattering "at the beginning of training." |
| 2 | `attn_fusion_mode` | 'add' | 0.75 (ambiguity_001) | **Untested for correctness.** aMLP smoke test ran without numerical issues (loss curve nearly identical to gMLP — expected at random init since attn_gate starts near 0 contribution). This does NOT confirm 'add' is the paper's intended fusion; only confirms it doesn't crash. |
| 3 | `pool_mode` | 'avg' | 0.70 (ambiguity_003) | **Untested.** Vision smoke test used pool_mode='avg' successfully (forward/backward OK), but with random labels/images there's no signal to distinguish 'avg' vs 'cls' correctness — only shape correctness was verified. |

**Severity assessment**: All three remain **Significant** open items for full-scale
reproduction — none were *resolved* by the smoke test, only shown to be
non-crashing. This is expected; a 60-step random-data run cannot adjudicate
these design choices. Re-flagging for visibility before any paper-scale run.

---

## Omission Hallucinations
*(Components present in SIR but absent/stubbed in generated code)*

| # | SIR Component | Status | Notes |
|---|---------------|--------|-------|
| 1 | `SQuADDataset` sliding-window inference | Present but simplified | `glue_dataset.py` SQuADDataset uses only the first tokenizer window (`enc["input_ids"][0]`) rather than full sliding-window aggregation across overflow tokens. SIR `evaluation_protocol.nlp` expects full-context SQuAD evaluation. **Not exercised by this smoke test** (NLP smoke used synthetic MLM data only, not SQuAD). |
| 2 | Tokenizer (32K cased SentencePiece) | Proxy substitution | `t5-base` tokenizer used as proxy (SIR `assume_007`/`risk_007`). Smoke test used a synthetic 1000-word vocab and bypassed the tokenizer entirely, so this substitution was **not exercised**. |
| 3 | Toeplitz Appendix-C exact TF translation | Implemented via index-gather, not pad/tile/reshape | `ToeplitzLinear._build_toeplitz()` uses `row_idx - col_idx + (n-1)` gather indexing rather than literally replicating the TF `pad→tile→reshape` sequence from Appendix C. Mathematically equivalent (verified by `test_toeplitz_matrix_is_toeplitz`), so this is **not a hallucination** but noted for traceability. |

**Severity**:
- #1 (SQuAD windowing): **Moderate** — will affect SQuAD F1 if/when finetuning is run; needs proper sliding-window aggregation before SQuAD numbers can be trusted.
- #2 (tokenizer proxy): **Significant** at paper scale (affects perplexity comparability to Table 3/6 directly), but **not applicable** to this smoke test.
- #3 (Toeplitz construction): **None** — equivalent implementation, covered by unit test.

---

## Smoke-Test-Specific Findings

No new hallucinations were *discovered* by this run — its purpose was execution
validation, and all three model variants (gMLP-NLP, aMLP-NLP, gMLP-Vision)
executed without shape errors, NaNs, or crashes across 60 steps each. This
corroborates (but does not newly prove) the unit test suite's 68/68 pass result
from Stage 4.

---

## Suggested Fixes (Priority Order)

1. **Before any finetuning run on SQuAD**: implement proper sliding-window
   span aggregation in `SQuADDataset` (omission #1).
2. **Before any full pretraining run**: pick a tokenizer strategy — either
   accept the t5-base proxy explicitly (document expected perplexity offset
   vs. paper's 32K cased SentencePiece) or train a custom SentencePiece model
   on C4 to match paper vocab exactly (parametric/omission #2).
3. **During the first 1–5K steps of a real ablation run**: monitor SGU gate
   norms (`spatial_proj.weight.std()`) to validate `w_init_std=0.002` doesn't
   cause early instability or vanishing gradients (parametric #1).
4. **After ~10K steps of ablation training**: compare gMLP vs aMLP perplexity
   curves — if aMLP shows no improvement over gMLP on MNLI-style tasks, revisit
   `attn_fusion_mode` (parametric #2), trying `'concat'` as alternative.
