# Domain: AI — Evaluation Standards (Stage 6 Enrichment)

Load this file alongside `agents/06_results_comparator.md` when the detected domain is **AI**.

---

## Acceptable deviation thresholds by metric

These thresholds define what counts as "Good" reproduction before escalating to "Moderate".
Generic thresholds are overridden by these domain-specific ones.

| Metric | Excellent | Good | Moderate | Significant | Notes |
|--------|-----------|------|----------|-------------|-------|
| Top-1 accuracy | ≤ 0.3% | ≤ 0.8% | ≤ 2% | ≤ 5% | Hardware and augmentation sensitive |
| Top-5 accuracy | ≤ 0.2% | ≤ 0.5% | ≤ 1.5% | ≤ 3% | |
| Perplexity | ≤ 1% | ≤ 3% | ≤ 8% | ≤ 15% | Tokeniser-dependent — check first |
| BLEU | ≤ 0.5pt | ≤ 1.5pt | ≤ 3pt | ≤ 6pt | Absolute points, not percentage |
| FID | ≤ 2pt | ≤ 5pt | ≤ 15pt | ≤ 30pt | High variance — single run not conclusive |
| IS (Inception Score) | ≤ 0.3 | ≤ 0.8 | ≤ 2.0 | ≤ 4.0 | |
| ROUGE-L | ≤ 0.5% | ≤ 1.5% | ≤ 3% | ≤ 6% | |
| mAP | ≤ 0.3% | ≤ 1% | ≤ 2.5% | ≤ 5% | |

---

## Common root causes in AI papers

When a deviation is classified as Moderate or above, check these in order:

1. **Tokenisation mismatch** — for perplexity and BLEU. Different tokenisers produce incomparable
   perplexity values. Verify the tokeniser is identical (not just the same family — the exact vocab).

2. **Training duration** — AI papers frequently require 100k–1M steps. If the user ran fewer steps,
   deviation is expected. Always check `training_steps_completed` vs paper's reported steps.

3. **Data augmentation stack** — augmentation order and parameters are rarely fully specified.
   Even minor differences (RandAugment magnitude, crop ratio) can cause 1–2% accuracy gaps.

4. **Mixed precision divergence** — fp16 vs bf16 can produce different results on some architectures
   (especially with softmax and layer norm). Check if the user's precision matches the paper.

5. **Weight initialisation** — never stated in most papers. If the paper uses a non-standard
   architecture or the SIR flagged init as assumed, note this as a likely cause.

6. **Evaluation procedure mismatch** — beam search parameters, ensemble vs single model, test-time
   augmentation (TTA). Check the SIR's `evaluation_protocol.special_conditions`.

---

## Hallucination patterns common in AI papers

**Structural hallucinations to check:**
- Extra normalisation layers (e.g. adding LayerNorm where paper uses none)
- Wrong skip connection pattern (additive vs concatenation)
- Extra linear projection added to match dimensions that paper resolves differently
- Attention mask type wrong (padding mask vs causal mask)

**Parametric hallucinations to check:**
- Initialisation std (often assumed — Xavier vs Kaiming vs 0.02)
- Dropout placement (pre vs post attention, before vs after FFN)
- Weight tying (embedding ↔ unembedding) — common in LMs, often assumed

**Omission hallucinations to check:**
- Auxiliary loss terms (common in MoE, detection, segmentation papers)
- EMA of model weights used at evaluation time
- Gradient clipping (if SIR marked it as null but paper uses it implicitly)
