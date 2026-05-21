# Domain: AI — Data Pitfalls (Stage 1 + Stage 4 Enrichment)

---

## Critical data-level reproducibility traps in AI papers

**Test set contamination**
Large language models trained on web-scale data may have seen benchmark test sets during
pretraining. Always note if the paper acknowledges this. For reproduction: use the same
benchmark version the paper used — newer versions of benchmarks (e.g. MMLU variants) may
have different questions.

**Dataset version drift**
ImageNet, COCO, and other major datasets have multiple versions with different class counts,
annotation revisions, and train/val splits. Always extract the exact dataset version from the paper.
If unspecified, add to ambiguities — it can cause 0.5–2% accuracy differences.

**Preprocessing pipeline order**
Augmentation order is not commutative. `RandomCrop → HorizontalFlip` ≠ `HorizontalFlip → RandomCrop`.
Extract the exact order from the paper. If described in natural language without a list, flag
as ambiguous.

**Tokeniser vocabulary mismatch**
BPE vocabularies trained on different corpora produce different tokenisations of the same text.
Perplexity and BLEU are not comparable across tokenisers even of the same type. Always extract
the exact tokeniser name, version, and vocabulary size. Flag if the paper uses a custom vocab.

**Train/val/test split for non-standard datasets**
When papers use their own splits of a public dataset, the exact split is almost never released.
Flag this as a high-confidence data pitfall and note it in the `data/README_data.md`.

**Deduplication**
Pretraining datasets are often deduplicated in ways not described. If the paper uses a custom
pretraining set, flag deduplication methodology as unknown.
