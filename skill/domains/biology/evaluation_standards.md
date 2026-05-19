# Domain: Biology — Evaluation Standards (Stage 6 Enrichment)

## Deviation thresholds

| Metric | Excellent | Good | Moderate | Notes |
|--------|-----------|------|----------|-------|
| TM-score | ≤ 0.01 | ≤ 0.03 | ≤ 0.07 | Absolute (0–1) |
| RMSD (Å) | ≤ 0.1Å | ≤ 0.3Å | ≤ 0.8Å | Absolute |
| AUC-ROC | ≤ 0.005 | ≤ 0.015 | ≤ 0.04 | |
| AUPRC | ≤ 0.01 | ≤ 0.03 | ≤ 0.07 | Preferred for imbalanced |
| Pearson r | ≤ 0.01 | ≤ 0.03 | ≤ 0.07 | |
| ARI | ≤ 0.02 | ≤ 0.05 | ≤ 0.12 | Clustering metric |

## Root causes specific to biology

1. **Wrong split strategy** — random vs sequence-identity-based split is the biggest
   source of over-optimistic results in biology ML. If deviation is suspiciously small
   (better than paper), check the split strategy first.
2. **Pre-trained model version** — ESM, AlphaFold, and other foundation models have
   multiple versions. Different versions produce different embeddings.
3. **Preprocessing order** — normalise-then-log vs log-then-normalise for count data
   produces different results. Check preprocessing order against paper.
4. **Database version** — UniProt, PDB, STRING are updated regularly. Different versions
   have different protein counts and annotations.
