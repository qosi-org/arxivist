# Verification Log

**Paper ID**: arxiv_2607_14410
**Comparison run at**: 2026-07-22T09:00:00Z

---

## Provenance

| Field | Value |
|---|---|
| SIR version used | 1 (`sir-registry/arxiv_2607_14410/sir.json`) |
| Architecture plan version used | 1 (`sir-registry/arxiv_2607_14410/architecture_plan.json`) |
| Paper metrics available (M5, `is_primary=true`) | 5 (ARI, NMI, spatial_contiguity, silhouette, MUS) |
| User results submitted | 30 (5 modality-ladder levels × 6 metrics: ari, nmi, spatial_contiguity, silhouette, bio_knn, bio_jaccard) |
| Matched pairs (numeric comparison) | 4 (ARI, NMI, spatial_contiguity, silhouette) |
| Unmatched | 1 (MUS — different normalization pool, see below) |
| User results hash | `sha256:e0507ee868ee37a3e709b622808364275bd002202397f733555f957792286c84` |

---

## How the "user results" were obtained

The person asking for this comparison had only run the exploratory
demonstration cell in `notebooks/reproduce_arxiv_2607_14410.ipynb` (toy
shape-checks and a 10-step mini-training loop on 100 synthetic spots) — not
`train.py`/`evaluate.py` for a real run, and had no access to the paper's
real (private) cohort. Since a genuine Stage 6 comparison requires actual
computed evaluation metrics, ArXivist ran the pipeline itself in its
sandbox to produce this report, using a scaled-down but otherwise complete
(non-`--debug`) training + evaluation pass:

```
python train.py --config <scaled-down config>
python evaluate.py --config <scaled-down config> --checkpoint checkpoints/best.pt --modality-level {M1..M5}
```

## User-reported / sandbox-applied config modifications

- `data.num_synthetic_samples`: 11 → 4
- `data.spots_per_sample`: 4,992 → 800
- `training.num_epochs`: 100 → 25 (ran to full 25 epochs; no early stop triggered — val loss was still trending down at epoch 24)
- `training.early_stopping_patience`: 20 → 6
- **All other config values kept at their paper-derived defaults** (hidden_dim=128, 3 TransformerConv layers/4 heads, lambda_rec/align/spatial = 1.0/0.5/0.1, masking_ratio=0.15, recon_loss=huber, alignment_temperature=0.1, knn_k=6, seed=42)
- **Dataset is synthetic throughout** — this is the dominant, unavoidable deviation from the paper's setup (no public substitute exists at this modality/resolution combination; see SIR `ambiguities[5]`)

## Metrics compared

ari, nmi, spatial_contiguity, silhouette, mus — against SIR
`evaluation_protocol.reported_results` rows with `dataset="melanoma cohort
(LATTICE, M5)"` and `is_primary=true`.

## Requires manual review: **Yes**

Reasons:
1. Comparison is against synthetic data at a reduced scale, not the paper's real cohort at full scale (unavoidable — the real cohort is private).
2. MUS could not be matched on a like-for-like basis: the paper's MUS values are min-max normalized jointly across 5 baselines + the 5 LATTICE ladder levels evaluated on the real cohort; this repro's MUS values are normalized only across its own 5 synthetic ladder levels. The resulting 0.0–1.0 scales measure "relative rank within a pool," and the two pools are different, so a direct numeric comparison would be misleading. Flagged as `unmatched` rather than assigned a fabricated deviation.
3. Training was run for 25 epochs at reduced scale (4 samples, 800 spots) rather than the paper's up to 100 epochs / 11 samples / 4,992 spots, for sandbox runtime reasons.

## What would upgrade this to a higher-confidence comparison

- Running the **full-scale config** (`configs/config.yaml`, unmodified: 11 samples, 4,992 spots/sample, up to 100 epochs with patience 20) — this alone would remove config-modification as a confound, though the synthetic-vs-real data gap would remain.
- Access to a **real or realistically-simulated** multimodal spatial cohort matching the paper's five modality blocks, so ARI/NMI are being measured against a meaningful reference clustering rather than a synthetic Gaussian-mixture stand-in.
- If/when the original authors release code or the private cohort becomes available under some access arrangement, re-run this comparison against their exact reported per-sample numbers rather than the cohort mean ± std alone.
