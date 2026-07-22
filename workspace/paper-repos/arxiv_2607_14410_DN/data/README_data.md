# Data — README

## Short version

The paper's evaluation cohort is **private and cannot be publicly released**:

> "The raw data cannot be publicly released because of institutional data-use and
> privacy restrictions." — Section 4.1
>
> "The cohort tensors are de-identified clinical biospecimen-derived profiles
> under a collaborator institution's proprietary agreement, cannot be
> redistributed publicly, and have no public five-modality substitute at this
> lattice resolution." — Appendix G.1

There is currently no public dataset combining Visium RNA + scMultiome RNA +
scMultiome ATAC gene scores + spatial ATAC + spatial CUT&Tag at this same
per-spot resolution. As a result, **this repository trains and evaluates on a
synthetic dataset by default** (`src/lattice/data/dataset.py`,
`SyntheticSpatialMultimodalDataset`), which matches the paper's documented
shapes:

| Quantity | Paper (Table 1 / Appendix H) | Synthetic default |
|---|---|---|
| Samples (slides) | 11 | 11 (`data.num_synthetic_samples`) |
| Spots per slide | 4,992 | 4,992 (`data.spots_per_sample`) |
| Modality blocks | 5 | 5 |
| Genes per block (G) | 129–322 | 129–322 (`data.gene_count_range`) |
| Total feature dim D | 5×G | 5×G |

**Results produced with the synthetic dataset will not numerically match
Table 2 / Table 3 in the paper.** They exist so the full pipeline
(training → evaluation → notebook) is runnable end-to-end without access to
the private cohort.

## If you have access to an equivalent multimodal spatial cohort

To plug in real data, implement a `Dataset` with the same interface as
`SyntheticSpatialMultimodalDataset.__getitem__`:

```python
{
    "modality_blocks": [Tensor[N, G], ...] ,  # 5 tensors, one per modality block
    "presence_mask":  Tensor[N, 5],           # 1 = modality observed at this spot
    "coords":          Tensor[N, 2],           # spatial (array) coordinates
    "domain_labels":   Tensor[N],               # optional reference clustering
                                                  # (e.g. Space Ranger RNA clusters)
                                                  # for ARI/NMI evaluation
}
```

Point `train.py` / `evaluate.py` at your dataset class instead of
`SyntheticSpatialMultimodalDataset` (a `--dataset-module` style CLI hook can
be added trivially if you need to switch datasets without editing code).

Modality block order in the paper (Appendix H) is:
1. Visium RNA
2. Projected scMultiome RNA
3. Projected scMultiome ATAC gene scores
4. Spatial ATAC (CGMC prediction)
5. Spatial CUT&Tag (CGMC prediction)

Harmonization in the paper (ReCAST + SARSIM, Appendix F) uses a strict
five-way gene intersection and `log1p_then_zscore` per-modality
standardization before concatenation — replicate this preprocessing on your
own data before feeding it to the model for the closest match to the paper's
setup.

## Verifying downloads

`download.py` in this directory checks for a pre-existing `data/raw/`
directory and, since no public download exists, prints these same
instructions instead of fetching anything.
