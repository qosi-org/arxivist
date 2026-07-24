# Hallucination Report — arxiv_2607_18311

**Comparison date**: 2026-07-22
**SIR version**: 1 | **Architecture plan version**: 1

This report audits the generated implementation (`paper-repos/arxiv_2607_18311/`)
against the SIR (`sir.json`) for three categories of deviation: components invented
that aren't in the paper, unstated hyperparameters that were guessed, and paper
components that are missing or stubbed out.

---

## Structural Hallucinations

**None found.**

Every class in `src/spr_gnn/models/` and `src/spr_gnn/data/` traces directly to a named
SIR `architecture.modules[]` entry:

| Code component | SIR module |
|---|---|
| `NewickToGraph` | `NewickParser` |
| `NodeFeatureExtractor` | `NodeFeatureExtractor` |
| `TaxonomicEmbedding` | `TaxonomicEmbeddingLayer` |
| `GINEncoder` (layers 1..N) | `GINLayer_1`, `GINLayer_2` |
| `global_add_pool` call inside `GINEncoder.forward` | `GlobalAddPooling` |
| `torch.cat([h_a, h_b])` in `SiameseGINRegressor.forward` | `PairConcatenation` |
| `mlp_head` in `SiameseGINRegressor` | `MLPRegressionHead` |

`Trainer`, `Config`, and `RegressionMetrics` are training/eval infrastructure required
to run the model at all — they don't claim to encode any paper-specific mathematical
content beyond what's cited from Sec 4.3, and are not counted as "architecture"
components.

---

## Parametric Hallucinations

Six hyperparameters were set to assumed values because the paper does not state them.
None are rated Critical because — critically — **no real training run has occurred yet**,
so there is no confirmed evidence any of these are actually wrong; they are pre-identified
risk factors from Stage 3's architecture plan, carried forward here for visibility.

| Hyperparameter | Assumed value | Severity | Evidence | Suggested fix |
|---|---|---|---|---|
| `model.num_gin_layers` | 2 | **Significant** | SIR `ambiguities[0]`, confidence 0.6 — paper text says 2 layers, Fig. 5 diagrams 3 | Config-swappable; run both 2 and 3 against real data, compare R² to Table 2 (0.873) |
| `training.batch_size` | 32 | Minor | Not stated at all (confidence 0.3) | Tune if convergence looks poor on real data |
| `training.max_epochs` | 200 | Minor | Not stated; bounded by early-stopping patience (confidence 0.4) | Increase if not converged by epoch 200 |
| `training.huber_delta` | 1.0 | Minor | Not stated; PyTorch default used (confidence 0.55). SPR distances range ~0-2000 | Consider scaling up (e.g. 10-50) if loss looks poorly calibrated to the target range |
| `training.adam_beta1/2` | 0.9 / 0.999 | Minor | Not stated at all (confidence 0.55) | Low expected impact |
| `evaluation.cross_validation.n_folds` | 5 | Minor | Paper reports CV mean±std but never states fold count | Adjust if authors' code confirms a different value |

`model.num_gin_layers` is flagged **Significant** rather than Minor because it changes the
model's receptive field and parameter count directly, and is the one assumption most
likely to visibly move R² once real training is attempted — worth resolving empirically
before drawing conclusions from any future comparison.

---

## Omission Hallucinations

| Missing component | SIR/paper location | Severity | Suggested fix |
|---|---|---|---|
| Full dataset-construction pre-processing pipeline: cgMLST profile slicing/stratification (small/medium/large tiers), controlled seeded shuffling, UPGMA/NJ tree inference via PhyloLib, midpoint re-rooting, and phangorn/rspr SPR labelling | Paper Sec 3, Fig. 1-2 (outside `sir.json`'s `architecture` section, which covers only the GIN model in Sec 4) | Significant | This was a **deliberate, documented scope decision** (see `architecture_plan.json → risk_assessment`, "Medium severity: dataset lives on external Zenodo DOI"), not an accidental gap. This repo consumes the already-labelled Zenodo release rather than rebuilding it from raw cgMLST profiles. Full end-to-end reproduction (including dataset construction) would need a separate R-based module (PhyloLib, phangorn, rspr, Bio.Phylo) not implemented here. |

No other omissions found — all nine `architecture.modules[]` entries from the SIR have a
corresponding, exercised code path (confirmed via the executed
`notebooks/reproduce_arxiv_2607_18311.ipynb` and the `train.py`/`evaluate.py`/`inference.py`
smoke tests).

---

## Summary

| Type | Count | Critical | Significant | Minor |
|---|---|---|---|---|
| Structural | 0 | 0 | 0 | 0 |
| Parametric | 6 | 0 | 1 | 5 |
| Omission | 1 | 0 | 1 | 0 |

The implementation is a faithful, complete mapping of the SIR's modeled architecture
(Sec 4). The two Significant items are both **known, documented, and already surfaced**
to you elsewhere (architecture plan risk assessment, README, config comments) — this
audit did not find anything new or hidden. The main open question remains empirical:
whether `num_gin_layers=2` or `3` matches the paper's reported R² once trained on real data.
