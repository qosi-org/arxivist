# Approximating SPR Distance Between Phylogenetic Trees with Graph Neural Networks

**Paper:** Renata Martins Castanheira, Miguel Bugalho, Cátia Vaz — [arXiv:2607.18311](https://arxiv.org/abs/2607.18311) (17 Jul 2026)
**ArXivist-generated reproduction** — `paper_id: arxiv_2607_18311`

Comparing phylogenetic tree topologies matters for tracking epidemics, but the
biologically meaningful **Subtree Prune and Regraft (SPR) distance** is NP-hard
to compute exactly on trees with thousands of leaves. This repo reproduces the
paper's **Siamese Graph Isomorphism Network (GIN) regressor**, which learns to
approximate a fast polynomial-time heuristic (`phangorn::SPR.dist`) for SPR
distance in a single forward pass, after the paper first validates that
heuristic against the exact rooted distance on small trees (Pearson 0.98–0.99).

## Quick start

```bash
git clone <this-repo-url> && cd arxiv_2607_18311
pip install -r requirements.txt
pip install -e .
python data/download.py            # see data/README_data.md -- dataset is on Zenodo
python train.py --config configs/config.yaml
python evaluate.py --checkpoint outputs/best_model.pt --regime in_distribution
```

Or smoke-test the whole pipeline in seconds with no download, using the
bundled tiny fixture:

```bash
python train.py --config configs/config.yaml --debug --data-dir data/toy
```

## Installation

**pip:**
```bash
pip install -r requirements.txt   # or requirements-dev.txt for lint/test tools
pip install -e .
```

**conda:**
```bash
conda env create -f environment.yaml
conda activate spr_gnn
```

**Docker:**
```bash
docker compose -f docker/docker-compose.yml up train      # training
docker compose -f docker/docker-compose.yml up notebook    # JupyterLab on :8888
```

## Training

```bash
python train.py --config configs/config.yaml --output-dir outputs/
```
Supports `--resume <checkpoint>`, `--seed <int>`, `--debug` (tiny smoke test),
`--dry-run` (build everything, skip the actual loop).

## Evaluation

```bash
python evaluate.py --checkpoint outputs/best_model.pt --regime all
```
Reproduces the in-distribution numbers from Table 2. Cross-species and
size-extrapolation regimes require pre-split master CSVs — see
`data/README_data.md`.

## Inference on your own trees

```bash
python inference.py --tree-a mytree_a.nwk --tree-b mytree_b.nwk --checkpoint outputs/best_model.pt
```

## Expected results (Table 2, Sec 5.3 of the paper)

| Regime | MAE | RMSE | R² | Baseline MAE |
|---|---|---|---|---|
| In-distribution (4 species, mixed sizes) | 127.13 | 202.24 | 0.873 | 485.78 |
| Cross-species (train 2 sp. / test 2 sp.) | 208.70 | 325.72 | 0.368 | 406.05 |
| Size extrapolation (small+med → large) | 375.90 | 643.02 | −0.14 | 553.65 |

Stratified cross-validation (overall): R² = 0.905 ± 0.191, MAE = 92.24 ± 7.02,
RMSE = 128.14 ± 11.07, MAPE = 2.16% ± 0.37%.

Heuristic-vs-exact validation (small trees, 20–90 leaves): Pearson correlation
0.983 (Clostridium), 0.9935 (Vibrio), 0.99 pooled (n=26) — but the heuristic
systematically *underestimates* the exact rooted magnitude by ~29–35%.

## Implementation assumptions (things the paper does not fully specify)

See `sir-registry/arxiv_2607_18311/sir.json → implementation_assumptions[]`
for the complete, confidence-scored list. The most consequential:

| Assumption | Value used | SIR confidence |
|---|---|---|
| GIN layer count | 2 (paper text) — Fig. 5 shows 3 | 0.6 |
| Batch size | 32 | 0.3 (**not stated at all**) |
| Max training epochs | 200 (bounded by early stopping) | 0.4 (**not stated**) |
| Adam β1 / β2 | 0.9 / 0.999 (PyTorch default) | 0.55 |
| Huber loss δ | 1.0 (PyTorch default) | 0.5 (**not stated**) |
| Taxonomic id semantics | per-species (not per-isolate) | 0.55 |

## Reproducibility notes / known deviations

- **GIN layer count is genuinely ambiguous in the source paper** (body text
  says 2, Figure 5's diagram shows 3). `model.num_gin_layers` in
  `configs/config.yaml` is a one-line switch; `GINEncoder` is written
  generically over any layer count. If your reproduced R² doesn't match
  Table 2, try `num_gin_layers: 3` first.
- **Dataset is external** (Zenodo DOI `10.5281/zenodo.20476872`), not bundled
  here — see `data/README_data.md`. A tiny synthetic fixture at `data/toy/`
  lets you sanity-check the pipeline without it.
- **Cross-species and size-extrapolation regimes** (Table 2 rows 2–3) require
  you to pre-partition `master_pairs.csv` by species/size tier yourself;
  this repo does not automate that split since the paper doesn't give exact
  per-pair size cutoffs beyond the small/medium/large tiers (Sec 3).
- **Training compute/wall-clock time is not specified** by the paper for the
  GNN itself (only for the labelling-pipeline stress test, on a 240 GiB RAM /
  Xeon Silver VM) — outside this repo's control to reproduce exactly.
- Batch size, total epochs, and Huber loss δ are unstated in the paper; the
  values above are reasonable defaults, exposed in `configs/config.yaml` for
  tuning.

## Repository structure

```
src/spr_gnn/
├── data/          newick_parser.py, node_features.py, dataset.py
├── models/        embedding.py, gin_encoder.py, siamese_gin.py
├── training/       losses.py, trainer.py
├── evaluation/     metrics.py
└── utils/          config.py
configs/config.yaml
train.py · evaluate.py · inference.py
data/download.py · data/README_data.md · data/toy/ (smoke-test fixture)
notebooks/reproduce_arxiv_2607_18311.ipynb · notebooks/explore_arxiv_2607_18311.ipynb
docker/Dockerfile · docker/docker-compose.yml
checkpoints/ · results/ · comparison/  (populated at runtime / by Stage 6)
```

## Citation

```bibtex
@inproceedings{castanheira2026spr,
  title     = {Approximating {SPR} Distance Between Phylogenetic Trees with Graph Neural Networks},
  author    = {Castanheira, Renata Martins and Bugalho, Miguel and Vaz, C{\\'a}tia},
  year      = {2026},
  eprint    = {2607.18311},
  archivePrefix = {arXiv},
  primaryClass  = {q-bio.PE}
}
```

## Generated by ArXivist

SIR and architecture plan for this reproduction live in
`sir-registry/arxiv_2607_18311/` (`sir.json`, `architecture_plan.json`,
`architecture_plan_summary.md`). Stage 6 (Results Comparator) has not yet
been run — no experiment results have been supplied for this paper. Once you
have run `train.py`/`evaluate.py`, feed the results back to ArXivist to
populate `comparison/`.
