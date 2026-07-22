# Data — arXiv:2607.18311

The paper's dataset is publicly released on Zenodo:

> Castanheira, R., Bugalho, M., Vaz, C. "Phylogenetic trees inferred with
> different algorithms dataset and their SPR comparison." Zenodo (2026).
> **DOI: 10.5281/zenodo.20476872**

It is **not bundled in this repository** (external hosting, per the
Architecture Planner's risk assessment — Medium severity).

## How to obtain it

```bash
python data/download.py
```

This prints the Zenodo record URL and, once you've placed the downloaded
archive at `data/spr_dataset.zip`, extracts it in place. Re-run the script
after downloading.

## Expected structure after extraction

```
data/
├── master_pairs.csv       # pair_id, tree_a_path, tree_b_path, species, spr_label, [split]
└── trees/
    ├── clostridium/*.nwk
    ├── salmonella/*.nwk
    ├── vibrio/*.nwk
    └── streptococcus_pneumoniae/*.nwk
```

`master_pairs.csv` columns, per Sec 3 ("Pairing and labelling") of the paper:

| column | description |
|---|---|
| `pair_id` | unique id for the tree pair |
| `tree_a_path`, `tree_b_path` | paths to the two Newick files being compared |
| `species` | one of the four bacterial species |
| `spr_label` | `phangorn::SPR.dist` heuristic value (the paper's supervision target — **not** the exact rooted `rspr` distance) |
| `split` | optional; if absent, `TreePairDataModule.setup()` creates a 70/15/15 train/val/test split under seed 42 |

The 388 pairs follow the paper's fixed niche distribution (Fig. 2b):
UPGMA vs. NJ (40%), NJ vs. NJ_shuffled (25%), UPGMA vs. UPGMA_shuffled (15%),
NJ vs. NJ / UPGMA vs. UPGMA zero-distance controls (10% each).

## Cross-species / size-extrapolation regimes (Table 2, rows 2–3)

To reproduce the **cross-species** regime (train on Clostridium+Vibrio,
test on Salmonella+S. pneumoniae) or the **size-extrapolation** regime
(train on small+medium trees, test on large trees), pre-filter
`master_pairs.csv` into separate CSVs and point `data.master_pairs_csv` in
a copy of `configs/config.yaml` at each. This repository does not automate
that re-partitioning — it is a deliberate scope boundary from Stage 3
(Architecture Planner), since the exact size/species cutoffs used by the
authors beyond "small/medium/large" tiers (Sec 3) are not fully specified
down to the pair level.

## Smoke-testing without the full dataset

`data/toy/` (see below) contains a hand-written, tiny synthetic
`master_pairs.csv` + a handful of small `.nwk` files so you can run
`python train.py --config configs/config.yaml --debug --data-dir data/toy`
to sanity-check the pipeline without downloading anything.
