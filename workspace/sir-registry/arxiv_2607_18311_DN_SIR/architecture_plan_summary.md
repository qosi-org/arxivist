# Architecture Plan Summary — arxiv_2607_18311

**Paper:** Approximating SPR Distance Between Phylogenetic Trees with Graph Neural Networks
**Framework:** PyTorch + PyTorch Geometric (explicit in paper), YAML config, no CUDA hard requirement.

## Model: Siamese GIN Regressor
```
Newick tree ─▶ NewickToGraph ─▶ NodeFeatureExtractor ──┐
                                                        ▼
                                        concat([N,3] continuous, [N,16] species-embed) = [N,19]
                                                        │
                                          GINEncoder (shared weights, 2 layers* × 128-d)
                                                        │
                                          global_add_pool ─▶ [128] per tree
                        (same encoder applied to tree_a and tree_b)
tree_a [128] ──┐
               ├─ concat ─▶ [256] ─▶ Linear(256,128)-ReLU-Drop(.3)-Linear(128,64)-ReLU-Drop(.3)-Linear(64,1) ─▶ clamp(≥0) ─▶ predicted SPR
tree_b [128] ──┘
```
`*` 2 layers per the paper's body text; Figure 5 shows 3 — kept as a one-line config flag (`num_gin_layers`) rather than hardcoded, since the SIR confidence here is only 0.6.

## Repo layout
```
src/spr_gnn/
├── data/          newick_parser.py, node_features.py, dataset.py
├── models/        embedding.py, gin_encoder.py, siamese_gin.py
├── training/       losses.py (Huber), trainer.py (Adam, ReduceLROnPlateau, early stop)
├── evaluation/     metrics.py (MAE/RMSE/MAPE/R²)
└── utils/          config.py
train.py · evaluate.py · inference.py
configs/config.yaml · requirements.txt · environment.yaml · Dockerfile
```

## Key config decisions (with SIR confidence)
| Setting | Value | SIR confidence | Note |
|---|---|---|---|
| GIN layers | 2 | 0.6 | text vs. Fig.5 conflict — swappable |
| Hidden dim | 128 | 0.9 | explicit |
| Optimizer | Adam, lr=1e-4, wd=1e-4 | 0.75 | explicit lr/wd; betas assumed |
| LR schedule | ReduceLROnPlateau, patience=10, factor=0.5 | 0.85 | explicit |
| Early stopping | patience=25 | 0.85 | explicit |
| Batch size | 32 (assumed) | 0.3 | **not stated in paper** |
| Max epochs | 200 (assumed, bounded by early stopping) | 0.4 | **not stated in paper** |
| Huber delta | 1.0 (assumed, library default) | 0.5 | **not stated in paper** |
| Train/val/test split | 70/15/15, seed 42 | 0.9 | explicit |
| Dropout | 0.3 | 0.9 | explicit |

## Risks (full detail in `architecture_plan.json.risk_assessment`)
- **Medium** — GIN layer-count ambiguity (2 vs. 3)
- **Medium** — batch size / epochs / Adam betas unstated
- **Medium** — dataset lives on external Zenodo DOI, not bundled
- **Low** — Huber delta unstated
- **Low** — training hardware/wall-clock unstated
- **Low** — taxonomic-id semantics (species vs. isolate) inferred

## Entrypoints
- `train.py --config configs/config.yaml`
- `evaluate.py --checkpoint <path> --regime all` (reproduces Table 2: in-distribution / cross-species / size-extrapolation)
- `inference.py --tree-a a.nwk --tree-b b.nwk --checkpoint <path>`
