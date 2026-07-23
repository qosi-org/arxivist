# Architecture Plan — LATTICE (arxiv_2607_14410)

## Framework
**PyTorch + PyTorch Geometric**, Python 3.10+, CPU-only (matches the paper's own `device=cpu`
training runs). Plain YAML config (no Hydra/OmegaConf needed — the hyperparameter surface is flat).

## Module Hierarchy
```
src/lattice/
├── models/
│   ├── modality_adapters.py   ModalityInputAdapter, ModalityAwareFusion
│   ├── graph_encoder.py       SpatialGraphBuilder, LatticeGraphEncoder (3x TransformerConv, 4 heads)
│   ├── decoder.py             ReconstructionDecoder (1 hidden layer, width 256)
│   ├── projection_heads.py    ModalityProjectionHead (2-layer MLP, out dim 64)
│   └── lattice_model.py       LatticeModel (wires everything together)
├── data/
│   ├── dataset.py             SyntheticSpatialMultimodalDataset (see "Data" note below)
│   └── graph_utils.py         kNN graph + optional Gaussian edge kernel
├── training/
│   ├── losses.py               masked reconstruction (Huber/MSE), NCE alignment, spatial smoothness
│   └── trainer.py               LatticeTrainer (epoch loop, early stopping, checkpointing)
├── evaluation/
│   └── metrics.py               ARI, NMI, spatial contiguity, silhouette, MUS (Eq. 11)
└── utils/
    └── config.py                 YAML loading + global seeding
```

## Data — important note
The paper's evaluation cohort (11 private melanoma samples, 54,912 Visium spots, 5 modality
blocks) **cannot be publicly released** and has no public substitute at this exact resolution
(SIR ambiguity #6, confidence 0.8). The generated repo therefore ships a **synthetic data
generator** matching the documented shapes (4,992 spots/sample, 5 blocks, D = 5×G with
G ∈ [129, 322]) so the full pipeline — training, evaluation, notebook — runs end-to-end out of
the box. `data/README_data.md` documents exactly how to swap in real data if you have access to
it or an equivalent public cohort.

## Key Equation → Code Mapping
| Paper element | Code location |
|---|---|
| Eq. 1 — concatenated multimodal matrix X | `data/dataset.py` |
| Eq. 2 — spatial kNN neighborhood, k=6 | `data/graph_utils.py::build_knn_graph` |
| Eq. 3 — optional Gaussian edge kernel | `data/graph_utils.py::gaussian_kernel_weights` (off by default, see risk assessment) |
| Eq. 4 — masked input | `training/trainer.py::sample_mask` |
| Eq. 5 — decoder | `models/decoder.py` |
| Eq. 6 — masked reconstruction loss | `training/losses.py::masked_huber_reconstruction_loss` (default) / `masked_mse_reconstruction_loss` (alt., see risk assessment) |
| Eq. 7-8 — cross-modal alignment (NCE) | `training/losses.py::nce_alignment_loss` |
| Eq. 9 — spatial smoothness | `training/losses.py::spatial_smoothness_loss` |
| Eq. 10 — combined objective | `training/trainer.py::LatticeTrainer._compute_loss` |
| Eq. 11 — Multimodal Utility Score (MUS) | `evaluation/metrics.py::multimodal_utility_score` |

## Config Highlights (see `configs/config.yaml` for full, commented file)
- `hidden_dim=128`, 3 TransformerConv layers, 4 heads, dropout 0.1, LayerNorm — all explicit in paper (confidence ≥ 0.9).
- `lambda_rec=1.0, lambda_align=0.5, lambda_spatial=0.1` — explicit (Appendix H).
- `beta1/beta2` for AdamW and the LR schedule are **ASSUMED** (ordinary PyTorch defaults / constant LR) — flagged with `# ASSUMED` comments.
- `recon_loss: huber` — **ASSUMED** resolution of the Eq.6-vs-Appendix-H inconsistency; switchable to `mse`.

## Entrypoints
- `train.py --config configs/config.yaml [--resume PATH] [--seed N] [--debug] [--dry-run]`
- `evaluate.py --checkpoint PATH [--modality-level M1..M5]`
- `inference.py --checkpoint PATH [--sample-index N] [--output PATH]`

## Docker
Base image `pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime` (CPU-only execution still works on
this image); `docker/docker-compose.yml` exposes a `train` service and a `notebook` service on
port 8888.

## Risk Assessment (full list also in `architecture_plan.json`)
| Severity | Risk | Mitigation |
|---|---|---|
| High | No public dataset at this resolution/modality-count | Ship synthetic generator matching documented shapes; document real-data swap-in path |
| Medium | Eq.6 (MSE) vs Appendix H (Huber) inconsistency | Default to Huber per Appendix H; expose both behind `recon_loss` config flag |
| Medium | Alignment pair scope ambiguous (one pair vs. all pairs) | Default to the one stated pair; `nce_alignment_loss` accepts a list of pairs so this is swappable |
| Medium | AdamW betas / LR schedule unstated | Defaulted to PyTorch defaults / constant LR, marked `# ASSUMED` |
| Low | Gaussian-kernel edge weight sigma unspecified | Default to uniform edge weights; kernel implemented but off by default |
| Low | Missing-modality-per-spot handling unspecified | Zero-imputation + presence-mask-weighted pooling |
| Low | SARSIM-derived Leiden K undisclosed | Independent resolution sweep + silhouette-based K selection |
| Low | `leidenalg`/`python-igraph` install friction on some systems | Pinned versions in requirements; conda fallback documented in README |
