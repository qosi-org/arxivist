# Architecture Plan — NeRF (arXiv:2003.08934)

## Framework
**TensorFlow 2.15 (eager + `tf.GradientTape`)**, Python 3.10+, GPU (CUDA 11.2+) required for full training; CPU-viable for the "tiny" consumer preset.
The paper explicitly states its reference implementation is TensorFlow (Appendix A), so this plan follows that rather than defaulting to PyTorch. Config management is plain YAML (the hyperparameter surface is flat, no need for Hydra).

## Module Hierarchy
```
src/nerf/
├── models/
│   ├── positional_encoding.py   → PositionalEncoding (γ(x), γ(d))
│   ├── nerf_mlp.py               → NeRFMLP (8-layer trunk + skip + density/color heads)
│   ├── radiance_field.py         → NeRFModel (coarse + fine NeRFMLP pair, query + render_rays)
│   └── volume_rendering.py       → VolumeRenderer (Eq.1-3), HierarchicalSampler (Eq.5)
├── data/
│   ├── rays.py                   → RayGenerator (ray gen, stratified sampling, NDC transform)
│   └── dataset.py                → BlenderSyntheticDataset, LLFFRealDataset
├── training/
│   ├── losses.py                 → PhotometricMSELoss (Eq.6)
│   └── trainer.py                → Trainer (train_step, fit, checkpointing)
├── evaluation/
│   └── metrics.py                → ImageMetrics (PSNR, SSIM, LPIPS)
└── utils/
    ├── config.py                 → NeRFConfig (YAML loader)
    └── tiny_variant.py           → TinyNeRFPreset (consumer-hardware config override)
```
Every SIR architecture module (9/9) is mapped to a concrete file/class above — none omitted.

## Key Tensor Flows
1. **NeRFMLP forward**: `γ(x) [N,60]` → 8×Dense(256, ReLU) w/ skip-concat at layer 5 → `σ [N,1]` (ReLU) + `feature [N,256]`; `feature ⊕ γ(d) [N,280]` → Dense(128, ReLU) → Dense(3, sigmoid) → `rgb [N,3]`.
2. **Two-pass render**: stratified 64 coarse samples/ray → coarse MLP → composite → weights → inverse-CDF resample 128 fine points → union of 192 points → fine MLP → composite → final `rgb_fine [N_rays,3]`.
3. **Training step**: `GradientTape` over `‖rgb_coarse−target‖² + ‖rgb_fine−target‖²` summed over a 4096-ray batch → Adam (lr 5e-4→5e-5 exponential decay) update.

## Config Highlights (full schema in `config_schema` / `config.yaml`)
- Model: `pos_enc_freqs_x=10`, `pos_enc_freqs_d=4`, `trunk_depth=8`, `trunk_width=256`, `skip_layers=[4]`, `Nc=64`, `Nf=128`
- Training: Adam, lr 5e-4→5e-5 exponential decay, batch=4096 rays, **`num_training_steps=250000` — ASSUMED (paper states 100k-300k range, no single number given)**
- Data: `near/far` and `use_ndc` are dataset-dependent (bounding cube for synthetic, NDC for real forward-facing)

## Dependencies
Runtime: `tensorflow 2.15`, `numpy`, `imageio(+ffmpeg)`, `opencv-python-headless`, `configargparse`, `scikit-image` (SSIM), `lpips-tf2` (LPIPS, with a documented PyTorch-subprocess fallback), `tqdm`, `PyYAML`.
Dev: `pytest`, `pytest-cov`, `black`, `ruff`, `mypy`.

## Entrypoints
- `train.py --config --datadir --expname [--num_steps] [--tiny] [--resume]`
- `evaluate.py --config --checkpoint --datadir [--out_dir]`
- `inference.py --checkpoint [--pose] [--out]`
- `preprocess_llff.py --scenedir [--factor]` (COLMAP-based, paper-specific for the "Real Forward-Facing" dataset)

## Docker
Base image `tensorflow/tensorflow:2.15.0-gpu`; system deps `colmap, ffmpeg, libsm6, libxext6, git`; default CMD trains the `lego` synthetic scene end to end.

## Risk Assessment (7 risks identified)
| Severity | Risk | Mitigation |
|---|---|---|
| **High** | Full paper-faithful training (~1-2 days on a V100, 800×800/1008×756 images) won't fit a 4GB-VRAM consumer GPU | Ship both `configs/full_paper.yaml` and `configs/tiny_consumer.yaml` (TinyNeRF-inspired) as first-class, clearly documented alternatives |
| Medium | TinyNeRF adaptation's exact hyperparameters are unverified (external notebook, not the paper) | Isolated in `utils/tiny_variant.py`, opt-in via `--tiny` flag, never touches the default paper-faithful path |
| Medium | Exact training-step count / LR decay rate not pinned in paper | Exposed as top-level config values, not hardcoded, with assumption documented in comments |
| Medium | Hierarchical inverse-CDF sampling is a fiddly custom op | Epsilon-stabilized PDF/CDF, unit-tested, default `stop_gradient` on coarse weights (configurable) |
| Low | Weight init unspecified | Keras default (Glorot uniform), matches common community reproductions |
| Low | COLMAP system dependency for real scenes | Isolated to `preprocess_llff.py`; synthetic datasets need no COLMAP |
| Low | TF2 LPIPS port less standardized than PyTorch original | Pinned version + documented PyTorch-subprocess fallback for eval only |

## Handling of Low-Confidence SIR Sections
- **≥0.8 confidence** (architecture, math spec, tensor semantics, evaluation): implemented directly, no hedging.
- **0.6-0.79** (training_pipeline=0.8 sits at the boundary; specific ambiguous fields like exact step count/LR schedule): implemented with config-exposed values and inline "ASSUMED" documentation rather than hardcoded silent choices.
- **<0.6** (TinyNeRF adaptation=0.5): fully isolated behind an explicit opt-in swappable preset (`utils/tiny_variant.py`, `--tiny` flag), and flagged as the top risk-assessment item.

---
Next stage: **Stage 4 — Code Generator**, which will use this plan as its blueprint to write the actual `src/nerf/**/*.py`, `configs/*.yaml`, `Dockerfile`, and `requirements*.txt` files into `paper-repos/arxiv_2003_08934/`.
