# Hallucination Report

**Paper**: NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis (arXiv:2003.08934)
**Comparison Date**: 2026-07-29
**Scope**: Full review of `architecture_plan.json` and generated `src/nerf/**` against `sir.json`, cross-referenced against the observed deviation in `benchmark_comparison.md`.

---

## Summary

| Type | Count | Critical | Significant | Minor |
|------|-------|----------|-------------|-------|
| Structural | 0 | 0 | 0 | 0 |
| Parametric | 3 | 0 | 1 | 2 |
| Omission | 2 | 0 | 1 | 1 |

**No structural hallucinations found** — every module in `architecture_plan.json`'s `module_hierarchy` traces to a specific SIR `architecture.modules[]` entry or SIR `mathematical_spec[]` equation; nothing was invented outside the SIR's scope.

---

## Structural Hallucinations

None found. All 9 SIR-documented architecture modules (`PositionalEncoding_x`, `PositionalEncoding_d`, `MLP_Trunk`, `DensityHead`, `ColorHead`, `CoarseNetwork`, `FineNetwork`, `HierarchicalSampler`, `VolumeRenderer`) map 1:1 to concrete classes in `src/nerf/models/`. No extraneous components (e.g. no invented regularizers, no invented loss terms, no invented architectural blocks) were introduced beyond what the SIR specifies.

---

## Parametric Hallucinations

### 1. TinyNeRF consumer-hardware preset hyperparameters
- **Severity**: Significant
- **Location**: `src/nerf/utils/tiny_variant.py::TinyNeRFPreset._OVERRIDES`, `configs/tiny_consumer.yaml`
- **Evidence**: `trunk_depth: 4`, `trunk_width: 128`, `pos_enc_freqs_x: 6`, `ray_batch_size: 1024`, `num_training_steps: 5000` are all marked `# ASSUMED` in the config and traced to SIR `implementation_assumptions[3]` (confidence **0.5** — the lowest-confidence entry in the entire SIR). These values are the repo's own best-effort estimate of "what the official TinyNeRF demo notebook likely uses," not a verified extraction from either the paper or the notebook itself (the notebook was never actually fetched/parsed during Stage 1-4).
- **Coincidence with deviation**: **Directly implicated.** This is the dominant root cause of the -59.08% PSNR deviation reported in `benchmark_comparison.md`.
- **Suggested fix**: If exact TinyNeRF-notebook fidelity is desired, fetch and parse `https://colab.research.google.com/github/bmild/nerf/blob/master/tiny_nerf.ipynb` directly (flagged as a follow-up option in SIR `ambiguities[3]`) rather than relying on the current best-effort estimate. For your immediate reproducibility goal, however, the fix isn't "make TinyNeRF more accurate" — it's to stop using it and instead use a higher-capacity single-network config (see `benchmark_comparison.md` Recommended Actions #1).

### 2. Learning-rate exponential decay formula
- **Severity**: Minor
- **Location**: `src/nerf/training/trainer.py::Trainer.__init__` (`ExponentialDecay` construction)
- **Evidence**: Traced to SIR `ambiguities[1]` (confidence 0.6). The paper states lr "begins at 5e-4 and decays exponentially to 5e-5" without specifying the exact decay-rate formula; the code assumes `decay_rate = lr_final/lr_init` over `decay_steps = num_training_steps`, which is a standard but unverified choice.
- **Coincidence with deviation**: Unlikely to meaningfully contribute — your training log showed the loss plateauing early, consistent with a capacity ceiling, not a suboptimal LR schedule.
- **Suggested fix**: Low priority; if pursuing exact paper fidelity later, this is a candidate for sensitivity analysis (e.g. try staircase decay vs. the current smooth exponential).

### 3. Weight initialization scheme
- **Severity**: Minor
- **Location**: `src/nerf/models/nerf_mlp.py::NeRFMLP` (uses Keras `Dense` layer defaults)
- **Evidence**: Traced to SIR `implementation_assumptions[1]` (confidence 0.55) — the paper does not specify an initialization scheme at all; Glorot-uniform (Keras default) was assumed as standard practice.
- **Coincidence with deviation**: Not implicated — initialization affects early training dynamics, not a hard capacity ceiling reached after 125k+ steps.
- **Suggested fix**: Not a priority.

---

## Omission Hallucinations

### 1. `lpips-tf2==0.1.0` dependency does not exist on PyPI
- **Severity**: Significant
- **Location**: `requirements.txt` (originally listed, since removed per user correction), `src/nerf/evaluation/metrics.py::ImageMetrics.lpips()`
- **Evidence**: This package was specified in `architecture_plan.json`'s `dependencies.runtime` (Stage 3) without verifying it resolves on PyPI. It does not. This directly caused the LPIPS `Unmatched` row in `benchmark_comparison.md` — one of the paper's three headline metrics (Table 1) is currently uncomputable with this repo as shipped.
- **Mitigating factor**: The code was written defensively — `ImageMetrics.lpips()` catches the `ImportError` and returns `None` with a printed warning rather than crashing the evaluation pipeline, so this omission degraded gracefully rather than blocking you entirely (you still got PSNR).
- **Suggested fix**: Per `README.md`'s documented fallback, run the well-maintained PyTorch `lpips` package in a small separate environment against the saved rendered PNGs in `results/*/`, decoupled from the main TensorFlow training stack. This is a real gap in the shipped repo, not something to route around silently.

### 2. `DeepVoxelsDataset` is an unimplemented stub
- **Severity**: Minor (for this specific comparison — not used in your Lego run)
- **Location**: `src/nerf/data/dataset.py::DeepVoxelsDataset`
- **Evidence**: SIR `evaluation_protocol.datasets` lists "Diffuse Synthetic 360 (DeepVoxels)" as one of three paper benchmarks (confidence 0.95 that it's used by the paper), but the class docstring documents that DeepVoxels' own pose/intrinsics file format could not be derived purely from the paper text and raises `NotImplementedError`. This is a documented, intentional omission (flagged in `data/README_data.md`), not a silent gap.
- **Coincidence with deviation**: None — irrelevant to your Lego/Realistic-Synthetic-360 run.
- **Suggested fix**: Only relevant if you later attempt to reproduce Table 3 (Diffuse Synthetic 360 numbers); implement against the official DeepVoxels dataset format at that point.

---

## Note on a Defect Found and Corrected Before Release

During Stage 4 code-generation QA (before this repo was ever shared with you), functional smoke-testing caught a real shape-logic bug in `HierarchicalSampler.sample_pdf` (incorrect bins/weights length invariant) and a related out-of-range skip-connection bug in the original `TinyNeRFPreset` draft (`trunk_depth=4` with `skip_layers=[4]`, which is out of bounds for a 4-layer trunk). Both were caught via gradient-flow and forward-pass testing and corrected prior to shipping — they are **not** present in the code you are running, and are noted here only for full audit-trail transparency (see `verification_log.md`).

---

## What This Report Does NOT Cover

This report reviews the *generated implementation* against the *SIR*. It does not re-litigate whether the SIR itself correctly captured the paper (that was Stage 1's job, with its own confidence scores) — see `sir.json`'s `ambiguities[]` for paper-parsing-level uncertainty, separate from the implementation-level items above.
