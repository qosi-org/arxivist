# Verification Log — Stage 6 Comparison Run

**Paper ID**: arxiv_2003_08934
**Run timestamp**: 2026-07-29T21:25:28Z

---

## Pipeline Provenance

| Artifact | Version / Path |
|---|---|
| SIR version used | 1 (`sir-registry/arxiv_2003_08934/sir.json`) |
| Architecture plan version used | 1 (`sir-registry/arxiv_2003_08934/architecture_plan.json`) |
| Overall SIR confidence | 0.85 |
| Repo generation stage | Stage 4 (Code Generator), validated via functional smoke tests prior to release |

---

## User Results Input

| Field | Value |
|---|---|
| Source file | `metrics.json` (uploaded by user) |
| SHA256 | `8c6ec724a9942061af1bdc80a931b6ab53437e5464ddb4261c88f6e80ab0d464` |
| Format | JSON, `evaluate.py` output schema (`{"per_image": [...], "mean": {...}}`) |
| Number of test images evaluated | 200 |
| Metric(s) present | PSNR only (SSIM/LPIPS not present — consistent with `configs/tiny_consumer.yaml`'s `evaluation.metrics: ["psnr"]`) |
| Reported mean PSNR | 12.690099668502807 |
| Reported PSNR range | min 9.370 (index 173) — max 18.943 (index 103) |

**User-reported config/run modifications** (from conversation context, not re-derivable from `metrics.json` alone):
- Config: `configs/tiny_consumer.yaml` with `--tiny` flag applied (`TinyNeRFPreset`)
- `--num_steps 200000` override (config default: 5000)
- Dataset: Lego scene only (`nerf_synthetic/lego`)
- Hardware: NVIDIA RTX 3050 (6GB VRAM), WSL2/Ubuntu
- Training completed to step 200,000 per prior conversation log (checkpoint `ckpt-200000` implied, exact checkpoint path not confirmed in this upload)

---

## Paper Metrics Retrieved

From `sir.json → evaluation_protocol.reported_results`:

| Metric | Dataset | Split | Value | is_primary |
|---|---|---|---|---|
| PSNR | Diffuse Synthetic 360 | test | 40.15 | true |
| SSIM | Diffuse Synthetic 360 | test | 0.991 | true |
| LPIPS | Diffuse Synthetic 360 | test | 0.023 | true |
| PSNR | Realistic Synthetic 360 | test | 31.01 | true |
| SSIM | Realistic Synthetic 360 | test | 0.947 | true |
| LPIPS | Realistic Synthetic 360 | test | 0.081 | true |
| PSNR | Real Forward-Facing | test | 26.50 | true |
| SSIM | Real Forward-Facing | test | 0.811 | true |
| LPIPS | Real Forward-Facing | test | 0.250 | true |

**Total paper metrics in scope for this comparison** (Realistic Synthetic 360 dataset, matching user's Lego scene): 3 (PSNR, SSIM, LPIPS).
**Matched to user results**: 1 (PSNR).
**Unmatched**: 2 (SSIM, LPIPS — not computed by user's run).

**Supplementary (not in `sir.json`'s `reported_results` array, recalled from paper Table 4 text for scene-level context)**: Lego per-scene PSNR = 32.54.

---

## Metrics Compared

- PSNR, Realistic Synthetic 360 (aggregate) vs. user's Lego PSNR → **-59.08%** deviation → Critical
- (Supplementary, non-scored) PSNR, Lego per-scene (Table 4) vs. user's Lego PSNR → -61.00% deviation

---

## Scoring Computation Trace

```
matched_pair: PSNR, pct_deviation = -59.08%
base_score = 1 - min(59.08/50, 1.0) = 1 - 1.0 = 0.0

sir_confidence_scores used = [architecture:0.90, mathematical_spec:0.93,
  tensor_semantics:0.88, training_pipeline:0.80, evaluation_protocol:0.95,
  implementation_assumptions:0.65]
mean(sir_confidence_scores) = 0.8517
sir_confidence_penalty = (1 - 0.8517) * 0.15 = 0.0222

unmatched_count = 2, total_paper_metrics_in_scope = 3
unmatched_penalty = (2/3) * 0.2 = 0.1333

reproducibility_score = max(0, 0.0 - 0.0222 - 0.1333) = max(0, -0.1556) = 0.0
```

**Score confidence classification**: Low — per methodology rule, applies when the user "modified config substantially" (here: `TinyNeRFPreset`/`--tiny`, a documented low-confidence architecture deviation), even though exactly 1 direct metric match was obtained.

---

## Hallucination Review Scope

Reviewed: `architecture_plan.json` `module_hierarchy` (17 entries), `sir.json` `architecture.modules` (9 entries), `sir.json` `implementation_assumptions` (6 entries), `sir.json` `ambiguities` (4 entries), and the shipped `src/nerf/**` source tree.

Findings: 0 structural, 3 parametric (1 Significant, 2 Minor), 2 omission (1 Significant, 1 Minor). Full detail in `hallucination_report.md`.

---

## Pre-Release Corrections Noted (Not Present in Shipped Code)

For full transparency, two defects were identified and fixed during Stage 4 QA, before this repository was shared:
1. `HierarchicalSampler.sample_pdf` shape-invariant bug (bins/weights length mismatch) — fixed, re-verified via smoke test.
2. `TinyNeRFPreset` out-of-range skip-connection (`skip_layers=[4]` with `trunk_depth=4`) — fixed to `skip_layers=[2]`, re-verified.

Neither is implicated in the current comparison result, as both were corrected prior to the version of the repo the user downloaded and trained with.

---

## Manual Review Determination

**Required**: No.

**Reasoning**: The observed -59.08% PSNR deviation has a clear, high-probability, documented root cause (the opt-in `TinyNeRFPreset` architecture reduction, SIR confidence 0.5) rather than an unexplained anomaly. No evidence of a silent implementation defect was found beyond the two pre-release-corrected items above (neither of which is present in the shipped code). The recommended next step is a re-run with a higher-capacity single-network config, not a code fix.
