# Verification Log
**Paper ID**: arxiv_2105_08050
**Run Date**: 2026-06-12
**Run Type**: Pipeline Smoke Test (Stage 6, reduced scope per user request)

---

## Audit Trail

| Field | Value |
|---|---|
| SIR version used | 1 (`sir-registry/arxiv_2105_08050/sir.json`) |
| SIR overall confidence | 0.91 |
| Architecture plan version used | 1 (`architecture_plan.json`) |
| Architecture plan confidence | 0.93 |
| Generated code commit state | Stage 4 output, 68/68 unit tests passing |
| Comparison scope | Execution validation only — NOT a metric reproduction |

---

## Execution Environment

- Hardware: CPU (sandbox container)
- PyTorch: as installed in Stage 4 environment
- Precision: float32
- Seed: 42 (set via `gmlp.utils.config.set_seed`)
- Data: synthetic (random token IDs / random images) — no real C4/ImageNet data used

---

## Configurations Tested

### 1. gMLP NLP (MLM)
```
model_type=nlp, use_tiny_attn=False, num_layers=4, d_model=128, d_ffn=512,
seq_len=64, vocab_size=1000, use_toeplitz=True, w_init_std=0.002
```
- Params: 656,868
- Steps: 60, batch=8
- Optimizer: AdamW (lr=1e-3, betas=(0.9,0.999), eps=1e-6, wd=0.01) — paper Table 8 values
- Result: loss 6.3096 → 3.2418 (−48.6%)

### 2. aMLP NLP (MLM + tiny attention)
```
model_type=nlp, use_tiny_attn=True, num_layers=4, d_model=128, d_ffn=512,
seq_len=64, vocab_size=1000, d_attn=32, attn_fusion_mode='add', use_toeplitz=True
```
- Params: 738,788
- Steps: 60, batch=8
- Result: loss 6.3066 → 3.2366 (−48.7%)

### 3. gMLP Vision
```
model_type=vision, num_layers=4, d_model=128, d_ffn=512, seq_len=16,
img_size=32, patch_size=8, num_classes=10, use_toeplitz=False,
survival_prob=1.0, pool_mode='avg'
```
- Params: 426,186
- Steps: 60, batch=8
- Optimizer: AdamW (lr=1e-3, wd=0.05) — paper Table 7 values
- Result: loss 2.3250 → 2.3356 (flat, expected on random data)

---

## Metrics Compared to Paper

**None.** Per user request, this run was scoped explicitly as a smoke test.
Paper Table 3/4/6 perplexity and downstream-task numbers were NOT targeted.

`metrics_compared = 0`
`metrics_matched = 0`

---

## Config Modifications from Paper Defaults

All three runs deliberately use drastically reduced configs vs. paper presets
(`gmlp-base-mlm`, `gmlp-S-imagenet`, etc.) for tractability:

| Param | Paper (gMLPbase) | Smoke Test |
|---|---|---|
| num_layers | 48 | 4 |
| d_model | 512 | 128 |
| d_ffn | 3072 | 512 |
| seq_len | 512 | 64 |
| vocab_size | 32000 | 1000 |
| batch_size | 256 | 8 |
| num_steps | 1,000,000 | 60 |
| dataset | C4 (real) | random tensors |

These are NOT config "bugs" — they are intentional smoke-test reductions, distinct
from the SIR ambiguities (`w_init_std`, `attn_fusion_mode`, `pool_mode`) which
were left at their assumed values (0.002 / 'add' / 'avg') unchanged.

---

## Output Artifacts

- `comparison/smoke_test_results.json` — raw loss curves and param counts
- `comparison/benchmark_comparison.md` — this comparison, framed as smoke test
- `comparison/reproducibility_score.json` — score fields null/N/A, smoke summary populated
- `comparison/hallucination_report.md` — audit of Stage 4 code vs SIR (no new findings; 3 open parametric ambiguities re-flagged, 1 moderate omission re-flagged)

---

## Input Hash

No external user-provided result file was given (results generated in-session).
`sha256(smoke_test_results.json)`:
19b0dead58410f9836f8a6a8312ffda92ca486990797c40471abde0555403f3a
