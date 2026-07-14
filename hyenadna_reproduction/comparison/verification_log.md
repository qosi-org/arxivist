# Verification Log

**Comparison run**: 2026-07-12
**Paper ID**: arxiv_2306_015794

## Provenance
- ArXivist SIR version used: 1
- Architecture plan version used: 1
- Pretrained checkpoint: `LongSafari/hyenadna-tiny-1k-seqlen` (weights.ckpt, 5.38 MB)
- Backbone load result: **missing=0, unexpected=4** (full faithful load)
- Model: HyenaDNAModel(d_model=128, n_layer=2), params ≈ 0.44M
- Hardware (user run): CUDA GPU (Colab T4), bf16 autocast

## Metrics
- Paper metrics available for this dataset: 1 (accuracy)
- User results provided: 3 (accuracy, mcc, f1)
- Matched pairs: 1 (accuracy)
- Unmatched (no paper target): 2 (mcc, f1)

## Compared values
| metric | paper | user | dev % | severity |
|---|---|---|---|---|
| accuracy | 0.9660 | 0.9471 | −1.96 | excellent |

## User-reported config modifications
- epochs: 5 → 20
- lr: 6e-5 → 2e-4
- (both applied by the user in the config; recorded and legitimate)

## Reproduction repair history (Stage 4 loops)
1. AutoModel rejected HyenaDNA config (no `model_type`) → switched to direct hf_hub_download.
2. PyTorch 2.6 `weights_only=True` blocked the OmegaConf-embedded ckpt → `weights_only=False`.
3. Hand-written model keys mismatched the checkpoint → vendored authors' `standalone_hyenadna.py`,
   fixed vocab pad-to-16 and tokenizer ids (A→7…N→11). Result: missing=0.

## Integrity
- User results SHA256: `0da942d2fe39439ca052563441ff4eb3fb0cfc56c0d829f52aa4b0472e63d562`
- Manual review required: No
- Review reasons: none

## Confidence of this comparison
**Medium.** Single matched metric on a single dataset; full training run with a genuine pretrained
backbone. Would rise to High with ≥3 matched metrics (e.g. run 2–3 more GenomicBenchmarks datasets).
