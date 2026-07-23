# Verification Log

**Comparison run**: 2026-07-23
**Paper ID**: arxiv_2502_007272

## Provenance
- ArXivist SIR version used: 1
- Architecture plan version used: 1
- Model: `GenerTeam/GENERator-eukaryote-1.2b-base` (official), loaded via `AutoModel`
- Reported at runtime: **1.15B parameters** (paper ~1.2B ✅), `model_type: llama`
- Weights: model.safetensors 4.65 GB; `lm_head.weight` UNEXPECTED (pretraining head, correctly unused)
- Tokenizer: official GENERator 6-mer, vocab 4128 (paper Sec 4.2 ✅)
- Hardware (user run): A100-class GPU, bf16 autocast; ~9 min/epoch × 10 epochs

## Architecture confirmation (SIR inferred vs actual config)
| Field | SIR | HF config.json | Match |
|---|---|---|---|
| hidden_size | 2048 | 2048 | ✅ |
| num_hidden_layers | 26 | 26 | ✅ |
| num_attention_heads / KV | 32 / 4 | 32 / 4 | ✅ |
| intermediate_size | 5632 | 5632 | ✅ |
| vocab_size | 4128 | 4128 | ✅ |
| max_position_embeddings | 16384 | 16384 | ✅ |
| hidden_act | SiLU | silu | ✅ |

## Metrics
- Paper metrics available for this dataset: 1 (accuracy)
- User results provided: 1 (accuracy)
- Matched pairs: 1

| metric | paper | user | dev % | severity |
|---|---|---|---|---|
| accuracy | 0.958 | 0.8963 | −6.44 | moderate |

## Training summary
Best val accuracy 0.8963 at epoch 7 (checkpointed). Loss 0.35 → 0.02; validation plateaued from
~epoch 4, indicating convergence at lr=1e-5 (supports the "LR too low" root cause).

## User-reported config modifications
None — stock `configs/config.yaml` (task human_nontata_promoters, lr 1e-5, bs 64, 10 epochs,
last-token pooling, reduce-on-plateau + early stop).

## Integrity
- User results SHA256: `7acd5f8d92a1e59a3c865b86da27540105fe921bb4eb0b05f1680a7cabd3f2a2`
- Manual review required: No
- Review reasons: none

## Confidence of this comparison
**Medium.** One matched metric, single fold, single seed, untuned LR — versus the paper's 10-fold CV
+ grid-searched hyperparameters. Would rise to **High** with a grid search + 10-fold average, which
would also likely close most of the −6.44% gap.
