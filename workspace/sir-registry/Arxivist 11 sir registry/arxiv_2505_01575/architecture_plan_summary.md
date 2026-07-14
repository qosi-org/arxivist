# Architecture Plan Summary — Asset Pricing in Pre-trained Transformers (arxiv_2505_01575)

## Framework
PyTorch (>=2.1), Python 3.10+, CPU-friendly (paper trains on a single RTX 3070 or CPU laptop
for limited data size), plain YAML config (no Hydra needed — small, paper-scoped config surface).

## What gets built
Six model families sharing common building blocks:

| Family | Decoder? | Cross-attn? | Pretrain MLP autoencoder? | LNF? |
|---|---|---|---|---|
| PretrainedTransformer (`P_Trans_H*`) | Yes | Yes | Yes | No |
| PretrainedTransformerLNF (`P_Trans_LNF`) | Yes | Yes | Yes | Yes |
| StandardTransformer (`C_Trans_H*`, benchmark) | Yes | Yes | No | No |
| SERT (`SERT_H*`) | No | No | Yes | No |
| SERTLNF (`SERT_LNF`) | No | No | Yes | Yes |
| EncoderOnlyTransformer (`En_Trans_H*`, benchmark) | No | No | No | No |

## Module hierarchy (12 files)
- `models/positional_encoding.py` — Sinusoidal PE (Eq. 1-2)
- `models/mlp_autoencoder.py` — shared MLP autoencoder (used both as pretrain projector 182→420
  and as in-block FFN)
- `models/attention.py` — causal mask builder + masked multi-head self/cross attention (Eq. 3-11)
- `models/blocks.py` — EncoderBlock / DecoderBlock combining attention + FFN + Add&Norm (post-LN)
  or LNF (pre-LN)
- `models/transformer_variants.py` — top-level model classes for all 6 families
- `data/dataset.py`, `data/transforms.py` — rolling-window panel dataset (102-month train /
  30-month val / step 12), missing-value filtering (drop factors with >40% missing)
- `training/losses.py`, `training/trainer.py` — MSE+L1 loss, Adam optimizer, early stopping,
  rolling re-estimation every 12 months
- `evaluation/metrics.py` — corrected OOS R2 (Campbell-Thompson denominator), Diebold-Mariano
  with HAC estimator
- `evaluation/backtest.py` — sign-signal & softmax-filtered backtests, equal/value-weighted,
  static (50bps) & dynamic (20bps × turnover) transaction costs, Sharpe/Sortino/MDD
- `utils/config.py` — YAML config loader + Python/NumPy/PyTorch seeding

## Key tensor flow (Pretrained Transformer)
```
X_raw [B,T,182] → MLPAutoencoder → [B,T,420] → InputEmbedding → [B,T,420]
→ +SinusoidalPE → EncoderBlock(N*=1, causal self-attn) → [B,T,420]
→ DecoderBlock(self-attn + cross-attn to encoder) → [B,T,420] → OutputDenseLayer → [B,T,1]
```
SERT drops the decoder/cross-attention entirely and connects the encoder output straight to the
output dense layer.

## Entrypoints
- `train.py --config configs/config.yaml [--resume] [--seed] [--debug] [--dry-run]`
- `evaluate.py --config ... --checkpoint ... --period {1911,2112,2212}`
- `backtest.py --config ... --checkpoint ... --weighting {equal,value} --tc-mode {static,dynamic} [--softmax-filter]`
- `inference.py --config ... --checkpoint ... --input-window <csv>`

## Top risks (see architecture_plan.json → risk_assessment for full list)
1. **HIGH** — Exact 420-stock universe & 182-factor construction are not enumerated in the paper;
   true numerical reproduction needs the original Zimmermann factor library + CRSP extract.
2. **MEDIUM** — L1 lambda and pretrain-autoencoder internal sizing are unspecified; exposed as
   config-tunable `ASSUMED` values.
3. **MEDIUM** — No LR schedule/batch size/epoch count given; using constant Adam LR=1e-3 with
   early stopping, config-tunable.
4. **LOW** — H5 (5-head) variant is absent from all paper tables with no explanation; supported
   in code but not run by default.
5. **LOW** — DM-test HAC lag length defaults to 12 (monthly, one year), config-tunable.

## Docker
Base image `pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime`, system deps `git wget`,
workdir `/workspace`, default CMD shows `train.py --help`.
