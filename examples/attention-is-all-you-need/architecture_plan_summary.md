# Architecture Plan Summary — Attention Is All You Need

**Paper ID**: arxiv_1706_03762
**Plan version**: 1
**Framework**: PyTorch 2.1+

---

## Framework decisions

| Decision | Choice | Reason |
|---|---|---|
| Primary framework | PyTorch | Explicit community standard for NLP; no framework mentioned in paper |
| Python version | 3.10+ | Type annotation support, modern stdlib |
| CUDA required | Yes (recommended) | Training budget of 300K steps is GPU-dependent |
| HuggingFace | Optional | Tokenisation only (`tokenizers` library for BPE) |
| Config library | YAML | Simple; no hydra dependency needed |

---

## Module hierarchy

```
src/
└── transformer/
    ├── __init__.py
    ├── models/
    │   ├── __init__.py
    │   ├── transformer.py          # Seq2SeqTransformer — top-level model
    │   ├── encoder.py              # Encoder + EncoderLayer
    │   ├── decoder.py              # Decoder + DecoderLayer
    │   ├── attention.py            # MultiHeadAttention + ScaledDotProductAttention
    │   ├── feedforward.py          # PositionwiseFeedForward
    │   └── embeddings.py           # InputEmbedding + PositionalEncoding
    ├── data/
    │   ├── __init__.py
    │   ├── dataset.py              # TranslationDataset wrapping torchtext WMT
    │   └── transforms.py           # BPE tokenisation, padding, masking utilities
    ├── training/
    │   ├── __init__.py
    │   ├── trainer.py              # Trainer class with Noam schedule
    │   └── losses.py               # LabelSmoothedCrossEntropy
    ├── evaluation/
    │   ├── __init__.py
    │   └── metrics.py              # BLEU computation via sacrebleu
    └── utils/
        ├── __init__.py
        └── config.py               # Config dataclass + seed utilities
```

---

## Tensor flows

### Encoder forward pass
```
src_tokens: [B, T_src]           int64
  → InputEmbedding               → [B, T_src, 512]   float32
  → PositionalEncoding           → [B, T_src, 512]   float32   (add, no params)
  → for layer in encoder_layers:
      → MultiHeadAttention (self) → [B, T_src, 512]
      → LayerNorm(x + attn_out)  → [B, T_src, 512]
      → PositionwiseFeedForward  → [B, T_src, 512]
      → LayerNorm(x + ffn_out)   → [B, T_src, 512]
encoder_output: [B, T_src, 512]  float32
```

### Decoder forward pass
```
tgt_tokens: [B, T_tgt]           int64
  → InputEmbedding               → [B, T_tgt, 512]   float32
  → PositionalEncoding           → [B, T_tgt, 512]   float32
  → for layer in decoder_layers:
      → MultiHeadAttention (masked self) → [B, T_tgt, 512]
      → LayerNorm(x + self_attn_out)     → [B, T_tgt, 512]
      → MultiHeadAttention (cross)       → [B, T_tgt, 512]   (K,V from encoder)
      → LayerNorm(x + cross_attn_out)    → [B, T_tgt, 512]
      → PositionwiseFeedForward          → [B, T_tgt, 512]
      → LayerNorm(x + ffn_out)           → [B, T_tgt, 512]
  → OutputProjection (linear)    → [B, T_tgt, 37000]  float32
logits: [B, T_tgt, 37000]        float32
```

---

## Config schema (key fields)

```yaml
model:
  d_model: 512          # base; 1024 for big
  d_ff: 2048            # base; 4096 for big
  h: 8                  # attention heads; base; 16 for big
  N: 6                  # encoder/decoder layers
  d_k: 64               # d_model / h
  d_v: 64               # d_model / h
  vocab_size: 37000
  max_len: 5000
  dropout: 0.1          # ASSUMED: 0.3 for big model
  share_embeddings: true

training:
  optimizer: adam
  beta1: 0.9
  beta2: 0.98
  eps: 1.0e-9
  warmup_steps: 4000    # Noam schedule
  max_steps: 100000     # base model
  batch_tokens: 25000   # ASSUMED: total tokens per batch
  label_smoothing: 0.1
  checkpoint_every: 600 # seconds (10 min)
  keep_last_n_ckpts: 5

data:
  src_lang: en
  tgt_lang: de
  dataset: wmt14
  tokenizer: bpe
  max_tokens_per_sample: 512

evaluation:
  beam_size: 4
  length_penalty_alpha: 0.6
  metric: bleu

hardware:
  device: cuda
  seed: 42
  deterministic: false  # true slows training ~15%
  num_workers: 4
```

---

## Dependencies

```
torch>=2.1.0
torchtext>=0.16.0
sacrebleu>=2.3.1
tokenizers>=0.15.0
pyyaml>=6.0
tqdm>=4.66.0
numpy>=1.24.0
```

---

## Risk assessment

| Severity | Risk | Mitigation |
|---|---|---|
| High | Exact BLEU match requires identical BPE vocabulary (37k shared tokens) | Use SentencePiece with the WMT14 en-de vocab released by the community |
| High | Batch token count (25k) ambiguity may cause training instability | Expose `batch_tokens` as a config flag; start with 8192 and scale |
| Medium | Weight tying across three embedding matrices is easy to implement incorrectly | Unit test that `model.src_embed.weight.data_ptr() == model.tgt_embed.weight.data_ptr()` |
| Medium | Layer norm epsilon not specified — affects numerical stability at low lr | Default 1e-6; make configurable |
| Low | Mixed precision not described — float32 assumed | Add `torch.autocast` opt-in flag in config |

---

## Entrypoints

| Script | Purpose | Key flags |
|---|---|---|
| `train.py` | Training | `--config`, `--resume`, `--seed`, `--debug`, `--dry-run` |
| `evaluate.py` | BLEU evaluation | `--config`, `--checkpoint`, `--split` |
| `inference.py` | Translate a sentence | `--config`, `--checkpoint`, `--src` |
| `data/download.sh` | Download WMT14 en-de | (no flags — downloads to `data/raw/`) |
