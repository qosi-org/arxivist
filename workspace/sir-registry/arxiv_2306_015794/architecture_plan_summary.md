# Architecture Plan — HyenaDNA (arxiv_2306_015794)

**Framework:** PyTorch ≥2.1 · HuggingFace Transformers (pretrained weights) · YAML config · CUDA optional (Colab-friendly)

## Strategy
Primary path loads **official `LongSafari/hyenadna` pretrained weights** from the HuggingFace Hub and attaches a classification head for downstream fine-tuning. A **from-scratch order-2 Hyena operator** (`hyena_operator.py`) is included as a labeled *reference* implementation (FFT long conv + implicit filter MLP) but is not on the reproduction critical path.

## Module hierarchy
```
src/hyenadna/
├── models/
│   ├── hyena_operator.py   # HyenaFilter, HyenaOperator, HyenaBlock (reference)
│   └── pretrained.py       # HyenaDNAClassifier.from_pretrained(...)
├── data/
│   ├── tokenizer.py        # CharTokenizer (A/C/G/T/N + specials)
│   └── dataset.py          # GenomicDataset
├── training/
│   ├── trainer.py          # Trainer.fit()
│   └── losses.py           # CE + reference causal-LM loss
├── evaluation/
│   └── metrics.py          # accuracy, MCC, F1
└── utils/
    └── config.py           # YAML loader + seed_everything
```

## Data flow (downstream)
`input_ids [B,L]` → pretrained backbone → `hidden [B,L,D]` → mean-pool → `[B,D]` → linear head → `logits [B,num_classes]`

## Entrypoints
- `train.py` — `--config --resume --seed --debug --dry-run`
- `evaluate.py` — `--config --checkpoint`
- `inference.py` — `--config --sequence --checkpoint`
- `data/download.py` — **`{genomic-benchmarks | hg38 | nt-benchmarks}` via API/curl** `--dataset --data-dir`

## Key config defaults
`variant=hyenadna-tiny-1k-seqlen`, `d_model=128`, `n_layer=2`, `max_len=1024`, `lr=6e-5 (# ASSUMED downstream)`, `wd=0.1`, `batch=32`, `cosine + 10% warmup`, `bf16`.

## Top risks
1. **[Med]** Hyena filter MLP internals inferred (conf 0.55) → mitigated by using pretrained weights.
2. **[Med]** Training LR/warmup/precision inferred (conf 0.62) → all in config, `# ASSUMED` tagged.
3. **[Low]** NT datasets need processed download; **[Low]** hg38 ~1GB.
