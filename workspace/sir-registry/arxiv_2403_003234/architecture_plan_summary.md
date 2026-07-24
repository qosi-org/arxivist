# Architecture Plan — Caduceus (arxiv_2403_003234)

**Framework:** PyTorch ≥2.1 · HuggingFace (official weights, `trust_remote_code`) · YAML · **CUDA required (Mamba kernels)**

## Strategy
Load the **official `kuleshov-group/caduceus-*`** checkpoints — bi-directional, RC-equivariant DNA
LMs built on the **Mamba** selective-SSM block. Loads via `AutoModel(trust_remote_code=True)`; the
real forward pass needs the **mamba-ssm + causal-conv1d** fused CUDA kernels (GPU-only). Attach a
**mean-pool + linear** classification head (Sec 5.2.1) and fine-tune on **Genomic Benchmarks**.
Two equivariance modes: **Caduceus-Ph** (post-hoc conjoining = RC-ensemble at inference, best overall)
and **Caduceus-PS** (parameter sharing; average the two channel-split halves for RC-invariance).

A pure-PyTorch **BiMamba / MambaDNA reference** is shipped to unit-test the RC-equivariance property
(Thm 3.1) on CPU without the CUDA kernels.

## Module hierarchy
```
src/caduceus/
├── models/
│   ├── rc_equivariance.py  # BiMamba, MambaDNA, RC-equiv embed/LM head (reference, CPU-testable)
│   └── classifier.py       # CaduceusClassifier.from_pretrained(...) (mean-pool + conjoin)
├── data/
│   ├── tokenizer.py        # CharTokenizer (A/C/G/T/N) + reverse_complement
│   └── benchmarks.py       # GenomicDataset (Genomic Benchmarks, RC aug)
├── training/trainer.py     # Trainer.fit() — AdamW, cosine, early stop, 5-fold CV
├── evaluation/metrics.py   # accuracy, MCC, F1, AUROC
└── utils/config.py         # YAML + seed + task registry
```

## Data flow (downstream, Caduceus-Ph)
`raw DNA` → char-tokenize → `input_ids [B,L]` → Caduceus (BiMamba stack) → `hidden [B,L,D]`
→ mean-pool → `[B,D]` → linear head → `logits [B,num_classes]`
(inference: also run on RC(seq), average logits — post-hoc conjoining)

## RC equivariance (Thm 3.1)
`MambaDNA(X) = concat( M(X_a), RC(M(RC(X_b))) )`, split along channels, shared `M`.
Unit test: `RC(MambaDNA(x)) ≈ MambaDNA(RC(x))`.

## Entrypoints
- `train.py` — `--config --task --seed --debug --dry-run`
- `evaluate.py` — `--config --checkpoint`
- `data/download.py` — Genomic Benchmarks via API

## Key config
`model_name=kuleshov-group/caduceus-ph_seqlen-1k_d_model-118_n_layer-4_lr-8e-3`, variant `ph`,
`lr=1e-3 (# ASSUMED from {1e-3,2e-3} grid)`, cosine, early-stop patience 3, mean pooling, RC aug + conjoin.

## Top risks
1. **[High]** Official forward needs mamba-ssm/causal-conv1d CUDA kernels → GPU-only, version-sensitive build. Ship CPU reference + graceful fallback.
2. **[Med]** Per-task LR grid-searched {1e-3,2e-3} → default one point, `# ASSUMED`; reproduce one task not all 8.
3. **[Low]** Confirm HF repo id / variant at load (read hidden size from config).
4. **[Low]** PS vs Ph post-processing differ → `variant` flag drives it; default Ph.
