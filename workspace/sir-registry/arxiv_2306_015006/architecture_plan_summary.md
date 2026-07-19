# Architecture Plan — DNABERT-2 (arxiv_2306_015006)

**Framework:** PyTorch ≥2.1 · HuggingFace (official weights) · YAML · CUDA optional (Colab-friendly)

## Strategy
Load the **official `zhihanzhou/DNABERT-2-117M`** via `AutoModel` + `AutoTokenizer`
(`trust_remote_code=True`) — this ships the BPE tokenizer (Sec 3.1) and ALiBi/GEGLU encoder
(Sec 3.2) key-for-key, so no reimplementation is needed (unlike HyenaDNA). Attach a classification
head and fine-tune per the paper's **Appendix A.3** recipe. A **configurable GUE loader** covers all
28 GUE datasets; default target is promoter_detection/all (paper MCC 86.77).

## Module hierarchy
```
src/dnabert2/
├── models/classifier.py     # DNABERT2Classifier.from_pretrained(...)
├── data/
│   ├── tokenizer.py         # DNATokenizer (official BPE)
│   └── gue.py               # GUEDataset — all 28 tasks, configurable
├── training/trainer.py      # Trainer.fit() — AdamW lr3e-5, warmup50, best-by-val
├── evaluation/metrics.py    # MCC + F1 (Table 12 metrics)
└── utils/config.py          # YAML + seed + GUE task registry
```

## Data flow
`raw DNA` → BPE tokenize → `input_ids/attention_mask [B,L]` → DNABERT-2 backbone → `hidden [B,L,768]`
→ masked mean-pool → `[B,768]` → linear head → `logits [B,num_classes]`

## Entrypoints
- `train.py` — `--config --task --subset --seed --debug --dry-run`
- `evaluate.py` — `--config --checkpoint`
- `inference.py` — `--config --sequence --checkpoint`
- `data/download.py` — **GUE via API** `--task --data-dir`

## Key config (from Appendix A.3 — high confidence)
`model_name=zhihanzhou/DNABERT-2-117M`, `lr=3e-5`, `wd=0.01`, `batch=32`, `warmup_steps=50`,
`epochs=4` (per-task per Table 7), `metric=mcc`, `pool=mean (# ASSUMED)`.

## Top risks
1. **[Med]** GUE dataset size/format varies → download.py + small default task.
2. **[Low]** Flash-Attn falls back to eager on CPU/T4 → handled by remote code.
3. **[Low]** Pooling not specified → config-selectable, default mean.
