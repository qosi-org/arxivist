# Architecture Plan — DNAGPT (arxiv_2307_005628)

**Framework:** PyTorch ≥2.1 · **from-scratch (no HF Hub, no custom kernels)** · YAML · **CPU-runnable architecture; GPU only for fine-tuning**

## Strategy
DNAGPT's contribution **is** a custom GPT architecture: dual **Sequential + Numerical** embeddings, dual
**Classification + Regression** heads, a bespoke **token language**, and a **3-task pre-training loss**.
Official weights are `.pth` on **Google Drive / Weiyun** (not HF Hub), loaded via the authors' own class
(`test.py`). So we (a) **re-implement DNAGPT from scratch** in plain PyTorch — fully **CPU-unit-testable**
(token language, dual heads, combined loss), and (b) provide an **official-weights loading path**
(`from_pretrained` with state_dict key-mapping) + a **GSR fine-tuning** harness to hit Table S2 on GPU.

## Module hierarchy
```
src/dnagpt/
├── models/
│   ├── dnagpt.py     # DNAGPT: dual embeds -> GPT decoder -> dual heads; from_pretrained(.pth)
│   └── blocks.py     # GPTBlock (causal MHSA+FFN+LN), Numerical embed, Regression head
├── data/
│   ├── tokenizer.py  # DNAGPTTokenizer (non-overlap k-mer + token language, Fig 2/S2)
│   └── gsr.py        # GSRDataset (DeepGSR PAS/TIS, 300bp flanks, motif removed)
├── training/
│   ├── losses.py     # L = 0.01*MSE + CE  (NTP + GC-regression + sequence-order)
│   └── trainer.py    # GSR fine-tune (AdamW, cosine+warmup, 10 ep, lr 3e-5, wd 1e-1, bs 8)
├── evaluation/metrics.py  # acc/f1/mcc/precision/recall, r2, Wasserstein
└── utils/config.py   # YAML + seed + variant registry (H/M/S-512/B-512) + GSR tasks
```

## Data flow (GSR classification)
`DNA` → non-overlap 6-mer + species instruction token → SequentialEmbedding → GPT decoder ×12
→ final-token hidden → Classification head → `logits [B,2]` (True/False = 'A'/'N' tokens)

## Combined pre-training loss (Eq 1)
`L = λ·MSE_loss + CrossEntropy_loss`, **λ = 0.01**. NTP (CE) + GC-content (MSE) + sequence-order (CE, flip p=0.5).

## Variants
| | H | M | S-512 | B-512 |
|---|---|---|---|---|
| layers | 12 | 12 | 12 | 60 |
| hidden | 768 | 768 | 768 | 2048 |
| params | 0.1B | 0.1B | 0.1B | 3B |
| seq len | 4096 | 4096 | 512 | 512 |
| pretrain | human | 9 species | all mammals | all mammals |

## Entrypoints
- `train.py` — `--config --task --seed --debug --dry-run`
- `evaluate.py` — `--config --checkpoint`
- `data/download.py` — official `.pth` (Google Drive via gdown) + GSR data

## Reproduction target (Table S2, DNAGPT-M)
Human PAS(AATAAA): **91.51% acc / 82.99 MCC**; Human TIS(ATG): **97.46% acc**. Finetune 10 ep, lr 3e-5.

## Top risks
1. **[High]** Google-Drive `.pth` + custom class → state_dict key-map must be validated vs real checkpoint; from-scratch fine-tune is the fallback. Architecture + loss unit-tested on CPU regardless.
2. **[Med]** GSR data needs per-organism genome download + 300bp-flank/motif-removal preprocessing; synthetic generator keeps pipeline/tests runnable.
3. **[Med]** Numerical-embed / regression-head MLP shapes unspecified → `# ASSUMED` (GSR path doesn't use them).
4. **[Low]** Fig S3 'momentum 0.937' alongside AdamW betas is odd → use standard betas.
