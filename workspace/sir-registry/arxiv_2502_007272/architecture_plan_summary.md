# Architecture Plan — GENERator (arxiv_2502_007272)

**Framework:** PyTorch ≥2.1 · HuggingFace (official weights) · YAML · **CUDA required (A100-class)**

## Strategy
Load the **official `GenerTeam/GENERator-eukaryote-1.2b-base`** — a LLaMA decoder-only causal LM
(`model_type: llama`, confirmed: 26 layers / d 2048 / 32 heads / 4 KV / intermediate 5632 / vocab 4128
/ ctx 16384 / SiLU). Loads directly via `AutoModel` (no custom kernels). Attach a classification head
on the **last-token (`<EOS>`) embedding** per the paper's Appendix C.4, and fine-tune. A zero-shot
**VEP** path implements the token→nucleotide marginalization + log-likelihood ratio (Sec 4.5).

## Module hierarchy
```
src/generator/
├── models/
│   ├── classifier.py   # GENERatorClassifier.from_pretrained(...)  (last-token pool)
│   └── vep.py          # VEPScorer (zero-shot variant effect)
├── data/
│   ├── tokenizer.py    # GenomicTokenizer (official 6-mer)
│   └── benchmarks.py   # GenomicDataset (Genomic Benchmarks / NT / Gener)
├── training/trainer.py # Trainer.fit() — AdamW β0.9/0.95 wd0.1, reduce-on-plateau, early stop
├── evaluation/metrics.py # accuracy, MCC, weighted-F1, AUROC/AUPRC
└── utils/config.py     # YAML + seed + task registry
```

## Data flow (downstream)
`raw DNA` → 6-mer tokenize → `input_ids [B,L]` → GENERator backbone → `hidden [B,L,2048]`
→ last-token (`<EOS>`) → `[B,2048]` → linear head → `logits [B,num_classes]`

## Entrypoints
- `train.py` — `--config --task --seed --debug --dry-run`
- `evaluate.py` — `--config --checkpoint`
- `vep.py` — `--config --sequence --pos --ref --alt` (zero-shot VEP)
- `data/download.py` — benchmark datasets via API

## Key config (paper Appendix B/C.4)
`model_name=GenerTeam/GENERator-eukaryote-1.2b-base`, pretrain AdamW β=(0.9,0.95) wd 0.1; downstream
`lr=1e-5 / bs=64 (# ASSUMED from grid-search optima)`, reduce-on-plateau + early-stop patience 5,
last-token pooling.

## Top risks
1. **[High]** 1.2B params → needs A100; T4 will OOM. `load_in_8bit` + grad-checkpointing exposed.
2. **[Med]** Per-task LR/BS from grid search → default one point, `# ASSUMED`.
3. **[Low]** Confirm HF repo id / class at load (done: GENERatorForCausalLM, model_type llama).
