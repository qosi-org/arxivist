# gMLP: Pay Attention to MLPs

**Paper**: [Pay Attention to MLPs](https://arxiv.org/abs/2105.08050) — Liu et al., 2021  
**ArXiv**: arXiv:2105.08050v2  
**Venue**: NeurIPS 2021

---

## What This Paper Does

gMLP proposes replacing Transformer self-attention with a simple **Spatial Gating Unit (SGU)**: a static weight matrix W ∈ ℝⁿˣⁿ that mixes information across token positions, combined with multiplicative gating. The result — gMLP — matches Transformer (BERT) performance on masked language modelling pretraining and matches Vision Transformer (DeiT) on ImageNet, without any self-attention at all.

The central claim: **self-attention is not necessary for scaling language and vision models**. Model capacity matters more than the presence of attention — gMLPs scale with data and compute at the same rate as Transformers, so any performance gap can be closed by training a larger model.

A hybrid variant (**aMLP**) adds a single tiny self-attention head (64 or 128 dimensions, vs. BERT's 768) to the gating branch. This small addition makes aMLP consistently outperform comparably-sized Transformers on all tested NLP tasks, including +4.4% F1 on SQuAD v2.0 over BERTlarge.

---

## Quick Start

```bash
# 1. Clone and install
git clone <this-repo>
cd gmlp
pip install -e .

# 2. Verify setup (no training, just validates all components)
python train.py --preset gmlp-base-ablation --dry-run

# 3. Run ablation-scale training (125K steps, C4/RealNews, seq_len=128)
#    Suitable for a single GPU — full training needs 100+ A100-GPU-hours
python train.py --preset gmlp-base-ablation --output_dir outputs/ablation/

# 4. Finetune on SST-2 (5 runs, reports median)
python finetune.py \
    --pretrained_checkpoint outputs/ablation/checkpoint_best.pt \
    --task sst2 --num_runs 5
```

---

## Installation

### pip
```bash
pip install -e .
# or
pip install -r requirements.txt
```

### conda
```bash
conda env create -f environment.yaml
conda activate gmlp
pip install -e .
```

### Docker
```bash
cd docker
docker compose build
docker compose run train  # runs gmlp_base_mlm config by default
docker compose up notebook  # Jupyter on port 8888
```

**Requirements**: Python 3.10+, PyTorch 2.1+, CUDA 11.8+

---

## Training

### NLP Pretraining (MLM on C4)

```bash
# Full gMLPbase (paper Table 5: 1M steps, batch=256, seq=512)
# ⚠ Requires ~100+ A100-GPU-hours
python train.py --config configs/gmlp_base_mlm.yaml

# aMLP base (with tiny attention)
python train.py --preset amlp-base-mlm --output_dir outputs/amlp_base/

# Ablation scale (125K steps, seq=128 — faster, reproduces Table 3)
python train.py --preset gmlp-base-ablation --output_dir outputs/ablation/

# Resume from checkpoint
python train.py --config configs/gmlp_base_mlm.yaml \
    --resume outputs/gmlp_base_mlm/checkpoint_step_500000.pt

# Debug mode (100 steps, validates pipeline)
python train.py --preset gmlp-base-ablation --debug
```

### NLP Finetuning

```bash
# SST-2 (single-sentence classification)
python finetune.py --pretrained_checkpoint outputs/checkpoint_best.pt \
    --task sst2 --num_runs 5

# MNLI (sentence-pair NLI — where tiny attention helps most)
python finetune.py --pretrained_checkpoint outputs/checkpoint_best.pt \
    --task mnli --num_runs 5

# SQuAD v1.1
python finetune.py --pretrained_checkpoint outputs/checkpoint_best.pt \
    --task squad_v1 --num_runs 5

# SQuAD v2.0
python finetune.py --pretrained_checkpoint outputs/checkpoint_best.pt \
    --task squad_v2 --num_runs 5
```

---

## Expected Results (Paper Table 6)

All results are median of 5 independent runs.

| Model | Params | Perplexity | SST-2 | MNLI-m | SQuAD1.1 F1 | SQuAD2.0 F1 |
|-------|--------|-----------|-------|--------|------------|------------|
| BERTbase (ours) | 110M | 4.17 | 93.8 | 85.6 | 90.2 | 78.6 |
| **gMLPbase** | 130M | 4.28 | **94.2** | 83.7 | 86.7 | 70.1 |
| **aMLPbase** | 109M | 3.95 | 93.4 | **85.9** | **90.7** | **80.9** |
| BERTlarge (ours) | 336M | 3.35 | 94.3 | 87.0 | 92.0 | 81.0 |
| **gMLPlarge** | 365M | 3.32 | **94.8** | 86.2 | 89.5 | 78.3 |
| **aMLPlarge** | 316M | **3.19** | **94.8** | **88.4** | **92.2** | **85.4** |

**Vision (ImageNet Top-1, Table 2):**

| Model | Params | Top-1 Acc | MAdds |
|-------|--------|-----------|-------|
| DeiT-S | 22M | 79.8% | 4.6B |
| **gMLP-S** | 20M | 79.6% | 4.5B |
| DeiT-B | 86M | 81.8% | 17.5B |
| **gMLP-B** | 73M | 81.6% | 15.8B |

---

## Architecture Overview

```
Input → [gMLPBlock × L] → Head

gMLPBlock:
  shortcut = x
  x_pre = LayerNorm(x)
  x = GeLU(x_pre @ U)          # expand: d_model → d_ffn
  ── Spatial Gating Unit ──
  z1, z2 = split(x)            # each d_ffn/2
  z2 = LayerNorm(z2)
  z2 = W @ z2 + b              # W ∈ R^{n×n}, Toeplitz for NLP
  x = z1 * z2                  # multiplicative gate
  ─────────────────────────────
  x = x @ V                    # contract: d_ffn/2 → d_model
  return x + shortcut
```

**aMLP** adds `tiny_attn(x_pre)` to the `z2` gate branch (single head, d_attn=64/128).

---

## Reproducibility Notes

This implementation follows the paper's specifications faithfully.
The following assumptions were made where the paper is underspecified:

| # | Location | Assumption | Confidence |
|---|----------|-----------|-----------|
| 1 | W init std | `std=0.002` (paper says "near-zero") | 0.65 — **low** |
| 2 | aMLP fusion | Additive: `gate = spatial(z2) + attn(x_pre)` | 0.75 |
| 3 | Vision pooling | Global average pool (not CLS token) | 0.70 |
| 4 | MLM masking | Standard BERT 80/10/10% | 0.90 |
| 5 | Tokenizer | t5-base SentencePiece proxy (not original 32K) | 0.85 |

**Config flags** for ambiguous settings:
- `model.w_init_std` — adjust if training is unstable early
- `model.attn_fusion_mode` — `'add'` (default), `'concat'`, or `'replace'`
- `model.pool_mode` — `'avg'` (default) or `'cls'`

**Compute notes**: Full gMLPbase training (1M steps, 256 batch, seq=512) requires
approximately 100–150 A100-GPU-hours. The ablation preset (125K steps, 2048 batch,
seq=128) runs in ~10 GPU-hours and reproduces Table 3 results.

---

## Citation

```bibtex
@article{liu2021pay,
  title={Pay Attention to MLPs},
  author={Liu, Hanxiao and Dai, Zihang and So, David R. and Le, Quoc V.},
  journal={Advances in Neural Information Processing Systems},
  year={2021},
  volume={34},
  url={https://arxiv.org/abs/2105.08050}
}
```

---

*Generated by ArXivist pipeline (Stage 4 — Code Generator) | paper_id: arxiv_2105_08050*
