# Domain: AI — Architecture Patterns (Stage 3 Enrichment)

Load this file alongside `agents/03_architecture_planner.md` when the detected domain is **AI**.

---

## Standard module implementations

When the SIR names a module without full implementation detail, use these canonical patterns:

**Transformer Encoder Layer**
```
MultiHeadAttention → Dropout → Residual → LayerNorm
PositionwiseFeedForward (Linear → GELU/ReLU → Dropout → Linear) → Dropout → Residual → LayerNorm
```
Pre-norm variant: LayerNorm applied before the sub-layer, not after.

**Causal Transformer Decoder Layer**
Same as encoder but attention uses a causal mask. Cross-attention is inserted between
self-attention and FFN if the model is encoder-decoder.

**Vision Transformer (ViT) Block**
Patch embedding (Conv2d with kernel=patch_size, stride=patch_size) → flatten → linear projection
→ prepend CLS token → add positional embedding → standard transformer encoder stack.

**Diffusion UNet**
Typical structure: encoder path (ResBlock + Attention + Downsample) → bottleneck (ResBlock +
Attention) → decoder path (ResBlock + Attention + Upsample + skip connections).
Time conditioning via sinusoidal timestep embedding projected into each ResBlock via AdaGN or
addition after a linear layer.

**MoE (Mixture of Experts)**
Router (linear layer + softmax) selects top-k experts per token. Each expert is an independent
FFN. Load balancing loss is auxiliary — always implement it even if confidence is low.

**Convolutional Backbone families:**
- ResNet: `Conv → BN → ReLU` in residual blocks, bottleneck for ResNet-50+
- ConvNeXt: depthwise conv → LayerNorm → Linear → GELU → Linear (inverted bottleneck)
- EfficientNet: MBConv with squeeze-excitation

---

## Framework and library defaults for AI papers

**Default to PyTorch** unless paper specifies JAX/Flax (common for Google papers) or TensorFlow.

**HuggingFace Transformers** — use when:
- Paper is about a language model and the architecture matches a known HF model class
- Paper explicitly mentions using HF
- Base model is a known pretrained checkpoint (BERT, GPT-2, LLaMA, etc.)

**timm** — use for vision backbone implementations (ResNet, ViT, ConvNeXt, Swin).

**diffusers** — use for diffusion model implementations.

**Recommend `torch.compile`** in config as an opt-in flag for PyTorch 2.0+ speed gains.

---

## Config schema additions for AI

Always add these fields beyond the generic config:

```yaml
model:
  use_flash_attention: false    # opt-in — requires flash-attn package
  dtype: bfloat16               # or float32 — match paper's hardware era
  gradient_checkpointing: false # opt-in for memory-constrained runs

training:
  ema_decay: null               # set if paper uses EMA
  warmup_ratio: 0.1             # if warmup_steps not specified
  weight_decay_exclude: ["bias", "norm"]  # standard exclusion pattern

generation:                     # for generative models only
  temperature: 1.0
  top_k: null
  top_p: null
  cfg_scale: null               # classifier-free guidance scale
```

---

## Entrypoints for AI papers

Add beyond the generic set:
- `generate.py` — for generative models (text, image, audio)
- `pretrain.py` — if paper distinguishes pretraining from fine-tuning
- `finetune.py` — with support for LoRA / adapter flags if applicable
