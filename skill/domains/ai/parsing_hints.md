# Domain: AI — Parsing Hints (Stage 1 Enrichment)

Load this file alongside `agents/01_paper_parser.md` when the detected domain is **AI**.
These instructions extend and override the generic parser for deep learning and foundation model papers.

---

## Architecture extraction — AI-specific rules

**Naming conventions to recognise:**
- "Block", "Layer", "Head", "Tower", "Branch", "Stream" — always separate named modules
- "Base / Large / XL / XXL / 7B / 13B / 70B" — always extract all scale variants
- Subscript numbers in architecture names (e.g. ViT-L/16) encode patch size or depth — extract both
- "Backbone" refers to a feature extractor — extract its family (ResNet, ViT, ConvNeXt) and size

**Figures are primary sources in AI papers.** Architecture figures often contain shape/channel
information not present in the text. Extract:
- All dimension labels on arrows or tensor flow diagrams
- All block counts shown in repeated-module diagrams (e.g. "×12" means 12 identical layers)
- Colour-coded components that indicate weight sharing or separate training stages

**Attention variants — extract the specific type:**
- Self-attention vs cross-attention vs causal (masked) attention
- Multi-head vs multi-query vs grouped-query attention (MQA / GQA)
- Flash attention, linear attention, sparse attention — note if mentioned
- Sliding window, local+global hybrids

**Positional encoding — always extract explicitly:**
- Absolute (learned vs sinusoidal), RoPE, ALiBi, NoPE, relative, 2D/3D variants
- If not stated, mark as `null` and add to ambiguities — it materially affects reproducibility

**Normalisation — always extract:**
- LayerNorm, RMSNorm, BatchNorm, GroupNorm — location matters (pre-norm vs post-norm)
- Pre-norm vs post-norm is often described visually only — extract from figure if text is silent

---

## Mathematical spec — AI-specific rules

**Always extract:**
- Attention score equation (exact scaling factor)
- Loss function with all terms (e.g. auxiliary losses in MoE, VQ losses, KL terms)
- Any regularisation terms (weight decay form, dropout placement)
- Sampling procedure for generative models (temperature, top-k, top-p, CFG scale)

**Diffusion papers:** extract both the forward (noising) and reverse (denoising) process equations.
Extract the noise schedule type (linear, cosine, sigmoid) and its parameters explicitly.

**Contrastive learning papers:** extract the NT-Xent / InfoNCE / SimCLR loss with temperature τ.

---

## Training pipeline — AI-specific rules

**Always look for and extract:**
- Warmup steps / warmup ratio (critical for transformer training stability)
- Gradient clipping value (almost always present but often in appendix)
- Weight decay value and whether it applies to all parameters or excludes bias/norm layers
- Mixed precision type: fp16 vs bf16 (bf16 is now dominant — default to bf16 if unspecified and
  paper uses modern hardware, flag as assumed)
- Gradient accumulation steps (often unlisted — compute from effective vs per-GPU batch size)
- EMA (exponential moving average) coefficient if used
- Number of GPUs / TPUs and whether data-parallel or model-parallel

**Pretraining vs fine-tuning:** if the paper fine-tunes a pretrained model, extract:
- The base model name and checkpoint source
- Which parameters are frozen vs updated
- Whether a different learning rate is used for different parameter groups

---

## Evaluation — AI-specific rules

**Classification papers:** extract top-1 and top-5 accuracy separately.
**Generation papers:** extract all of FID, IS, CLIP score, human eval scores if present.
**Language model papers:** extract perplexity on the specific test set (PTB, WikiText-103, etc.)
  with exact tokenisation method — perplexity is tokeniser-dependent.
**Multimodal papers:** extract each modality's metric separately.

**Always note:** whether evaluation uses greedy, beam search, or sampling, and with what parameters.

---

## Data pitfalls — load `data_pitfalls.md` alongside this file for Stage 1.
