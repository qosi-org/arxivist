# Domain: AI — Code Conventions (Stage 4 Enrichment)

Load this file alongside `agents/04_code_generator.md` when the detected domain is **AI**.

---

## Module implementation standards

**Every `nn.Module` must:**
- Accept a config dataclass or dict in `__init__`, never raw positional hyperparameters
- Assert input tensor shapes in `forward()` with descriptive messages
- Have a `__repr__` that shows key dimensions

**Attention implementation:**
- Always implement a standalone `ScaledDotProductAttention` class, even if PyTorch 2.0+
  `F.scaled_dot_product_attention` is used — wrap it for clarity and fallback support
- Causal mask must be registered as a buffer, not recomputed every forward pass
- KV-cache support: include `past_key_values` parameter in decoder attention with `None` default

**Positional encodings:**
- Register sinusoidal encodings as non-trainable buffers
- Learned embeddings: use `nn.Embedding` with explicit `max_position_embeddings` config field

---

## Training loop requirements

Beyond the generic trainer, AI papers require:

```python
# Gradient clipping — always implement, config-driven
if config.training.grad_clip > 0:
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)

# Learning rate logging — log lr every step, not just loss
scheduler.get_last_lr()[0]  # log this

# Parameter count — always print at start of training
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters: {total:,} total, {trainable:,} trainable")

# Checkpoint saving — save both model state dict AND optimiser state dict
torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "step": step,
    "config": config,
}, checkpoint_path)
```

**Mixed precision:** always implement with `torch.amp.autocast` and `GradScaler`.
Make it config-driven (`training.mixed_precision: "bf16"` | `"fp16"` | `"no"`).

---

## Data pipeline standards

**For text data:**
- Use `datasets` library (HuggingFace) for dataset loading where possible
- Tokenisation must be deterministic and reproducible — always save tokeniser config with checkpoint
- Always implement `--max_seq_len` truncation in the dataset class

**For image data:**
- Use `torchvision.transforms.v2` (not deprecated v1)
- Normalisation constants must come from config, not hardcoded — annotate if assumed (e.g. ImageNet mean/std)
- Always implement both train transforms and eval transforms separately

**DataLoader settings:**
```python
DataLoader(
    dataset,
    batch_size=config.training.batch_size,
    num_workers=config.hardware.num_workers,
    pin_memory=config.hardware.device == "cuda",
    persistent_workers=config.hardware.num_workers > 0,
    prefetch_factor=2 if config.hardware.num_workers > 0 else None,
)
```

---

## What Must NOT be done in AI papers

- Do NOT hardcode `torch.float16` — use config-driven dtype
- Do NOT use `model.cuda()` — use `model.to(device)` with device from config
- Do NOT implement custom CUDA kernels unless paper explicitly provides them
- Do NOT import `from transformers import *` — always import specific classes
