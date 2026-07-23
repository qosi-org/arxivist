# Architecture Plan Summary — Dropout Reduces Underfitting

**Paper ID**: arxiv_2303_01500
**Plan version**: 1
**Framework**: PyTorch 2.1+, timm (Vision Transformers)

---

## Framework decisions

| Decision | Choice | Reason |
|---|---|---|
| Primary framework | PyTorch | Standard for transformer training; native dropout support |
| Model suite | ViT-T/S/B | Canonical models from paper; timm pretrained weights available |
| Validation task | ImageNet-1K top-1 accuracy | Direct match to paper evaluation |
| Dropout variants | Early, Late, Standard | Three independent implementations per paper Algorithms |

---

## Key insights

- **Early dropout**: Applied epochs 0–50 (configurable), then disabled. Improves training fit.
- **Late dropout**: Disabled epochs 0–50, then applied at rate p for remaining epochs. Prevents overfitting.
- **Gradient metrics**: Gradient Direction Variance (GDV) and Gradient Direction Error (GDE) measure consistency.

## Hyperparameters

```yaml
early_dropout:
  epochs: 50
  rate: 0.1
  
late_dropout:
  start_epoch: 50
  rate: 0.1
  
batch_size: 4096
base_lr: 0.001
optimizer: sgd_momentum
warmup_epochs: 20
```

## Risk assessment

| Risk | Severity | Mitigation |
|---|---|---|
| Dropout scheduling implementation | Low | Follow paper Section 5.2 exactly |
| ViT architecture variance | Medium | Use timm reference for consistency |
| Large batch size memory | Medium | Gradient accumulation if needed |

Overall SIR confidence: 0.92 — no human review required.
