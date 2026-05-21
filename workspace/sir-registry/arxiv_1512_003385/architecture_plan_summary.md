# Architecture Plan — ResNet for CIFAR-10 (arXiv:1512.03385)

**Scope (user-specified):** CIFAR-10 only. Variants: ResNet-20, ResNet-32, ResNet-44, ResNet-56, ResNet-110.
ImageNet bottleneck blocks and the 7×7 stem are out of scope.

## Framework

- **PyTorch ≥ 2.0** (Python 3.10+). CUDA optional — the network is small enough to train on CPU
  for smoke tests and on a single consumer GPU for real runs (ResNet-20 ≈ 20 min on a 4090).
- **YAML** for config (PyYAML). No Hydra — keeps dependencies thin.
- **torchvision** for the CIFAR-10 download. No HuggingFace.

## Module layout

```
paper-repos/arxiv_1512_003385/
├── src/resnet_cifar/
│   ├── models/
│   │   ├── resnet.py       # BasicBlock, IdentityPadShortcut, ResNetCIFAR
│   │   └── factory.py      # name -> ResNetCIFAR
│   ├── data/
│   │   ├── cifar10.py      # CIFAR10DataModule
│   │   └── transforms.py   # PerPixelMeanSubtract + paper augmentations
│   ├── training/
│   │   ├── trainer.py      # SGD trainer, iteration-based loop
│   │   └── schedule.py     # StepLRWithWarmup (drops at 32k/48k; warmup for ResNet-110)
│   ├── evaluation/
│   │   └── metrics.py      # AccuracyMeter
│   └── utils/
│       ├── config.py       # YAML loader + CLI override merging
│       └── seed.py         # Deterministic seeding
├── configs/                # One YAML per variant
├── data/                   # Dataset download scripts
├── docker/Dockerfile
├── notebooks/              # Stage 5 output
├── train.py                # CLI entrypoint
├── evaluate.py
├── inference.py
├── requirements.txt
└── README.md
```

## Forward pass (ResNetCIFAR)

```
image [B,3,32,32]
  → Conv3x3(3→16) → BN → ReLU                       [B,16,32,32]
  → stage1: n × BasicBlock(16→16, stride=1)          [B,16,32,32]
  → stage2: BasicBlock(16→32, stride=2) + (n-1)×     [B,32,16,16]
  → stage3: BasicBlock(32→64, stride=2) + (n-1)×     [B,64,8,8]
  → GAP → flatten                                    [B,64]
  → Linear(64→10)                                    [B,10]
```

## Training recipe (paper Section 4.2)

| Item | Value | Source / Confidence |
|---|---|---|
| Optimizer | SGD, momentum=0.9 | Paper Sec. 4.2 / 0.95 |
| Weight decay | 1e-4 (conv/FC only, not BN) | Paper / 0.95; BN-exclusion is convention (conf 0.70) |
| LR | 0.1 initial, ÷10 at iter 32k and 48k | Paper / 0.95 |
| ResNet-110 warmup | lr=0.01 for first 400 iters | Paper / 0.95 |
| Batch size | 128 | Paper / 0.95 |
| Total iterations | 64,000 (≈164 epochs on 50k train) | Paper / 0.95 |
| Init | He / Kaiming, fan_in, relu nonlinearity | Implicit (cites He 2015) / 0.90 |
| Augmentation | per-pixel mean sub; 4-px pad + random crop; random hflip | Paper / 0.85 |
| Loss | Cross-entropy | Standard / 0.90 |

## Open assumptions (config-exposed)

1. **Per-pixel vs per-channel mean subtraction** — paper says "per-pixel"; expose
   `data.mean_subtraction` flag. Default: per-pixel.
2. **Weight decay on BN params** — excluded by default (convention).
3. **Shortcut option** — A (zero-pad identity) by default per paper; B (1×1 projection) available
   as a config flag for ablations.

## Entrypoints

- `train.py --config configs/resnet20.yaml` — train + evaluate end-to-end.
- `evaluate.py --checkpoint <path>` — evaluate a saved model on the CIFAR-10 test set.
- `inference.py --checkpoint <path> --image <path>` — single-image inference.

## Risk register (top 3)

1. **Per-pixel mean ambiguity (Medium)** — exposed via config flag.
2. **BN weight decay ambiguity (Medium)** — split optimizer param groups so BN gets wd=0.
3. **ResNet-110 needs LR warmup (Low)** — auto-enabled when `model.name == 'resnet110'`.

Full risk register and confidence scores are in `architecture_plan.json`.
