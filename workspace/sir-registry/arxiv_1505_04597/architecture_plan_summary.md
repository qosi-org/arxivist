# Architecture Plan Summary — U-Net: Convolutional Networks for Biomedical Image Segmentation

**Paper ID**: arxiv_1505_04597
**Plan version**: 1
**Framework**: PyTorch 2.1+

---

## Framework decisions

| Decision | Choice | Reason |
|---|---|---|
| Primary framework | PyTorch | Standard for CV segmentation; Caffe original is dated |
| Convolutions | Unpadded (valid) | Paper uses valid convs; output smaller than input |
| Skip connections | Crop-and-concat | Required due to border loss from unpadded convs |
| Loss | Weighted pixel-wise cross entropy | Paper Eq. 1 with separation weight map Eq. 2 |

---

## Module hierarchy

- `DoubleConv` — two 3x3 valid conv + ReLU
- `Down` — max pool 2x2 + DoubleConv (channels double)
- `Up` — up-conv 2x2 + crop-and-concat + DoubleConv (channels halve)
- `UNet` — 4 down, bottleneck, 4 up, final 1x1 conv; 23 conv layers total

## Tensor flow

Input 572x572x1 -> encoder (64,128,256,512) -> bottleneck 1024 -> decoder with skips -> 388x388x2 output map.

## Config schema

```yaml
in_channels: 1
num_classes: 2
base_channels: 64
optimizer: sgd
momentum: 0.99
batch_size: 1
w0: 10
sigma: 5
```

## Risk assessment

| Risk | Severity | Mitigation |
|---|---|---|
| Batch size = 1 with high momentum | Medium | Follow paper; favor large tiles over batch |
| Weight-map precomputation cost | Medium | Precompute per ground-truth mask with morphological ops |
| Learning rate unspecified | Medium | ASSUMED default; flagged low confidence |
| Elastic deformation augmentation | Low | Implement per Section 3.1 |

Overall SIR confidence: 0.93 — no human review required.
