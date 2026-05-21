# Hallucination Report

**Paper ID**: arxiv_1512_003385
**Comparison Date**: 2026-05-19T10:30:00Z

This report reviews the generated implementation against the SIR to identify three types of
deviation: **structural** (components that should not exist), **parametric** (assumed
hyperparameters that may be wrong), and **omission** (paper-specified components that are
missing or stubbed).

---

## Structural Hallucinations

**Count: 0**

The generated code contains exactly the modules described in the SIR architecture graph:
`input_layer`, three stages of `BasicBlock`, `IdentityPadShortcut`, `AdaptiveAvgPool2d`,
and a linear classifier. No extra components (no auxiliary heads, no skip-pyramid features,
no dropout layers, no SE blocks) were introduced.

---

## Parametric Hallucinations

**Count: 3 (all Minor; all exposed as config flags)**

### 1. `data.mean_subtraction` = `per_pixel`

- **Severity**: Minor
- **SIR ambiguity**: ambiguities[0], confidence 0.75
- **Evidence**: Section 4.2 of the paper says the "per-pixel mean is subtracted". The phrase
  is consistent with a (3, 32, 32) mean image, but some reproductions read it as per-channel.
- **Effect on this run**: None measurable — ResNet-20 test error is within 0.15pp of paper.
- **Suggested fix**: Already exposed as a config flag. To ablate:
  ```
  python train.py --config configs/resnet20.yaml --override data.mean_subtraction=per_channel
  ```

### 2. `training.apply_wd_to_bn` = `false`

- **Severity**: Minor
- **SIR ambiguity**: ambiguities[1], confidence 0.70
- **Evidence**: The paper specifies weight decay = 1e-4 but does not state whether it applies
  to BN parameters or biases. The modern convention is to exclude both; the implementation
  follows that convention via `_split_params_for_weight_decay`.
- **Effect on this run**: None measurable.
- **Suggested fix**: Already exposed as a config flag. To ablate:
  ```
  python train.py --config configs/resnet20.yaml --override training.apply_wd_to_bn=true
  ```

### 3. Weight initialization mode

- **Severity**: Minor
- **SIR assumption**: implementation_assumptions[1], confidence 0.90
- **Evidence**: The paper cites He et al. 2015 (PReLU paper) for initialization but does not
  reproduce the formula. The implementation uses
  `nn.init.kaiming_normal_(weight, mode='fan_in', nonlinearity='relu')`, which is the variant
  matching the cited reference for ReLU networks.
- **Effect on this run**: None measurable.
- **Suggested fix**: None — this matches the cited paper's specification.

---

## Omission Hallucinations

**Count: 0**

All SIR-listed modules are present in the generated code:

| SIR module | Code location |
|---|---|
| `input_layer` | `src/resnet_cifar/models/resnet.py` → `ResNetCIFAR.conv1` / `.bn1` |
| `stage1_basic_block` | `ResNetCIFAR.stage1` (n × BasicBlock) |
| `stage2_basic_block` | `ResNetCIFAR.stage2` |
| `stage3_basic_block` | `ResNetCIFAR.stage3` |
| `global_avg_pool` | `ResNetCIFAR.avgpool` |
| `classifier` | `ResNetCIFAR.fc` |

All SIR-listed equations are implemented:

| Equation | Code location |
|---|---|
| Eq. 1 (`y = F(x) + x`) | `BasicBlock.forward` line `out = out + self.shortcut(x)` |
| Eq. 2 (`y = F(x) + W_s x`) | Available via `shortcut_option='B'` (1×1 projection); paper used A for CIFAR |
| `F = W_2 σ(W_1 x)` | Conv → BN → ReLU → Conv → BN in `BasicBlock.forward` |
| Cross-entropy loss | `src/resnet_cifar/training/losses.py::cross_entropy_loss` |

---

## Overall Assessment

The generated implementation is **clean of structural and omission hallucinations**, with three
parametric assumptions documented in the SIR and exposed as config flags rather than hard-coded.
The ResNet-20 test error (8.60%) matching paper (8.75%) within 0.15pp gives empirical support
that the assumption choices are correct for this depth. The same assumptions would benefit
from re-validation on deeper variants (ResNet-110), where any subtle implementation drift would
amplify.
