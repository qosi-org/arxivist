"""ResNet for CIFAR-10 — implements He et al. 2015 (arXiv:1512.03385), Section 4.2.

Architecture (depth = 6n + 2):
    image [B,3,32,32]
      -> Conv3x3(3->16) -> BN -> ReLU                       [B,16,32,32]
      -> stage1: n x BasicBlock(16->16, stride=1)           [B,16,32,32]
      -> stage2: BasicBlock(16->32, stride=2) + (n-1)x      [B,32,16,16]
      -> stage3: BasicBlock(32->64, stride=2) + (n-1)x      [B,64,8,8]
      -> GAP -> flatten                                     [B,64]
      -> Linear(64->num_classes)                            [B,num_classes]

Shortcuts use Option A (identity with zero-padded extra channels) by default, matching the
paper's CIFAR experiments. Option B (1x1 projection) is available via `shortcut_option='B'`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityPadShortcut(nn.Module):
    """Option A shortcut: stride-2 downsampling via slicing + zero-padded extra channels.

    Used when a block changes spatial resolution and/or filter count. Parameter-free, matching
    the paper's stated preference for CIFAR experiments (Sec. 4.2: "all shortcuts are identity").
    """

    def __init__(self, in_planes: int, out_planes: int, stride: int) -> None:
        super().__init__()
        if out_planes < in_planes:
            raise ValueError(
                f"IdentityPadShortcut requires out_planes >= in_planes, got "
                f"in_planes={in_planes}, out_planes={out_planes}"
            )
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride
        self.pad_channels = out_planes - in_planes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride > 1:
            x = x[:, :, :: self.stride, :: self.stride]
        if self.pad_channels > 0:
            # F.pad takes (last_dim_left, last_dim_right, ..., front, back) for channel dim;
            # the channel dim is dim=1, so we pad (W_l, W_r, H_l, H_r, C_l, C_r) = (0,0,0,0,0,pad).
            x = F.pad(x, (0, 0, 0, 0, 0, self.pad_channels))
        return x

    def __repr__(self) -> str:
        return (
            f"IdentityPadShortcut(in={self.in_planes}, out={self.out_planes}, "
            f"stride={self.stride})"
        )


class BasicBlock(nn.Module):
    """Two-conv residual block — paper Eq. 1:  y = F(x, {W_i}) + x.

    Implements post-activation ResNet (original 2015 paper), NOT the 2016 pre-activation variant:

        out = conv1(x) -> BN -> ReLU -> conv2 -> BN -> add(shortcut) -> ReLU
    """

    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        shortcut_option: str = "A",
    ) -> None:
        super().__init__()
        if shortcut_option not in {"A", "B"}:
            raise ValueError(f"shortcut_option must be 'A' or 'B', got {shortcut_option!r}")

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1 and in_planes == planes:
            self.shortcut: nn.Module = nn.Identity()
        elif shortcut_option == "A":
            self.shortcut = IdentityPadShortcut(in_planes, planes, stride)
        else:
            # Option B: 1x1 projection. Provided for completeness; not the paper default for CIFAR.
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4, f"Expected [B,C,H,W], got {tuple(x.shape)}"
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)  # Eq. 1: residual addition
        out = F.relu(out, inplace=True)
        return out

    def __repr__(self) -> str:
        return f"BasicBlock(in={self.in_planes}, out={self.planes}, stride={self.stride})"


class ResNetCIFAR(nn.Module):
    """CIFAR-10 ResNet with depth 6n+2 (He et al. 2015, Sec. 4.2).

    Args:
        n: number of BasicBlocks per stage. Depth = 6n + 2.
            n=3  -> ResNet-20
            n=5  -> ResNet-32
            n=7  -> ResNet-44
            n=9  -> ResNet-56
            n=18 -> ResNet-110
        num_classes: number of output classes (10 for CIFAR-10).
        shortcut_option: 'A' (paper default for CIFAR) or 'B'.
    """

    def __init__(self, n: int, num_classes: int = 10, shortcut_option: str = "A") -> None:
        super().__init__()
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        self.n = n
        self.depth = 6 * n + 2
        self.num_classes = num_classes
        self.shortcut_option = shortcut_option

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.stage1 = self._make_stage(in_planes=16, planes=16, n=n, stride=1)
        self.stage2 = self._make_stage(in_planes=16, planes=32, n=n, stride=2)
        self.stage3 = self._make_stage(in_planes=32, planes=64, n=n, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

        self._init_weights()

    def _make_stage(self, in_planes: int, planes: int, n: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (n - 1)
        blocks: list[nn.Module] = []
        ip = in_planes
        for s in strides:
            blocks.append(BasicBlock(ip, planes, stride=s, shortcut_option=self.shortcut_option))
            ip = planes
        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        """He / Kaiming init (fan_in, ReLU nonlinearity). BN: gamma=1, beta=0. FC: standard."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4 and x.shape[1] == 3, (
            f"Expected [B,3,H,W], got {tuple(x.shape)}"
        )
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        return (
            f"ResNetCIFAR(depth={self.depth}, n={self.n}, "
            f"num_classes={self.num_classes}, shortcut={self.shortcut_option}, "
            f"params={self.num_parameters():,})"
        )
