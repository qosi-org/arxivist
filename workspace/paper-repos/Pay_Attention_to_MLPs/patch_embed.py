"""
gmlp/models/patch_embed.py
--------------------------
ViT-style patch embedding stem for vision gMLP.

Paper Section 3: "Pay Attention to MLPs" (arXiv:2105.08050)

The input/output protocol follows ViT/B16 (Dosovitskiy et al., 2021):
  - Raw image [B, 3, H, W] is split into non-overlapping 16×16 patches
  - Each patch is linearly projected to d_model
  - No positional encoding is added (gMLP captures position in spatial W)

For a 224×224 image with patch_size=16:
  num_patches = (224/16)^2 = 196

Note: Unlike ViT, gMLP does NOT add a CLS token or positional embeddings
before the block stack. The paper explicitly states gMLPs do not require
positional encodings (Section 2, Figure 1 caption).

Paper ref: Section 3, SIR architecture.input_protocols.Vision
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class PatchEmbedding(nn.Module):
    """
    Converts an image into a sequence of patch tokens via Conv2d projection.

    Equivalent to ViT's patch embedding:
      Conv2d(in_channels=3, out_channels=d_model, kernel_size=patch_size, stride=patch_size)
      → flatten spatial dims [B, d_model, H/P, W/P] → [B, num_patches, d_model]

    No positional encoding is added. No CLS token is prepended.
    (gMLP vision uses global average pooling for classification head,
     not a CLS token — see ambiguity_003 in SIR; pool_mode is configurable.)

    Args:
        img_size:    Input image resolution (square). Paper: 224.
        patch_size:  Patch size (square). Paper: 16.
        in_channels: Image channels. Paper: 3 (RGB).
        d_model:     Output embedding dimension per patch.

    Paper ref: Section 3, Table 1 footnote, SIR architecture.input_protocols.Vision
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        d_model: int = 256,
    ) -> None:
        super().__init__()
        assert img_size % patch_size == 0, (
            f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"
        )
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 196 for 224/16

        # Single Conv2d with kernel=stride=patch_size implements patch tokenisation
        # This is equivalent to cutting the image into patches and applying a linear projection
        self.proj = nn.Conv2d(
            in_channels, d_model,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor of shape [B, 3, H, W]

        Returns:
            Tensor of shape [B, num_patches, d_model]
        """
        assert x.dim() == 4, f"[PatchEmbedding] Expected [B, C, H, W], got {x.shape}"
        B, C, H, W = x.shape
        assert H == W == self.img_size, (
            f"[PatchEmbedding] Input resolution {H}×{W} != expected {self.img_size}×{self.img_size}"
        )

        # [B, d_model, H/P, W/P]
        x = self.proj(x)
        # Flatten spatial dims and transpose: [B, d_model, n] → [B, n, d_model]
        x = x.flatten(2).transpose(1, 2)   # [B, num_patches, d_model]
        return x

    def __repr__(self) -> str:
        return (
            f"PatchEmbedding(img_size={self.img_size}, patch_size={self.patch_size}, "
            f"num_patches={self.num_patches})"
        )
