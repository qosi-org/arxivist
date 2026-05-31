"""
DeepVol: top-level model combining DCC stack, attention, and output head.
Implements the full forward pass from Eq. 22 and Eq. 27 (Section 4.2).

Reference: Moreno-Pino & Zohren, Quantitative Finance 2024.
DOI: 10.1080/14697688.2024.2387222
"""
import torch
import torch.nn as nn
from omegaconf import DictConfig

from deepvol.models.dcc_block import DCCBlock
from deepvol.models.attention import BahdanauAttention
from deepvol.models.output_head import OutputHead


class DeepVol(nn.Module):
    """
    DeepVol model: hierarchical Dilated Causal Convolutions for day-ahead
    realised volatility forecasting from raw high-frequency intraday returns.

    Architecture (Section 4.2 / Eq. 22-27):
        raw_returns [B, 1, T*J]
        → InputProjection [B, residual_channels, T*J]
        → num_blocks × num_layers DCCBlocks  (dilation d=2^l per layer)
        → SkipSum [B, skip_channels, T*J]
        → BahdanauAttention [B, skip_channels]
        → OutputHead [B, 1]
        → sigma2_hat (day-ahead realised variance)

    Args:
        cfg: OmegaConf DictConfig with model and data sub-configs
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        m = cfg.model

        seq_len = cfg.data.conditioning_range * cfg.data.intervals_per_day  # T*J

        # Input projection: scalar returns → residual channel space
        self.input_proj = nn.Conv1d(m.in_channels, m.residual_channels, kernel_size=1)

        # DCC stack: num_blocks × num_layers, dilation doubles per layer within block
        # Eq. 24-25: d = 2^l for l in 0..num_layers-1, repeated for each block
        self.dcc_blocks = nn.ModuleList()
        for _ in range(m.num_blocks):
            for layer_idx in range(m.num_layers):
                dilation = 2 ** layer_idx
                self.dcc_blocks.append(
                    DCCBlock(
                        residual_channels=m.residual_channels,
                        dilation_channels=m.dilation_channels,
                        skip_channels=m.skip_channels,
                        kernel_size=m.kernel_size,
                        dilation=dilation,
                    )
                )

        # Bahdanau attention over time dimension (Section 4.2)
        self.attention = BahdanauAttention(hidden_dim=m.skip_channels)

        # Output MLP head (Eq. 27 / Table 1: end_channels=64)
        # TODO: verify exact architecture — two-layer MLP assumed; confidence=0.65
        self.output_head = OutputHead(
            in_features=m.skip_channels,
            hidden_features=m.end_channels,
        )

        self._seq_len = seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementing Eq. 22 and Eq. 27.

        Args:
            x: [B, 1, T*J]  — raw intraday log-returns (T=conditioning_range days,
                               J=intervals_per_day intervals each)
        Returns:
            sigma2_hat: [B, 1]  — day-ahead realised variance forecast
        """
        assert x.dim() == 3, f"Expected [B, 1, L], got {x.shape}"
        assert x.shape[1] == 1, f"Expected 1 input channel, got {x.shape[1]}"

        # Project input to residual channel space
        h = self.input_proj(x)          # [B, residual_channels, L]

        # DCC stack with skip connections (Eq. 25-26)
        skip_sum = torch.zeros(
            h.shape[0], self.attention.W.in_features, h.shape[2],
            device=x.device, dtype=x.dtype
        )
        for block in self.dcc_blocks:
            h, skip = block(h)          # h: [B, residual_channels, L], skip: [B, skip_channels, L]
            skip_sum = skip_sum + skip  # accumulate skip outputs

        # Bahdanau attention collapses time dimension
        context = self.attention(skip_sum)   # [B, skip_channels]

        # Output head produces scalar volatility forecast
        sigma2_hat = self.output_head(context)   # [B, 1]
        return sigma2_hat

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return (f"DeepVol(blocks={len(self.dcc_blocks)}, "
                f"params={self.count_parameters():,})")


class DeepVolRM(nn.Module):
    """
    DeepVol-RM variant: uses precomputed realised measures from the previous
    22 days as input instead of raw intraday data (Section 5.3).
    Inspired by HAR model (Corsi 2009, Eq. 18-19).
    Same DCC architecture applied to the realised variance time series.
    """

    def __init__(self, cfg: DictConfig, rm_window: int = 22):
        super().__init__()
        m = cfg.model
        self.rm_window = rm_window

        self.input_proj = nn.Conv1d(1, m.residual_channels, kernel_size=1)

        self.dcc_blocks = nn.ModuleList()
        for _ in range(m.num_blocks):
            for layer_idx in range(m.num_layers):
                dilation = 2 ** layer_idx
                self.dcc_blocks.append(
                    DCCBlock(
                        residual_channels=m.residual_channels,
                        dilation_channels=m.dilation_channels,
                        skip_channels=m.skip_channels,
                        kernel_size=m.kernel_size,
                        dilation=dilation,
                    )
                )

        self.attention = BahdanauAttention(hidden_dim=m.skip_channels)
        self.output_head = OutputHead(
            in_features=m.skip_channels,
            hidden_features=m.end_channels,
        )

    def forward(self, rv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rv: [B, 1, rm_window]  — realised variance over past 22 days
        Returns:
            sigma2_hat: [B, 1]
        """
        assert rv.dim() == 3, f"Expected [B, 1, T], got {rv.shape}"
        h = self.input_proj(rv)
        skip_sum = torch.zeros(
            h.shape[0], self.attention.W.in_features, h.shape[2],
            device=rv.device, dtype=rv.dtype
        )
        for block in self.dcc_blocks:
            h, skip = block(h)
            skip_sum = skip_sum + skip
        context = self.attention(skip_sum)
        return self.output_head(context)

    def __repr__(self):
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"DeepVolRM(rm_window={self.rm_window}, params={n:,})"
