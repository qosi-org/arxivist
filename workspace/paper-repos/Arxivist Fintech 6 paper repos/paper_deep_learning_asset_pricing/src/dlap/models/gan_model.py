"""
models/gan_model.py — GAN Asset Pricing Model (top-level orchestrator).

Combines all sub-networks into the full GAN asset pricing framework from
Section III. The model has two branches:
  1. SDF branch: LSTM (macro→h_t) + FFN (h_t, chars→omega) → SDF factor
  2. Conditional branch: LSTM (macro→h_t^g) + FFN (h_t^g, chars→g) → instruments

Training follows the 3-step procedure from Section III.D:
  Step 1: Initialize — minimize unconditional loss (g=constant)
  Step 2: Fix SDF params — maximize loss over conditional network (adversary)
  Step 3: Fix conditional params — minimize conditional loss (SDF network update)

Paper: Deep Learning in Asset Pricing, Chen, Pelger & Zhu (2019), Section III.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from dlap.models.lstm_encoder import MacroLSTMEncoder
from dlap.models.sdf_network import SDFNetwork, LoadingNetwork
from dlap.models.conditional_network import ConditionalNetwork


class GANAssetPricingModel(nn.Module):
    """
    Full GAN asset pricing model as described in Section III.

    Architecture summary:
        StateMacroRNN: 178 macro series → 4 hidden states h_t (LSTM)
        SDFNetwork:    [h_t (4), chars (46)] → omega_{t,i} (FFN, 2 layers, 64 units)
        ConditionalMacroRNN: 178 macro series → 32 hidden states h_t^g (LSTM)
        ConditionalNetwork:  [h_t^g (32), chars (46)] → g_{t,j} (linear, 8 moments)
        LoadingNetwork: [h_t (4), chars (46)] → beta_{t,i} (FFN, same as SDF)

    SDF construction:
        F_{t+1} = omega_t^T R^e_{t+1}    (SDF factor / tangency portfolio)
        M_{t+1} = 1 - F_{t+1}             (stochastic discount factor)

    Args:
        cfg: configuration dictionary (from config.yaml model section)
    """

    def __init__(self, cfg: Dict) -> None:
        super().__init__()
        m = cfg["model"]

        self.num_macro_vars = m["num_macro_vars"]
        self.num_firm_chars = m["num_firm_chars"]

        # SDF branch — LSTM encoder for h_t
        self.sdf_macro_rnn = MacroLSTMEncoder(
            input_size=m["num_macro_vars"],
            num_states=m["sdf_macro_states"],
            lstm_hidden_size=m.get("lstm_hidden_size", m["sdf_macro_states"]),
            dropout=cfg["training"]["dropout"],
        )

        # SDF portfolio weight network: omega(h_t, I_{t,i})
        self.sdf_network = SDFNetwork(
            macro_state_dim=m["sdf_macro_states"],
            firm_char_dim=m["num_firm_chars"],
            num_layers=m["sdf_num_layers"],
            hidden_units=m["sdf_hidden_units"],
            dropout=cfg["training"]["dropout"],
        )

        # Conditional (adversarial) branch — separate LSTM for h_t^g
        self.cond_macro_rnn = MacroLSTMEncoder(
            input_size=m["num_macro_vars"],
            num_states=m["cond_macro_states"],
            lstm_hidden_size=m.get("cond_lstm_hidden_size", m["cond_macro_states"]),
            dropout=cfg["training"]["dropout"],
        )

        # Adversarial conditioning function: g(h_t^g, I_{t,j})
        self.cond_network = ConditionalNetwork(
            macro_state_dim=m["cond_macro_states"],
            firm_char_dim=m["num_firm_chars"],
            num_moments=m["cond_hidden_units"],
            num_layers=m["cond_num_layers"],
            hidden_units=m["cond_hidden_units"],
            dropout=cfg["training"]["dropout"],
        )

        # Loading network: beta(h_t, I_{t,i}) ∝ E_t[R^e * F_{t+1}]
        self.loading_network = LoadingNetwork(
            macro_state_dim=m["sdf_macro_states"],
            firm_char_dim=m["num_firm_chars"],
            num_layers=m["sdf_num_layers"],
            hidden_units=m["sdf_hidden_units"],
            dropout=cfg["training"]["dropout"],
        )

    def forward_sdf(
        self,
        macro_series: torch.Tensor,
        firm_chars: torch.Tensor,
        returns: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute SDF weights, SDF factor, and SDF values.

        Args:
            macro_series: [batch, T, 178] macroeconomic time series
            firm_chars:   [T, N, 46] firm characteristics (quantile-normalized)
            returns:      [T, N] excess returns R^e_{t+1,i}

        Returns:
            omega: [T, N] SDF portfolio weights
            F_t:   [T] SDF factor returns (tangency portfolio)
            M_t:   [T] SDF values M_{t+1} = 1 - F_{t+1}
            h_t:   [T, 4] macro hidden states
        """
        assert macro_series.dim() == 3, f"Expected [batch, T, 178], got {macro_series.shape}"

        # Encode macro series to hidden states h_t: [batch, T, 4]
        h_t_batch, _ = self.sdf_macro_rnn(macro_series)
        h_t = h_t_batch.squeeze(0)  # [T, 4] (single batch for panel data)

        # Compute SDF weights: omega_{t,i} = omega(h_t, I_{t,i})
        omega = self.sdf_network(h_t, firm_chars)  # [T, N]

        # Construct SDF factor: F_{t+1} = omega_t^T R^e_{t+1}  (Section II.A)
        F_t = (omega * returns).sum(dim=-1)  # [T]

        # SDF: M_{t+1} = 1 - omega_t^T R^e_{t+1}  (Section II.A, eq_sdf_definition)
        M_t = 1.0 - F_t  # [T]

        return omega, F_t, M_t, h_t

    def forward_conditional(
        self,
        macro_series: torch.Tensor,
        firm_chars: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute adversarial conditioning instruments g_{t,j}.

        Args:
            macro_series: [batch, T, 178]
            firm_chars:   [T, N, 46]

        Returns:
            g:     [T, N, 8] conditioning instruments
            h_t_g: [T, 32] conditional macro hidden states
        """
        h_t_g_batch, _ = self.cond_macro_rnn(macro_series)
        h_t_g = h_t_g_batch.squeeze(0)  # [T, 32]
        g = self.cond_network(h_t_g, firm_chars)  # [T, N, 8]
        return g, h_t_g

    def forward_loadings(
        self,
        h_t: torch.Tensor,
        firm_chars: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate risk loadings beta_{t,i}.

        Args:
            h_t: [T, 4] macro hidden states (from forward_sdf)
            firm_chars: [T, N, 46]

        Returns:
            beta: [T, N] risk loading estimates proportional to E_t[R^e * F_{t+1}]
        """
        return self.loading_network(h_t, firm_chars)

    def sdf_parameters(self):
        """Parameters for SDF network update (Step 1 and 3 of training)."""
        return (
            list(self.sdf_macro_rnn.parameters()) +
            list(self.sdf_network.parameters())
        )

    def conditional_parameters(self):
        """Parameters for conditional network update (Step 2 of training — adversary)."""
        return (
            list(self.cond_macro_rnn.parameters()) +
            list(self.cond_network.parameters())
        )

    def loading_parameters(self):
        """Parameters for loading network (trained separately)."""
        return list(self.loading_network.parameters())

    def __repr__(self) -> str:
        return (
            f"GANAssetPricingModel(\n"
            f"  sdf_macro_rnn={self.sdf_macro_rnn}\n"
            f"  sdf_network={self.sdf_network}\n"
            f"  cond_macro_rnn={self.cond_macro_rnn}\n"
            f"  cond_network={self.cond_network}\n"
            f"  loading_network={self.loading_network}\n"
            f")"
        )
