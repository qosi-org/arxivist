"""
models/lstm_encoder.py — LSTM encoder for macroeconomic time series.

Implements the Recurrent Neural Network with Long-Short-Term-Memory (LSTM) cells
described in Section III.C. The LSTM summarizes the dynamics of 178 macroeconomic
time series into a small number of hidden state processes.

Key insight from paper (Section III.C): PCA on macroeconomic variables fails here
because it operates on increments and loses dynamic/cyclical information. The LSTM
finds the appropriate stationary transformation while capturing long-term dependencies
like business cycles.

Paper: Deep Learning in Asset Pricing, Chen, Pelger & Zhu (2019), Section III.C.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class MacroLSTMEncoder(nn.Module):
    """
    LSTM encoder that maps macroeconomic time series to hidden state processes.

    From Section III.C: Instead of directly passing macroeconomic variables I_t
    as features to the feedforward network, we apply a non-linear transformation
    with a Recurrent Neural Network with LSTM cells. The output h_t contains all
    macroeconomic information in the past, while I_t only uses current information.

    The LSTM handles:
    1. Non-stationarity in macro series (learns appropriate transformations)
    2. Long-term dependencies (business cycles) via gating mechanism
    3. Dimensionality reduction: 178 series → num_states hidden states

    LSTM equations (Section III.C, Appendix A.2):
        c̃_t = tanh(W_h^(c) h_{t-1} + W_x^(c) x_t + w_0^(c))
        input_t = σ(W_h^(i) h_{t-1} + W_x^(i) x_t + w_0^(i))
        forget_t = σ(W_h^(f) h_{t-1} + W_x^(f) x_t + w_0^(f))
        out_t = σ(W_h^(o) h_{t-1} + W_x^(o) x_t + w_0^(o))
        c_t = forget_t ⊙ c_{t-1} + input_t ⊙ c̃_t
        h_t = out_t ⊙ tanh(c_t)

    Args:
        input_size: number of macroeconomic variables (178 in paper)
        num_states: number of hidden state processes to output
            - SDF network: 4 states (SMV=4, Table I)
            - Conditional network: 32 states (CSMV=32, Table I)
        lstm_hidden_size: internal LSTM hidden dimension
            # WARNING: low-confidence implementation (conf=0.65)
            # TODO: paper only specifies output state count, not internal hidden size.
            # Defaulting to num_states. May need tuning.
        dropout: dropout rate applied to output (matches global DR=0.95 from Table I)
    """

    def __init__(
        self,
        input_size: int = 178,
        num_states: int = 4,
        lstm_hidden_size: Optional[int] = None,
        dropout: float = 0.95,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.num_states = num_states
        # WARNING: low-confidence (conf=0.65) — hidden size assumed = num_states
        self.lstm_hidden_size = lstm_hidden_size if lstm_hidden_size is not None else num_states

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=1,
            batch_first=True,  # input: [batch, T, input_size]
        )

        # Project to num_states if internal size differs
        if self.lstm_hidden_size != num_states:
            self.projection = nn.Linear(self.lstm_hidden_size, num_states)
        else:
            self.projection = None

        # Dropout on output (paper uses dropout throughout, Section III.E)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode macroeconomic time series to hidden state processes.

        Note: output h_t is Ft-measurable (no look-ahead bias), as stated in Section III.C.
        The transformation uses history up to t: contains all past macro information.

        Args:
            x: [batch, T, input_size] macro time series (178 variables)
            hidden: optional initial (h_0, c_0) hidden state tuple

        Returns:
            states: [batch, T, num_states] hidden state processes h_t at each time step
            hidden: final (h_T, c_T) hidden state tuple for continuing sequence
        """
        assert x.dim() == 3, f"Expected [batch, T, input_size], got {x.shape}"
        assert x.shape[-1] == self.input_size, (
            f"Expected input_size={self.input_size}, got {x.shape[-1]}"
        )

        # LSTM forward: [batch, T, lstm_hidden_size]
        lstm_out, hidden_out = self.lstm(x, hidden)

        # Project to num_states if needed
        if self.projection is not None:
            states = self.projection(lstm_out)  # [batch, T, num_states]
        else:
            states = lstm_out

        states = self.dropout(states)
        return states, hidden_out

    def __repr__(self) -> str:
        return (
            f"MacroLSTMEncoder(input_size={self.input_size}, "
            f"num_states={self.num_states}, "
            f"lstm_hidden_size={self.lstm_hidden_size})"
        )
