"""
RWKV Attention modules for Transformer architectures.

Implements RWKV-7 (Goose) architecture:
- Generalized Delta Rule for state evolution
- Token Mixing with time decay
- Channel Mixing with GLU gating
- No self-attention mechanism - linear-time architecture

Reference:
- RWKV-7 Paper: arXiv:2503.14456
- https://github.com/BlinkDL/RWKV-LM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class TimeMixing(nn.Module):
    """
    RWKV Time Mixing layer - the core innovation of RWKV.

    Combines information across tokens using a time-shift (token-shift) mechanism
    with learned time decay, enabling O(N) complexity instead of O(N^2) attention.

    Key equations (simplified from RWKV-7):
        states = time_decay * states + token_shift(x) * wkv
        output = Linear(states)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        shift_size: float = 0.5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.shift_size = shift_size

        # Project d_model to working dimension
        self.time_decay = nn.Parameter(torch.zeros(d_state))
        self.time_first = nn.Parameter(torch.zeros(d_state))

        # Key, Value, Output projections
        self.key = nn.Linear(d_model, d_state, bias=False)
        self.value = nn.Linear(d_model, d_state, bias=False)
        self.output = nn.Linear(d_state, d_model, bias=False)

        # Token shift (time-shift) parameters
        self.time_shift = nn.Linear(d_model, d_model)
        self.time_shift_gates = nn.Parameter(torch.ones(d_model))

        # Reception field extension (RWKV-7 innovation)
        self.reception_w = nn.Parameter(torch.zeros(d_model, d_state))
        self.reception_u = nn.Parameter(torch.zeros(d_model, d_state))

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def time_shift_operation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply time-shift (token-shift) mechanism.

        Shifts the input by half the sequence, blending current and previous
        token information to provide local context.
        """
        if x.size(1) <= 1:
            return x

        # Shift by half the sequence
        shift_size = int(x.size(1) * self.shift_size)
        shift_size = max(1, shift_size)

        x_cat = torch.cat([
            x[:, shift_size:, :],
            x[:, :-shift_size, :]
        ], dim=1)

        return x_cat * self.time_shift_gates.sigmoid() + x * (1 - self.time_shift_gates.sigmoid())

    def forward(
        self,
        x: torch.Tensor,
        init_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            init_states: Optional initial hidden states (batch_size, d_state)

        Returns:
            output: (batch_size, seq_len, d_model)
            last_states: (batch_size, d_state) - final hidden states for RNN-style inference
        """
        batch_size, seq_len, d_model = x.shape

        # Apply time-shift
        x_shift = self.time_shift_operation(x)
        x_norm = self.norm(x_shift)

        # Compute key and value
        k = self.key(x_norm)  # (batch, seq_len, d_state)
        v = self.value(x_norm)  # (batch, seq_len, d_state)

        # Initialize or load states
        if init_states is not None:
            states = init_states
        else:
            states = torch.zeros(batch_size, self.d_state, device=x.device, dtype=x.dtype)

        # RWKV time mixing with decay
        outputs = []
        w = torch.exp(self.time_decay)  # (d_state,)
        u = self.time_first  # (d_state,)

        for t in range(seq_len):
            k_t = k[:, t, :]  # (batch, d_state)
            v_t = v[:, t, :]  # (batch, d_state)

            # RWKV-7 state update with Generalized Delta Rule
            wkv = torch.exp(-torch.exp(u) * k_t) * v_t  # Key-value interaction

            # State update with time decay
            states = states * w.unsqueeze(0) + wkv

            # Output projection
            out_t = self.output(states)  # (batch, d_model)
            outputs.append(out_t)

        output = torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)
        last_states = states.detach()

        return self.dropout(output), last_states


class ChannelMixing(nn.Module):
    """
    RWKV Channel Mixing layer.

    Processes channel dimensions using GLU-style gating similar to FFN.
    Applies non-linear transformation with sigmoid gating.
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden or (d_model * 4)

        # GLU-style projections
        self.key = nn.Linear(d_model, self.d_hidden, bias=False)
        self.value = nn.Linear(d_model, self.d_hidden, bias=False)
        self.gate = nn.Linear(d_model, self.d_hidden, bias=True)
        self.output = nn.Linear(self.d_hidden, d_model, bias=False)

        # Time-shift for channel mixing
        self.time_shift = nn.Linear(d_model, d_model)
        self.time_shift_gates = nn.Parameter(torch.ones(d_model))

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def time_shift_operation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply time-shift to channel mixing input."""
        if x.size(1) <= 1:
            return x

        shift_size = max(1, int(x.size(1) * 0.5))
        x_cat = torch.cat([
            x[:, shift_size:, :],
            x[:, :-shift_size, :]
        ], dim=1)

        return x_cat * self.time_shift_gates.sigmoid() + x * (1 - self.time_shift_gates.sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            output: (batch_size, seq_len, d_model)
        """
        x_shift = self.time_shift_operation(x)
        x_norm = self.norm(x_shift)

        k = self.key(x_norm)
        v = self.value(x_norm)
        g = self.gate(x_norm)

        # SiLU gating (Swish)
        output = self.output(F.silu(k) * v * torch.sigmoid(g))
        return self.dropout(output)


class RWKVLNBlock(nn.Module):
    """
    Combined RWKV block with LayerNorm before mixing operations.

    Each RWKV-7 block consists of:
        1. Pre-normalization (LayerNorm)
        2. Time Mixing (token-level linear attention)
        3. Residual connection
        4. Pre-normalization
        5. Channel Mixing (FFN with GLU)
        6. Residual connection
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_ffn: Optional[int] = None,
        dropout: float = 0.1,
        shift_size: float = 0.5,
    ):
        super().__init__()

        self.time_mix = TimeMixing(d_model, d_state, shift_size, dropout)
        self.channel_mix = ChannelMixing(d_model, d_ffn, dropout)

        # RWKV-7 residual gates
        self.time_mix_gate = nn.Parameter(torch.zeros(1, 1, d_model))
        self.channel_mix_gate = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(
        self,
        x: torch.Tensor,
        init_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            init_states: Optional initial hidden states

        Returns:
            output: (batch_size, seq_len, d_model)
            last_states: (batch_size, d_state) - final hidden states
        """
        # Time Mixing with residual
        residual = x
        x_norm = F.layer_norm(x, (x.size(-1),))
        time_out, last_states = self.time_mix(x_norm, init_states)
        x = residual + time_out * self.time_mix_gate.sigmoid()

        # Channel Mixing with residual
        residual = x
        channel_out = self.channel_mix(x)
        x = residual + channel_out * self.channel_mix_gate.sigmoid()

        return x, last_states


class RWKVState:
    """Container for RWKV hidden states (useful for RNN-style inference)."""

    def __init__(self, d_state: int, batch_size: int, device: torch.device):
        self.d_state = d_state
        self.batch_size = batch_size
        self.states = torch.zeros(batch_size, d_state, device=device)

    def detach(self):
        self.states = self.states.detach()
        return self

    def to(self, device: torch.device):
        self.states = self.states.to(device)
        return self
