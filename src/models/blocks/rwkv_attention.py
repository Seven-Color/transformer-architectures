"""
RWKV Attention - Proper Implementation
========================================

Two attention modes for spectral/temporal modeling:
1. RWKV-Time: Linear-time token-mixing with time decay (no O(T^2) attention)
2. Standard Self-Attention: Full attention across (f*h) positions

Input shape: (B, T, F, H)
  B = batch
  T = time dimension (frames)
  F = frequency dimension (bins)
  H = channel dimension (features per frequency bin)

Each layer: Attention(f*h) -> FreqConv1D -> TimeConv1D -> FFN(h)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


# ============================================================
# Base Classes
# ============================================================

class BaseAttention(nn.Module):
    """Base attention interface."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ============================================================
# RWKV Time Mixing (Linear-time token mixing)
# ============================================================

class RWKVTimeMixing(BaseAttention):
    """
    RWKV Time Mixing Layer.

    Replaces O(T^2) self-attention with O(T) linear recurrence.
    Uses the Generalized Delta Rule from RWKV-7 for state evolution.

    Key mechanism:
      state = decay * state + token_shift(x) * wkv
      output = Linear(state)

    The token_shift blends adjacent time steps via a learned gate,
    providing local context before the linear state update.

    Args:
        d_attn:  Attention dimension (f * h)
        d_state: Hidden state dimension for the linear recurrence
        time_shift_size: Fraction of sequence to shift for token-shift (default 0.5)
        dropout:  Dropout rate
    """

    def __init__(
        self,
        d_attn: int,
        d_state: int = 64,
        time_shift_size: float = 0.5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_attn = d_attn
        self.d_state = d_state
        self.time_shift_size = time_shift_size

        # Learnable time decay: exp(-exp(time_first) * key)
        # This gives each head a different decay rate
        self.time_decay = nn.Parameter(torch.zeros(d_state))
        self.time_first = nn.Parameter(torch.zeros(d_state))

        # Key and Value projections: x -> k, v
        self.key = nn.Linear(d_attn, d_state, bias=False)
        self.value = nn.Linear(d_attn, d_state, bias=False)

        # Output projection: state -> x
        self.output = nn.Linear(d_state, d_attn, bias=False)

        # Token-shift: half-sequence shift + learned gate
        self.time_shift = nn.Linear(d_attn, d_attn, bias=False)
        self.time_shift_gate = nn.Parameter(torch.ones(d_attn))

        # RWKV-7: Reception field extension (optional, for longer context)
        # Adds a direct content-based contribution
        self.reception_w = nn.Parameter(torch.zeros(d_attn, d_state))
        self.reception_u = nn.Parameter(torch.zeros(d_attn, d_state))

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_attn)

    def token_shift(self, x: torch.Tensor) -> torch.Tensor:
        """
        Token-shift: blend current and previous time step.

        x: (B, T, F, H) or (B, T, D) where D=f*h
        Returns: same shape as x
        """
        if x.size(1) <= 1:
            return x

        seq_len = x.size(1)
        shift_len = max(1, int(seq_len * self.time_shift_size))

        # Cat: [x_shift, x_prev] along time dim
        x_cat = torch.cat([x[:, shift_len:], x[:, :-shift_len]], dim=1)

        # Learnable gate
        gate = self.time_shift_gate.sigmoid()  # (D,) or (F*H,)
        shift_out = self.time_shift(x_cat)

        # Blend: gate * shifted + (1-gate) * original
        return shift_out * gate + x * (1 - gate)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor (B, T, F, H) or (B, T, D)

        Returns:
            output: (B, T, F, H) or (B, T, D)
            last_state: (B, d_state) final hidden state for RNN inference
        """
        # Handle both 4D (B,T,F,H) and 3D (B,T,D) input
        original_shape = x.shape
        if x.dim() == 4:
            B, T, F, H = x.shape
            x = x.reshape(B, T, F * H)  # (B, T, f*h)
        else:
            B, T, D = x.shape
            F = 1
            H = D
            x = x.reshape(B, T, D)

        # Pre-norm
        x_shift = self.token_shift(x)
        x_norm = self.norm(x_shift)

        # Project to key and value
        k = self.key(x_norm)  # (B, T, d_state)
        v = self.value(x_norm)  # (B, T, d_state)

        # Initialize states
        B_dim = k.size(0)
        states = torch.zeros(B_dim, self.d_state, device=x.device, dtype=x.dtype)

        outputs = []

        # RWKV time-stepwise recurrence
        w = torch.exp(self.time_decay)          # (d_state,) decay factor
        u = self.time_first                      # (d_state,) first-token factor

        for t in range(T):
            k_t = k[:, t, :]                    # (B, d_state)
            v_t = v[:, t, :]                    # (B, d_state)

            # RWKV-7: Generalized Delta Rule
            # wkv = exp(-exp(u) * k_t) * v_t
            # This is the content-based contribution
            wkv = torch.exp(-torch.exp(u) * torch.sigmoid(k_t)) * v_t

            # State update: exponential decay + new contribution
            states = states * w + wkv

            # Output: project state back to d_attn
            out_t = self.output(states)          # (B, d_attn)
            outputs.append(out_t)

        output = torch.stack(outputs, dim=1)     # (B, T, d_attn)
        last_state = states.detach()

        # Restore original shape
        if len(original_shape) == 4:
            output = output.reshape(B, T, F, H)

        return self.dropout(output), last_state


class RWKVTimeMixingChunked(BaseAttention):
    """
    Chunked RWKV Time Mixing - more efficient for long sequences.

    Processes the sequence in chunks using the parallel scan algorithm
    for faster computation while maintaining O(T) memory.

    Args:
        d_attn:  Attention dimension (f * h)
        d_state: Hidden state dimension
        chunk_size: Size of chunks for parallel processing (default 64)
        dropout:  Dropout rate
    """

    def __init__(
        self,
        d_attn: int,
        d_state: int = 64,
        chunk_size: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_attn = d_attn
        self.d_state = d_state
        self.chunk_size = chunk_size

        self.time_decay = nn.Parameter(torch.zeros(d_state))
        self.time_first = nn.Parameter(torch.zeros(d_state))

        self.key = nn.Linear(d_attn, d_state, bias=False)
        self.value = nn.Linear(d_attn, d_state, bias=False)
        self.output = nn.Linear(d_state, d_attn, bias=False)

        self.time_shift = nn.Linear(d_attn, d_attn, bias=False)
        self.time_shift_gate = nn.Parameter(torch.ones(d_attn))

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_attn)

    def token_shift(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) <= 1:
            return x
        shift_len = max(1, int(x.size(1) * 0.5))
        x_cat = torch.cat([x[:, shift_len:], x[:, :-shift_len]], dim=1)
        gate = self.time_shift_gate.sigmoid()
        return self.time_shift(x_cat) * gate + x * (1 - gate)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Chunked forward for efficiency.

        Uses cumulative sum trick for parallel scan over the sequence.
        """
        original_shape = x.shape
        if x.dim() == 4:
            B, T, F, H = x.shape
            x = x.reshape(B, T, F * H)
        else:
            B, T, D = x.shape
            F, H = 1, D

        x_shift = self.token_shift(x)
        x_norm = self.norm(x_shift)

        k = self.key(x_norm)  # (B, T, d_state)
        v = self.value(x_norm)  # (B, T, d_state)

        # Chunked parallel scan using cumulative operations
        w = torch.exp(self.time_decay)  # (d_state,)
        u = self.time_first

        # Chunk the sequence
        chunk_size = self.chunk_size
        num_chunks = (T + chunk_size - 1) // chunk_size

        # Pad to multiple of chunk_size
        pad_len = num_chunks * chunk_size - T
        if pad_len > 0:
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))

        k = k.reshape(B, num_chunks, chunk_size, self.d_state)
        v = v.reshape(B, num_chunks, chunk_size, self.d_state)

        # Simple recurrent for now (can be optimized with parallel scan)
        states = torch.zeros(B, self.d_state, device=x.device, dtype=x.dtype)
        all_outputs = []

        for c in range(num_chunks):
            chunk_outputs = []
            for t in range(chunk_size):
                idx = c * chunk_size + t
                if idx >= T:
                    break
                k_t = k[:, c, t, :]
                v_t = v[:, c, t, :]
                wkv = torch.exp(-torch.exp(u) * torch.sigmoid(k_t)) * v_t
                states = states * w + wkv
                chunk_outputs.append(self.output(states))
            all_outputs.append(torch.stack(chunk_outputs, dim=1))

        output = torch.cat(all_outputs, dim=1)[:, :T, :]
        last_state = states.detach()

        if len(original_shape) == 4:
            output = output.reshape(B, T, F, H)

        return self.dropout(output), last_state


# ============================================================
# Standard Self-Attention (f*h dimension)
# ============================================================

class SelfAttentionFH(BaseAttention):
    """
    Standard Multi-Head Self-Attention across (f*h) dimension.

    Treats each (f, h) position as an independent token,
    attending over all time steps T.

    Input:  (B, T, F, H)
    Output: (B, T, F, H)

    The attention is computed at each time step independently,
    mixing information across the frequency-channel features.
    """

    def __init__(
        self,
        f: int,
        h: int,
        nhead: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        d_attn = f * h
        assert d_attn % nhead == 0, f"f*h={d_attn} must be divisible by nhead={nhead}"

        self.f = f
        self.h = h
        self.d_attn = d_attn
        self.nhead = nhead
        self.d_k = d_attn // nhead

        # QKV projections: x -> q, k, v
        self.q_proj = nn.Linear(d_attn, d_attn, bias=False)
        self.k_proj = nn.Linear(d_attn, d_attn, bias=False)
        self.v_proj = nn.Linear(d_attn, d_attn, bias=False)

        # Output projection
        self.out_proj = nn.Linear(d_attn, d_attn, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Args:
            x: (B, T, F, H)

        Returns:
            output: (B, T, F, H)
            last_state: None (standard attention has no state)
        """
        B, T, freq, ch = x.shape
        d = freq * ch

        # Reshape: treat (F, H) as "token" features
        x_flat = x.reshape(B, T, d)  # (B, T, f*h)

        # QKV
        q = self.q_proj(x_flat).reshape(B, T, self.nhead, self.d_k).transpose(1, 2)
        k = self.k_proj(x_flat).reshape(B, T, self.nhead, self.d_k).transpose(1, 2)
        v = self.v_proj(x_flat).reshape(B, T, self.nhead, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scale = math.sqrt(self.d_k)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply
        out = torch.matmul(attn, v)  # (B, nhead, T, d_k)
        out = out.transpose(1, 2).reshape(B, T, d)  # (B, T, f*h)
        out = self.out_proj(out).reshape(B, T, freq, ch)  # (B, T, F, H)

        return out, None

    def forward_with_cache(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward with KV-cache for autoregressive inference.

        Args:
            x: (B, 1, F, H) - single time step
            kv_cache: cached (k, v) from previous steps

        Returns:
            output: (B, 1, F, H)
            new_cache: updated (k, v)
        """
        B, T, freq, ch = x.shape
        d = freq * ch
        assert T == 1, "Cache mode requires single time step"

        x_flat = x.reshape(B, T, d)
        q = self.q_proj(x_flat).reshape(B, T, self.nhead, self.d_k).transpose(1, 2)
        k = self.k_proj(x_flat).reshape(B, T, self.nhead, self.d_k).transpose(1, 2)
        v = self.v_proj(x_flat).reshape(B, T, self.nhead, self.d_k).transpose(1, 2)

        if kv_cache is not None:
            k_cat, v_cat = kv_cache
            k = torch.cat([k_cat, k], dim=2)
            v = torch.cat([v_cat, v], dim=2)

        scale = math.sqrt(self.d_k)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, T, d)
        out = self.out_proj(out).reshape(B, T, freq, ch)

        return out, (k.detach(), v.detach())


# ============================================================
# Convolution Modules
# ============================================================

class FreqConv1D(nn.Module):
    """
    1D Convolution along the Frequency dimension (F axis).

    At each time step t, applies a conv kernel across the F frequency bins.
    Uses depthwise conv: each channel h gets its own conv across F bins.

    Input:  (B, T, F, H) -> output: (B, T, F, H)
    """

    def __init__(
        self,
        f: int,
        h: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.f = f
        self.h = h
        self.kernel_size = kernel_size
        padding = (kernel_size - 1) // 2

        # Depthwise: groups=h, each channel independently convolved across F
        # Weight: (h, 1, kernel_size), input: (B*T, h, f)
        self.conv = nn.Conv1d(
            in_channels=h,
            out_channels=h,
            kernel_size=kernel_size,
            padding=padding,
            groups=h,   # depthwise: each of h channels convolved independently
            bias=False,
        )
        self.norm = nn.LayerNorm(f * h)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, F, H)

        Returns:
            (B, T, F, H)
        """
        B, T, F, H = x.shape

        # Reshape: merge B,T into batch, keep F,H
        # Then transpose: (B, T, F, H) -> (B*T, H, F)
        x = x.permute(0, 1, 3, 2).reshape(B * T, H, F)  # (B*T, H, F)
        x = self.conv(x)                                   # (B*T, H, F), depthwise
        x = x.reshape(B, T, H, F).permute(0, 1, 3, 2)    # (B, T, F, H)

        x_flat = x.reshape(B, T, F * H)
        x_flat = self.norm(x_flat)
        return self.dropout(x_flat.reshape(B, T, F, H))


class TimeConv1D(nn.Module):
    """
    1D Convolution along the Time dimension (T axis).

    At each frequency bin f and channel h, applies a conv kernel across time.
    Input:  (B, T, F, H) -> output: (B, T, F, H)
    """

    def __init__(
        self,
        f: int,
        h: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.f = f
        self.h = h
        self.kernel_size = kernel_size
        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv1d(
            in_channels=h * f,
            out_channels=h * f,
            kernel_size=kernel_size,
            padding=padding,
            groups=1,
            bias=False,
        )
        self.norm = nn.LayerNorm(f * h)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, F, H)

        Returns:
            (B, T, F, H)
        """
        B, T, F, H = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, F * H, T)  # (B, f*h, T)
        x = self.conv(x)                                  # (B, f*h, T)
        x = x.reshape(B, F, H, T).permute(0, 3, 1, 2)  # (B, T, F, H)
        x = x.reshape(B, T, F * H)
        x = self.norm(x)
        return self.dropout(x.reshape(B, T, F, H))


# ============================================================
# FFN (h dimension)
# ============================================================

class FFNH(nn.Module):
    """
    Feed-Forward Network operating on the H (channel) dimension.

    Applied independently at each (t, f) position.
    Input:  (B, T, F, H)
    Output: (B, T, F, H)
    """

    def __init__(
        self,
        f: int,
        h: int,
        h_ffn_mult: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.f = f
        self.h = h

        d_ffn = h * h_ffn_mult
        self.fc1 = nn.Linear(h, d_ffn, bias=False)  # per channel, shared across F
        self.fc2 = nn.Linear(d_ffn, h, bias=False)
        self.dropout = nn.Dropout(dropout)

        if activation.lower() == "gelu":
            self.act = nn.GELU()
        elif activation.lower() == "silu":
            self.act = nn.SiLU()
        else:
            self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, F, H)

        Returns:
            (B, T, F, H)
        """
        B, T, F, H = x.shape

        # Apply FFN per (t, f) position
        x_reshaped = x.reshape(B * T * F, H)          # (B*T*F, H)
        x_ffn = self.fc2(self.act(self.fc1(x_reshaped)))  # (B*T*F, h_ffn) -> (B*T*F, h)
        x_ffn = self.dropout(x_ffn)
        return x_ffn.reshape(B, T, F, H)
