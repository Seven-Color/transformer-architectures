"""
RWKV Spectral Model
==================

8-layer network for spectral/temporal modeling with shape (B, T, F, H).

Two attention modes:
  1. "rwkv": RWKV Time Mixing (linear-time, O(T) per step)
  2. "self_attention": Standard full attention across (f*h)

Each layer structure:
  Input (B, T, F, H)
    |
    +-> Attention(f*h) ----------------------------------------+
    |   (RWKV or Self-Attn across time T)                      |
    |                                                        |
    +-> FreqConv1D (along F) --------------------------------+ |
    |   (depthwise conv across frequency bins)                | |
    |                                                        | |
    +-> TimeConv1D (along T) ------------------------------+ | |
    |   (conv across time frames)                             | | |
    |                                                        | | |
    +-> FFN(h) -------------------------------------------+ | | |
        (channel-wise feed-forward)                           | | |
                                                             | | |
    Residual connections between each sub-layer               | | |
                                                             | | |
    Final output: (B, T, F, H)                               v v v

Reference: RWKV-7 "Goose" - arXiv:2503.14456
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List, Literal

from .blocks.rwkv_attention import (
    RWKVTimeMixing,
    RWKVTimeMixingChunked,
    SelfAttentionFH,
    FreqConv1D,
    TimeConv1D,
    FFNH,
)


# ============================================================
# Single RWKV Layer
# ============================================================

class SpectralRWKVLayer(nn.Module):
    """
    One layer of the Spectral RWKV model.

    Sub-components:
      1. Attention (f*h)   - token mixing across time
      2. FreqConv1D       - 1D conv along frequency F
      3. TimeConv1D        - 1D conv along time T
      4. FFN              - channel-wise feed-forward on H

    Each sub-layer has a residual connection and layer norm.
    """

    def __init__(
        self,
        f: int,
        h: int,
        attention_mode: Literal["rwkv", "self_attention"] = "rwkv",
        d_state: int = 64,
        nhead: int = 8,
        conv_kernel: int = 3,
        h_ffn_mult: int = 4,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.f = f
        self.h = h
        self.attention_mode = attention_mode

        d_attn = f * h

        # Pre-normalization for attention
        self.norm_attn = nn.LayerNorm(d_attn)

        # Attention: RWKV or Standard Self-Attention
        if attention_mode == "rwkv":
            self.attn = RWKVTimeMixing(
                d_attn=d_attn,
                d_state=d_state,
                dropout=attn_dropout,
            )
        elif attention_mode == "self_attention":
            self.attn = SelfAttentionFH(
                f=f,
                h=h,
                nhead=nhead,
                dropout=attn_dropout,
            )
        else:
            raise ValueError(f"Unknown attention_mode: {attention_mode}")

        self.dropout_attn = nn.Dropout(dropout)

        # Frequency convolution (along F)
        self.norm_freq = nn.LayerNorm(d_attn)
        self.freq_conv = FreqConv1D(
            f=f, h=h, kernel_size=conv_kernel, dropout=dropout
        )
        self.dropout_freq = nn.Dropout(dropout)

        # Time convolution (along T)
        self.norm_time = nn.LayerNorm(d_attn)
        self.time_conv = TimeConv1D(
            f=f, h=h, kernel_size=conv_kernel, dropout=dropout
        )
        self.dropout_time = nn.Dropout(dropout)

        # FFN (along H)
        self.ffn = FFNH(
            f=f, h=h, h_ffn_mult=h_ffn_mult, dropout=dropout
        )
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, F, H)

        Returns:
            (B, T, F, H)
        """
        B, T, F, H = x.shape
        d = F * H

        # ---- Attention with residual ----
        x_flat = x.reshape(B, T, d)
        x_norm = self.norm_attn(x_flat)

        if self.attention_mode == "rwkv":
            attn_out, _ = self.attn(x_norm)
            attn_out = attn_out.reshape(B, T, F, H)
        else:
            # SelfAttentionFH takes (B,T,F,H) directly
            attn_out, _ = self.attn(x)
            attn_out = self.dropout_attn(attn_out)

        # Residual: attention
        x = x + attn_out

        # ---- Frequency Conv with residual ----
        x_flat = x.reshape(B, T, d)
        x_norm = self.norm_freq(x_flat).reshape(B, T, F, H)
        freq_out = self.freq_conv(x_norm)
        x = x + self.dropout_freq(freq_out)

        # ---- Time Conv with residual ----
        x_flat = x.reshape(B, T, d)
        x_norm = self.norm_time(x_flat).reshape(B, T, F, H)
        time_out = self.time_conv(x_norm)
        x = x + self.dropout_time(time_out)

        # ---- FFN with residual ----
        x = x + self.dropout_ffn(self.ffn(x))

        return x


# ============================================================
# Full Model
# ============================================================

class RWKVSpectral(nn.Module):
    """
    8-layer RWKV Spectral Network.

    Input:  (B, T, F, H) - batch, time, frequency, channel
    Output: (B, T, F, H) - same shape

    Architecture:
      1. Input projection (optional): H -> H (identity by default)
      2. 8 x SpectralRWKVLayer
      3. Final layer norm
    """

    def __init__(
        self,
        f: int,
        h: int,
        num_layers: int = 8,
        attention_mode: Literal["rwkv", "self_attention"] = "rwkv",
        d_state: int = 64,
        nhead: int = 8,
        conv_kernel: int = 3,
        h_ffn_mult: int = 4,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.f = f
        self.h = h
        self.num_layers = num_layers
        self.attention_mode = attention_mode

        # Input embedding: project channel dim
        self.input_proj = nn.Linear(h, h)
        self.input_norm = nn.LayerNorm(h)

        # Stack of RWKV layers
        self.layers = nn.ModuleList([
            SpectralRWKVLayer(
                f=f,
                h=h,
                attention_mode=attention_mode,
                d_state=d_state,
                nhead=nhead,
                conv_kernel=conv_kernel,
                h_ffn_mult=h_ffn_mult,
                dropout=dropout,
                attn_dropout=attn_dropout,
            )
            for _ in range(num_layers)
        ])

        # Final normalization
        self.final_norm = nn.LayerNorm(h)

        # Initialize
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, F, H)

        Returns:
            (B, T, F, H)
        """
        # Input projection
        B, T, F, H = x.shape
        x = self.input_norm(self.input_proj(x))  # (B, T, F, H)

        # Pass through all layers
        for layer in self.layers:
            x = layer(x)

        # Final norm
        x = self.final_norm(x)

        return x

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# Config
# ============================================================

@dataclass
class RWKVSpectralConfig:
    """Configuration for RWKV Spectral model."""
    name: str = "RWKV-Spectral"

    # Input dimensions
    f: int = 257          # frequency bins
    h: int = 64           # channels

    # Architecture
    num_layers: int = 8
    attention_mode: Literal["rwkv", "self_attention"] = "rwkv"

    # Attention params
    d_state: int = 64     # RWKV hidden state dim
    nhead: int = 8        # number of attention heads (self_attention mode)

    # Conv params
    conv_kernel: int = 3

    # FFN params
    h_ffn_mult: int = 4   # FFN hidden = h * h_ffn_mult

    # Regularization
    dropout: float = 0.1
    attn_dropout: float = 0.1


# ============================================================
# Factory
# ============================================================

def create_rwkv_spectral(
    f: int,
    h: int,
    num_layers: int = 8,
    attention_mode: Literal["rwkv", "self_attention"] = "rwkv",
    **kwargs,
) -> RWKVSpectral:
    """
    Factory function to create a Spectral RWKV model.
    """
    return RWKVSpectral(
        f=f,
        h=h,
        num_layers=num_layers,
        attention_mode=attention_mode,
        **kwargs,
    )
