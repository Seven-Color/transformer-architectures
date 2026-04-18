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
from typing import Optional, List, Literal, Tuple, Dict, Any

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

        # ---- Streaming helpers ----
        # TimeConv1D needs a history buffer for temporal context
        # We store recent (kernel_size - 1) frames
        self._time_buf = None  # rolling buffer, set in streaming init

    def init_streaming_state(self, batch_size: int, device: torch.device):
        """
        Initialize per-layer streaming state.

        Returns a dict with:
          attn_state: None or zeros (depends on attention mode)
          time_buffer: ring buffer of recent frames for TimeConv1D
        """
        d_attn = self.f * self.h

        # Attention state
        if self.attention_mode == "rwkv":
            attn_state = torch.zeros(batch_size, self.attn.d_state, device=device)
        elif self.attention_mode == "self_attention":
            # KV cache: (k_cache, v_cache), each (B, nhead, seen_T, d_k)
            # Start empty — first frame initializes
            attn_state = None
        else:
            attn_state = None

        # TimeConv1D needs (kernel_size - 1) previous frames as buffer
        # Buffer: (B, kernel_size-1, F, H)
        time_buf = torch.zeros(
            batch_size, self.time_conv.kernel_size - 1, self.f, self.h,
            device=device,
            dtype=torch.float32,
        )

        return {"attn_state": attn_state, "time_buffer": time_buf}

    def forward_streaming(
        self,
        x: torch.Tensor,
        state: dict,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Process a single frame with streaming state.

        Args:
            x: (B, 1, F, H) — single frame
            state: dict with 'attn_state' and 'time_buffer'

        Returns:
            output: (B, 1, F, H)
            new_state: updated dict
        """
        B, T, F, H = x.shape
        assert T == 1, "Streaming expects single frame"
        d = F * H

        # ---- 1. Attention with state ----
        x_flat = x.reshape(B, 1, d)
        x_norm = self.norm_attn(x_flat)

        if self.attention_mode == "rwkv":
            attn_out, new_attn_state = self.attn.forward_streaming(
                x_norm, state["attn_state"]
            )
            attn_out = attn_out.reshape(B, 1, F, H)
        else:  # self_attention
            attn_out, new_attn_state = self.attn.forward_streaming(
                x, state["attn_state"]
            )
            attn_out = self.dropout_attn(attn_out)

        x = x + attn_out  # residual

        # ---- 2. Frequency Conv (stateless, per-frame) ----
        x_flat = x.reshape(B, 1, d)
        x_norm = self.norm_freq(x_flat).reshape(B, 1, F, H)
        freq_out = self.freq_conv(x_norm)
        x = x + self.dropout_freq(freq_out)

        # ---- 3. Time Conv with rolling buffer ----
        # Get time history from buffer
        time_buf = state["time_buffer"]  # (B, kernel-1, F, H)

        # Build full context: [buf_frames..., current_frame]
        # Conv1d needs (B, C, T) — T = kernel_size
        x_time = torch.cat([time_buf, x], dim=1)  # (B, kernel_size, F, H)

        # Update rolling buffer: drop oldest, add current
        new_time_buf = torch.cat(
            [time_buf[:, 1:], x], dim=1
        )  # (B, kernel_size-1, F, H)

        # TimeConv: (B, kernel_size, F, H) -> conv over T dim
        # reshape to (B, f*h, T) for Conv1d
        x_flat = x_time.permute(0, 2, 3, 1).reshape(B, d, self.time_conv.kernel_size)
        x_conv = self.time_conv.conv(x_flat)  # (B, f*h, T_out)
        # Take ONLY the last temporal position = current frame's conv result
        x_conv = x_conv[:, :, -1:]  # (B, f*h, 1)
        x_conv = x_conv.reshape(B, 1, d)
        x_conv = self.time_conv.norm(x_conv).reshape(B, 1, F, H)
        x = x + self.dropout_time(x_conv)

        # ---- 4. FFN (stateless, per-frame) ----
        x = x + self.dropout_ffn(self.ffn(x))

        new_state = {
            "attn_state": new_attn_state,
            "time_buffer": new_time_buf,
        }
        return x, new_state

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

    def init_streaming_state(
        self, batch_size: int, device: torch.device
    ) -> List[Dict[str, Any]]:
        """
        Initialize streaming state for all layers.

        Args:
            batch_size: batch dimension B
            device: torch device

        Returns:
            List of per-layer state dicts (one per layer)
        """
        return [
            layer.init_streaming_state(batch_size, device)
            for layer in self.layers
        ]

    def forward_streaming(
        self,
        x: torch.Tensor,
        states: List[Dict[str, Any]],
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """
        Stream one frame at a time through all layers.

        This is the key streaming interface. Each call processes a single
        frame (B, 1, F, H) using accumulated state from previous frames.

        Args:
            x: (B, 1, F, H) — single frame
            states: List of per-layer state dicts from init_streaming_state
                    or previous forward_streaming call

        Returns:
            output: (B, 1, F, H)
            new_states: updated list of per-layer states
        """
        B, T, F, H = x.shape
        assert T == 1, "forward_streaming expects single frame"

        # Input projection
        x = self.input_norm(self.input_proj(x))

        # Pass through each layer with its state
        new_states = []
        for li, layer in enumerate(self.layers):
            x, new_layer_state = layer.forward_streaming(x, states[li])
            new_states.append(new_layer_state)

        # Final norm
        x = self.final_norm(x)

        return x, new_states

    def forward_one_step(
        self,
        x: torch.Tensor,
        states: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """
        Convenience wrapper: init state if None, then forward_streaming.

        Args:
            x: (B, 1, F, H) — single frame
            states: None = initialize; or list from previous call

        Returns:
            output: (B, 1, F, H)
            new_states: for next call
        """
        if states is None:
            states = self.init_streaming_state(x.size(0), x.device)
        return self.forward_streaming(x, states)


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
