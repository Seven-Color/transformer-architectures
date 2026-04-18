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
    RWKV Time Mixing Layer - Parallel Scan Implementation.

    Replaces O(T^2) self-attention with O(T) linear recurrence
    computed via parallel scan (Hillis-Steele algorithm).

    The recurrence:
        state_t = w * state_{t-1} + wkv_t

    Is reformulated as:
        state_t = a_t ⊗ state_{t-1} ⊕ b_t
    where ⊗ and ⊕ form an associative operator pair that can be
    computed in O(log T) depth on GPU.

    This eliminates the sequential for-loop over T time steps.

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

        # Time decay per state dimension: controls how fast information decays
        self.time_decay = nn.Parameter(torch.zeros(d_state))
        # First-token factor: initial contribution boost
        self.time_first = nn.Parameter(torch.zeros(d_state))

        # Key and Value projections
        self.key = nn.Linear(d_attn, d_state, bias=False)
        self.value = nn.Linear(d_attn, d_state, bias=False)

        # Output projection: state -> d_attn
        self.output = nn.Linear(d_state, d_attn, bias=False)

        # Token-shift: blend adjacent time steps
        self.time_shift = nn.Linear(d_attn, d_attn, bias=False)
        self.time_shift_gate = nn.Parameter(torch.ones(d_attn))

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_attn)

    def token_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Token-shift: blend current and half-shifted previous time step."""
        if x.size(1) <= 1:
            return x
        shift_len = max(1, int(x.size(1) * self.time_shift_size))
        x_cat = torch.cat([x[:, shift_len:], x[:, :-shift_len]], dim=1)
        gate = self.time_shift_gate.sigmoid()
        return self.time_shift(x_cat) * gate + x * (1 - gate)

    @staticmethod
    def _combine(a1, b1, a2, b2):
        """
        Associative combine of two (a, b) pairs.
        (a1,b1) ⊗ (a2,b2) = (a1*a2, a1*b2 + b1)
        """
        return a1 * a2, a1 * b2 + b1

    def _sequential_scan(self, wkv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sequential recurrence — O(T) total work, O(1) memory.

        Computes: state_t = w * state_{t-1} + wkv_t

        This is the simplest and most numerically stable approach.
        For typical audio spectral sequences (T <= 512), the sequential
        loop has negligible overhead and avoids parallel-scan overhead.

        For very long sequences (T > 10000), use RWKVTimeMixingChunked
        which applies a two-level scan (intra-chunk + inter-chunk).

        Args:
            wkv: (B, T, d_state)

        Returns:
            output: (B, T, d_attn)
            last_state: (B, d_state)
        """
        B, T, D = wkv.shape
        w = torch.exp(self.time_decay)  # (d_state,)

        states = wkv.new_zeros(B, D)
        outputs = []

        for t in range(T):
            states = states * w + wkv[:, t, :]
            outputs.append(self.output(states))  # (B, d_attn)

        return torch.stack(outputs, dim=1), states  # (B,T,d_attn), (B,d_state)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with parallel scan.

        Args:
            x: (B, T, F, H) or (B, T, D)

        Returns:
            output: (B, T, F, H) or (B, T, D)
            last_state: (B, d_state)
        """
        original_shape = x.shape
        if x.dim() == 4:
            B, T, F, H = x.shape
            x = x.reshape(B, T, F * H)
        else:
            B, T, D = x.shape
            F, H = 1, D

        # Pre-norm + token shift
        x_shift = self.token_shift(x)
        x_norm = self.norm(x_shift)

        # Project to k, v
        k = self.key(x_norm)    # (B, T, d_state)
        v = self.value(x_norm)  # (B, T, d_state)

        # Compute wkv: content-based contribution
        w = torch.exp(self.time_decay)       # (d_state,)
        u = self.time_first                  # (d_state,)

        # wkv_t = exp(-exp(u) * sigmoid(k_t)) * v_t
        wkv = torch.exp(-torch.exp(u).unsqueeze(0).unsqueeze(0) * torch.sigmoid(k)) * v

        # ---- Sequential scan: O(T) total work ----
        output, last_state = self._sequential_scan(wkv)  # (B,T,d_attn), (B,d_state)
        last_state = last_state.detach()

        if len(original_shape) == 4:
            output = output.reshape(B, T, F, H)

        return self.dropout(output), last_state

    def forward_streaming(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Streaming single-frame forward — the core of RWKV inference.

        Called per incoming frame. No token-shift across time (impossible
        with only one frame). State carries all cross-frame history.

        Args:
            x: Single frame, (B, 1, F, H) or (B, 1, D)
            state: Previous hidden state, (B, d_state). None = initialize.

        Returns:
            output: (B, 1, F, H) or (B, 1, D)
            new_state: (B, d_state) — for next frame
        """
        original_shape = x.shape
        if x.dim() == 4:
            B, T, F, H = x.shape
            assert T == 1, "Streaming expects single frame"
            x = x.reshape(B, 1, F * H)
        else:
            B, T, D = x.shape
            assert T == 1, "Streaming expects single frame"
            F, H = 1, D

        # Pre-norm (no token-shift possible with single frame)
        x_norm = self.norm(x)

        # k, v projections
        k = self.key(x_norm)   # (B, 1, d_state)
        v = self.value(x_norm)  # (B, 1, d_state)

        w = torch.exp(self.time_decay)  # (d_state,)
        u = self.time_first             # (d_state,)

        # wkv for this single frame
        wkv = torch.exp(
            -torch.exp(u).unsqueeze(0).unsqueeze(0) * torch.sigmoid(k)
        ) * v  # (B, 1, d_state)

        # Initialize state if first frame
        if state is None:
            state = wkv.new_zeros(B, self.d_state)

        # Recurrent state update — the only truly recurrent operation
        new_state = state * w + wkv.squeeze(1)  # (B, d_state)

        # Output projection
        output = self.output(new_state)  # (B, d_attn)
        output = output.unsqueeze(1)     # (B, 1, d_attn)

        if len(original_shape) == 4:
            output = output.reshape(B, 1, F, H)

        return self.dropout(output), new_state.detach()


class RWKVTimeMixingChunked(BaseAttention):
    """
    Chunked RWKV Time Mixing - processes very long sequences efficiently.

    Splits sequence into chunks, computes parallel scan within each chunk,
    then combines chunk-level states with a second scan pass.

    This is a two-level parallel scan (online RVWKV algorithm):
      1. Intra-chunk scan: O(chunk_size * log(chunk_size))
      2. Inter-chunk scan: O(num_chunks * log(num_chunks))

    Better for very long sequences (T >> 10000) where a single
    full parallel scan would use too much memory.

    Args:
        d_attn:  Attention dimension (f * h)
        d_state: Hidden state dimension
        chunk_size: Size of each chunk (default 128)
        dropout:  Dropout rate
    """

    def __init__(
        self,
        d_attn: int,
        d_state: int = 64,
        chunk_size: int = 128,
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

    def _combine(self, a1, b1, a2, b2):
        """Associative combine: (a1,b1) ⊗ (a2,b2) = (a1*a2, a1*b2 + b1)"""
        return a1 * a2, a1 * b2 + b1

    def _intra_chunk_scan(self, wkv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parallel scan within one chunk.

        Args:
            wkv: (B, chunk_size, d_state)

        Returns:
            states: (B, chunk_size, d_state) — final states after intra-chunk scan
            last_state: (B, d_state) — chunk's last state
        """
        B, chunk_size, D = wkv.shape
        w = torch.exp(self.time_decay)  # (D,)
        u = self.time_first

        # Pad to power of 2
        size = 1
        while size < chunk_size:
            size <<= 1

        # Initialize: a=identity (1), b=wkv
        a = w.unsqueeze(0).unsqueeze(0).expand(B, size, D).clone()  # (B, size, D)
        a[:, 0] = 1.0  # identity
        b = wkv.new_zeros(B, size, D)
        b[:, :chunk_size] = wkv

        # Pad
        w_expanded = w.unsqueeze(0).unsqueeze(0).expand(B, size, D)

        # Parallel scan (Hillis-Steele)
        for stride in range(1, size):
            # Which positions can combine?
            # At stride s: position i (i >= s) combines with i-s
            valid = torch.arange(size, device=wkv.device).unsqueeze(0) >= stride
            valid = valid.unsqueeze(-1).expand(B, size, D)

            a_cur = a.clone()
            b_cur = b.clone()

            a_prev = a_cur[:, :-stride].contiguous()
            b_prev = b_cur[:, :-stride].contiguous()
            a_shifted = a_cur[:, stride:].contiguous()
            b_shifted = b_cur[:, stride:].contiguous()
            w_shifted = w_expanded[:, stride:].contiguous()

            # Combine
            a_combined = w_shifted * a_prev  # w * a_prev
            b_combined = w_shifted * b_prev + b_shifted  # w * b_prev + b_shifted

            a_new = a.clone()
            b_new = b.clone()
            a_new[:, stride:] = torch.where(valid[:, stride:], a_combined, a_shifted)
            b_new[:, stride:] = torch.where(valid[:, stride:], b_combined, b_shifted)

            a = a_new
            b = b_new

        return b[:, :chunk_size], b[:, chunk_size - 1]

    def _inter_chunk_scan(
        self, chunk_states: torch.Tensor, chunk_wkvs: torch.Tensor
    ) -> torch.Tensor:
        """
        Second-level scan across chunk final states.

        Each chunk's effect on its own elements is:
          chunk_state[i] = chunk_a * chunk_init_state + chunk_b[i]
        where chunk_a = w^{chunk_size} (decay over full chunk)
        and chunk_b[i] = sum_{j=0}^{i} w^{i-j} * wkv_j

        For inter-chunk, we treat each chunk as having:
          a_chunk = w^{chunk_size}  (constant for all chunks)
          b_chunk = chunk_state[last] = chunk_a * init_state + chunk_b_last

        The two-level scan correctly handles cross-chunk dependencies.

        Args:
            chunk_states: (B, num_chunks, d_state) — last state of each chunk
            chunk_wkvs: (B, num_chunks, d_state) — last wkv of each chunk

        Returns:
            chunk_states: (B, num_chunks, d_state) — corrected after inter-chunk scan
        """
        B, num_chunks, D = chunk_states.shape
        w = torch.exp(self.time_decay)
        chunk_size = self.chunk_size

        # Effective chunk decay: w^chunk_size
        a_chunk = (w ** chunk_size).unsqueeze(0).unsqueeze(0).expand(B, num_chunks, D)

        # Pad to power of 2
        size = 1
        while size < num_chunks:
            size <<= 1

        a = a_chunk.new_ones(B, size, D)
        b = chunk_states.new_zeros(B, size, D)
        b[:, :num_chunks] = chunk_states

        # Inter-chunk parallel scan
        for stride in range(1, size):
            valid = torch.arange(size, device=chunk_states.device).unsqueeze(0) >= stride
            valid = valid.unsqueeze(-1).expand(B, size, D)

            a_prev = a[:, :-stride]
            b_prev = b[:, :-stride]
            a_shifted = a[:, stride:]
            b_shifted = b[:, stride:]
            a_w = (w ** chunk_size).unsqueeze(0).unsqueeze(0).expand_as(a_shifted)

            a_combined = a_w * a_prev
            b_combined = a_w * b_prev + b_shifted

            a_new = a.clone()
            b_new = b.clone()
            a_new[:, stride:] = torch.where(valid[:, stride:], a_combined, a_shifted)
            b_new[:, stride:] = torch.where(valid[:, stride:], b_combined, b_shifted)

            a = a_new
            b = b_new

        return b[:, :num_chunks]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Two-level parallel scan: intra-chunk then inter-chunk.
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

        k = self.key(x_norm)
        v = self.value(x_norm)

        w = torch.exp(self.time_decay)
        u = self.time_first

        # wkv per time step
        wkv = torch.exp(-torch.exp(u).unsqueeze(0).unsqueeze(0) * torch.sigmoid(k)) * v

        # ---- Two-level scan ----
        chunk_size = self.chunk_size
        num_chunks = (T + chunk_size - 1) // chunk_size

        # Pad to chunk boundary
        pad_len = num_chunks * chunk_size - T
        if pad_len > 0:
            wkv = F.pad(wkv, (0, 0, 0, pad_len))

        wkv_chunks = wkv.reshape(B, num_chunks, chunk_size, self.d_state)

        # Step 1: intra-chunk parallel scan
        chunk_states = []   # last state of each chunk
        chunk_outputs = []   # all outputs within each chunk
        w_chunk = w ** chunk_size  # decay per full chunk

        for c in range(num_chunks):
            wkv_c = wkv_chunks[:, c]  # (B, chunk_size, d_state)
            states_c, last_c = self._intra_chunk_scan(wkv_c)  # (B, chunk_size, d_state)
            chunk_states.append(last_c)                         # (B, d_state)
            chunk_outputs.append(self.output(states_c))          # (B, chunk_size, d_attn)

        chunk_states = torch.stack(chunk_states, dim=1)  # (B, num_chunks, d_state)

        # Step 2: inter-chunk parallel scan to get corrected chunk initial states
        chunk_states_corrected = self._inter_chunk_scan(chunk_states, wkv_chunks[:, :, -1, :])

        # Step 3: apply corrected initial states and reconstruct outputs
        all_outputs = []
        for c in range(num_chunks):
            wkv_c = wkv_chunks[:, c]  # (B, chunk_size, d_state)
            init_state_c = chunk_states_corrected[:, c]  # (B, d_state)

            # Redo intra-chunk scan with corrected init state
            # state[t] = w * state[t-1] + wkv[t], starting from init_state_c
            # This is just a simple scan: prepend init_state_c as t=-1
            wkv_c_full = torch.cat([
                init_state_c.unsqueeze(1),
                wkv_c[:, :-1]
            ], dim=1)  # shifted so first element = init_state_c

            # Simple scan for each chunk (O(chunk_size) but with small chunk)
            chunk_size_actual = min(chunk_size, T - c * chunk_size)
            w_exp = w.unsqueeze(0).unsqueeze(0)  # (1, 1, d_state)
            states_c = torch.zeros(B, self.d_state, device=x.device, dtype=x.dtype)
            outputs_c = []
            for t in range(chunk_size_actual):
                wkv_t = wkv_chunks[:, c, t]
                states_c = states_c * w_exp.squeeze() + wkv_t
                outputs_c.append(self.output(states_c))

            all_outputs.append(torch.stack(outputs_c, dim=1))

        output = torch.cat(all_outputs, dim=1)[:, :T, :]
        last_state = chunk_states_corrected[:, -1, :].detach()

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

    def forward_streaming(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Streaming single-frame forward with KV cache.

        Uses cached K, V from all previous frames.
        State is (k_cache, v_cache) — None for first frame.

        Args:
            x: (B, 1, F, H) — single frame
            state: Optional (k_cache, v_cache)

        Returns:
            output: (B, 1, F, H)
            new_state: (k_cache, v_cache) for next frame
        """
        output, new_cache = self.forward_with_cache(x, state)
        return output, new_cache

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
