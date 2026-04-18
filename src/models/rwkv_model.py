"""
RWKV-7 (Goose) Model Implementation.

A transformer-free language model that achieves GPT-level performance
using a novel linear-time architecture with:
- Time Mixing (token-level linear attention with time decay)
- Channel Mixing (GLU-style feed-forward)
- Generalized Delta Rule for state evolution (RWKV-7)

Reference:
- RWKV-7 Paper: arXiv:2503.14456
- RWKV Website: https://www.rwkv.cn
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple, List

from ..core.base import BaseModule
from .blocks.rwkv_attention import RWKVLNBlock, TimeMixing, ChannelMixing, RWKVState


@dataclass
class RWKVConfig:
    """Configuration for RWKV-7 model."""
    name: str = "RWKV-7"
    vocab_size: int = 65536
    d_model: int = 512
    num_layers: int = 12
    d_state: int = 64  # Hidden state dimension
    d_ffn: Optional[int] = None  # Defaults to d_model * 4
    dropout: float = 0.1
    shift_size: float = 0.5
    max_seq_len: int = 4096
    tie_weights: bool = True
    emb_dropout: float = 0.0


class Embedding(nn.Module):
    """Token embedding layer with optional dropout."""

    def __init__(self, vocab_size: int, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.embedding(x) * (self.d_model ** 0.5))


class PositionalEncoding(nn.Module):
    """
    RWKV does not use traditional positional encoding.
    Instead, we use a simple learned absolute position embedding
    that gets added during the time-shift operation.
    """

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        return self.embedding(positions)


class RWKVModelBody(nn.Module):
    """
    Core RWKV body - stacks of RWKV blocks.
    """

    def __init__(self, config: RWKVConfig):
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList([
            RWKVLNBlock(
                d_model=config.d_model,
                d_state=config.d_state,
                d_ffn=config.d_ffn or config.d_model * 4,
                dropout=config.dropout,
                shift_size=config.shift_size,
            )
            for _ in range(config.num_layers)
        ])

        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        init_states: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            init_states: Optional list of initial states per layer

        Returns:
            output: (batch_size, seq_len, d_model)
            final_states: List of hidden states per layer
        """
        layer_states = []

        for i, layer in enumerate(self.layers):
            init_state = init_states[i] if init_states is not None else None
            x, last_state = layer(x, init_state)
            layer_states.append(last_state)

        x = self.layer_norm(x)
        return x, layer_states


class RWKV7Model(BaseModule):
    """
    RWKV-7 (Goose) Language Model.

    A GPT-level performance RNN that:
    - Uses NO self-attention (100% attention-free)
    - Has O(N) inference time and memory
    - Supports unlimited context length (in principle)
    - Achieves transformer-quality results

    Architecture:
        Token Embedding
            -> RWKV Blocks (Time Mixing + Channel Mixing)
            -> LayerNorm
            -> Language Modeling Head
    """

    def __init__(self, config: Optional[RWKVConfig] = None):
        super().__init__()
        self.config = config or RWKVConfig()

        cfg = self.config

        # Embedding
        self.embedding = Embedding(cfg.vocab_size, cfg.d_model, cfg.emb_dropout)

        # Positional encoding (simple learned - optional for RWKV)
        self.pos_encoding = PositionalEncoding(cfg.d_model, cfg.max_seq_len)

        # RWKV body
        self.body = RWKVModelBody(cfg)

        # Output head
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Tie weights between embedding and output
        if cfg.tie_weights:
            self.head.weight = self.embedding.embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize weights with appropriate schemes."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        init_states: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            input_ids: (batch_size, seq_len) - token indices
            init_states: Optional list of initial hidden states per layer

        Returns:
            logits: (batch_size, seq_len, vocab_size)
            final_states: List of hidden states per layer
        """
        # Embed tokens
        x = self.embedding(input_ids)

        # Add positional info (subtle)
        seq_len = x.size(1)
        x = x + self.pos_encoding(seq_len, x.device) * 0.1

        # RWKV body
        x, final_states = self.body(x, init_states)

        # Language modeling head
        logits = self.head(x)

        return logits, final_states

    def forward_one_step(
        self,
        input_ids: torch.Tensor,
        states: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        RNN-style one-step forward (for autoregressive inference).

        Args:
            input_ids: (batch_size, 1) - single token
            states: Hidden states from previous step

        Returns:
            logits: (batch_size, 1, vocab_size)
            new_states: Updated hidden states
        """
        x = self.embedding(input_ids)
        x = x + self.pos_encoding(1, x.device) * 0.1

        layer_states = []
        for i, layer in enumerate(self.body.layers):
            state = states[i] if states is not None else None
            x, last_state = layer(x, state)
            layer_states.append(last_state)

        x = self.body.layer_norm(x)
        logits = self.head(x)

        return logits, layer_states

    def init_hidden_states(
        self,
        batch_size: int,
        device: torch.device,
    ) -> List[torch.Tensor]:
        """Initialize hidden states for RNN-style inference."""
        return [
            torch.zeros(batch_size, self.config.d_state, device=device)
            for _ in range(self.config.num_layers)
        ]

    def get_parameter_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Alias for convenience
RWKV7 = RWKV7Model
