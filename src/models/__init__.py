"""
Models module initialization
"""

from .transformer import TransformerEncoder, TransformerEncoderLayer
from .transformer_attention import (
    TransformerEncoderAttention,
    TransformerEncoderLayerAttention,
    MultiHeadAttention
)
from .transformer_residual import (
    TransformerEncoderResidual,
    TransformerEncoderLayerResidual,
    ResBlock
)
from .denoising_transformer import (
    # Factory
    create_denoising_transformer,
    DenoisingTransformerConfig,
    # Comparison utilities
    compare_architectures,
    print_comparison,
    # Model classes
    StandardDenoisingTransformer,
    ResidualAttentionDenoisingTransformer,
    MoEDenoisingTransformer,
    # Attention variants
    StandardSelfAttention,
    ResidualAttention,
    PreNormResidualAttention,
    MultiScaleAttention,
    # FFN variants
    StandardFFN,
    SwiGLUFFN,
    MoEFFN,
    # Encoder layers
    StandardEncoderLayer,
    ResidualAttentionEncoderLayer,
    MoEEncoderLayer
)
from .blocks import (
    PositionalEncoding,
    SinusoidalPositionalEmbedding,
    MultiHeadAttention as MHA,
    ScaledDotProductAttention,
    CausalSelfAttention,
    FeedForward,
    FeedForwardMoE
)
from .rwkv_model import RWKV7Model, RWKVConfig
from .blocks.rwkv_attention import (
    TimeMixing,
    ChannelMixing,
    RWKVLNBlock,
    RWKVState
)

__all__ = [
    # Transformer base
    'TransformerEncoder',
    'TransformerEncoderLayer',
    'TransformerEncoderAttention',
    'TransformerEncoderLayerAttention',
    'TransformerEncoderResidual',
    'TransformerEncoderLayerResidual',
    'ResBlock',
    'MultiHeadAttention',
    # Denoising models
    'create_denoising_transformer',
    'DenoisingTransformerConfig',
    'compare_architectures',
    'print_comparison',
    'StandardDenoisingTransformer',
    'ResidualAttentionDenoisingTransformer',
    'MoEDenoisingTransformer',
    # Attention
    'StandardSelfAttention',
    'ResidualAttention',
    'PreNormResidualAttention',
    'MultiScaleAttention',
    # FFN
    'StandardFFN',
    'SwiGLUFFN',
    'MoEFFN',
    # Encoder layers
    'StandardEncoderLayer',
    'ResidualAttentionEncoderLayer',
    'MoEEncoderLayer',
    # Blocks
    'PositionalEncoding',
    'SinusoidalPositionalEmbedding',
    'ScaledDotProductAttention',
    'CausalSelfAttention',
    'FeedForward',
    'FeedForwardMoE',
    # RWKV-7
    'RWKV7Model',
    'RWKVConfig',
    'TimeMixing',
    'ChannelMixing',
    'RWKVLNBlock',
    'RWKVState'
]
