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
    create_denoising_transformer,
    DenoisingTransformerConfig,
    compare_architectures,
    print_comparison,
    StandardDenoisingTransformer,
    ResidualAttentionDenoisingTransformer,
    MoEDenoisingTransformer,
    StandardSelfAttention,
    ResidualAttention,
    PreNormResidualAttention,
    MultiScaleAttention,
    StandardFFN,
    SwiGLUFFN,
    MoEFFN,
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
    FeedForwardMoE,
)
from .rwkv_model import RWKVSpectral, RWKVSpectralConfig, create_rwkv_spectral
from .blocks.rwkv_attention import (
    RWKVTimeMixing,
    RWKVTimeMixingChunked,
    SelfAttentionFH,
    FreqConv1D,
    TimeConv1D,
    FFNH,
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
    # Attention variants
    'StandardSelfAttention',
    'ResidualAttention',
    'PreNormResidualAttention',
    'MultiScaleAttention',
    # FFN variants
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
    # RWKV Spectral
    'RWKVSpectral',
    'RWKVSpectralConfig',
    'create_rwkv_spectral',
    'RWKVTimeMixing',
    'RWKVTimeMixingChunked',
    'SelfAttentionFH',
    'FreqConv1D',
    'TimeConv1D',
    'FFNH',
]
