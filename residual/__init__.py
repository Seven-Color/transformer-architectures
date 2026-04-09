"""
Residual Connections for Transformer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    """Basic residual connection wrapper"""
    def __init__(self, module, dropout=0.0):
        super().__init__()
        self.module = module
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, x):
        residual = x
        output = self.module(x)
        if self.dropout:
            output = self.dropout(output)
        return output + residual


class PreNormResidual(nn.Module):
    """Pre-norm residual connection (apply norm before module, then add residual)"""
    def __init__(self, d_model, module):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.module = module
    
    def forward(self, x):
        return self.module(self.norm(x)) + x


class ResBlock(nn.Module):
    """Residual feed-forward block"""
    def __init__(self, d_model, hidden_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or d_model * 4
        
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return x + self.net(x)


class SkipConnection(nn.Module):
    """Generic skip connection with optional transformation"""
    def __init__(self, module, transform=None):
        super().__init__()
        self.module = module
        self.transform = transform
    
    def forward(self, x):
        identity = x
        output = self.module(x)
        if self.transform:
            identity = self.transform(identity)
        return output + identity


class MultiHeadResidualAttention(nn.Module):
    """Multi-head attention with residual connections"""
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        return self.norm(x + self.dropout(attn_output))


class FeedForwardResidual(nn.Module):
    """Feed-forward network with residual connection"""
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.GELU()
    
    def forward(self, x):
        # FFN output
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        # Residual connection
        return self.norm(x + self.dropout(ff_output))


class TransformerEncoderLayerResidual(nn.Module):
    """Transformer encoder layer with residual connections"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, use_pre_norm=True):
        super().__init__()
        self.use_pre_norm = use_pre_norm
        
        if use_pre_norm:
            # Pre-norm architecture
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
                nn.Dropout(dropout)
            )
        else:
            # Post-norm architecture
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
                nn.Dropout(dropout)
            )
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src, src_mask=None):
        if self.use_pre_norm:
            # Pre-norm: norm -> attention -> residual
            src = src + self.self_attn(self.norm1(src), self.norm1(src), self.norm1(src), attn_mask=src_mask)[0]
            src = src + self.ffn(self.norm2(src))
        else:
            # Post-norm: attention -> residual -> norm
            src2 = self.self_attn(src, src, src, attn_mask=src_mask)[0]
            src = self.norm1(src + self.dropout1(src2))
            src = self.norm2(src + self.dropout2(self.ffn(src)))
        return src


class TransformerEncoderResidual(nn.Module):
    """Transformer encoder with configurable residual connection strategies"""
    def __init__(self, num_layers, d_model, nhead, dim_feedforward=2048, dropout=0.1, use_pre_norm=True):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayerResidual(d_model, nhead, dim_feedforward, dropout, use_pre_norm)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, src, src_mask=None):
        for layer in self.layers:
            src = layer(src, src_mask)
        return self.norm(src)