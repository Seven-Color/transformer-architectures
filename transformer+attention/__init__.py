"""
Transformer + Various Attention Mechanisms
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class SinusoidalPositionalEmbedding(nn.Module):
    """Learnable positional embedding alternative"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)
    
    def forward(self, x):
        batch_size, seq_len = x.shape[:2]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        return x + self.embedding(positions)


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention"""
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections and split into heads
        q = self.q_linear(query).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        k = self.k_linear(key).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        v = self.v_linear(value).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_linear(output)


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention (can be used separately)"""
    def __init__(self, temperature, dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q, k.transpose(2, 3)) / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        return torch.matmul(attn, v)


class CausalSelfAttention(nn.Module):
    """Causal attention for autoregressive models"""
    def __init__(self, d_model, nhead, dropout=0.1, max_seq_len=5000):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, nhead, dropout)
        self.d_model = d_model
        self.nhead = nhead
        
        # Causal mask
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        self.register_buffer('causal_mask', mask)
    
    def forward(self, x, mask=None):
        seq_len = x.size(1)
        causal_mask = self.causal_mask[:seq_len, :seq_len].unsqueeze(0).unsqueeze(0)
        if mask is not None:
            combined_mask = torch.max(causal_mask, mask)
        else:
            combined_mask = causal_mask
        return self.attention(x, x, x, combined_mask)


class TransformerEncoderLayerAttention(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, attention_type='multihead'):
        super().__init__()
        self.attention_type = attention_type
        
        if attention_type == 'multihead':
            self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        elif attention_type == 'causal':
            self.self_attn = CausalSelfAttention(d_model, nhead, dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, src, src_mask=None):
        src2 = self.self_attn(src, src, src, src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerEncoderAttention(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward=2048, dropout=0.1, attention_type='multihead'):
        super().__init__()
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            TransformerEncoderLayerAttention(d_model, nhead, dim_feedforward, dropout, attention_type)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, src, src_mask=None):
        src = self.pos_encoder(src)
        for layer in self.layers:
            src = layer(src, src_mask)
        return self.norm(src)
