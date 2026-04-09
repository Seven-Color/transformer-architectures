"""
Transformer + Mixture of Experts (MoE)
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


class FeedForwardMoE(nn.Module):
    """Mixture of Experts Feed Forward layer"""
    def __init__(self, d_model, num_experts, top_k=2, expert_capacity_factor=1.0, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_capacity_factor = expert_capacity_factor
        self.d_model = d_model
        
        # Gating network
        self.gate = nn.Linear(d_model, num_experts)
        
        # Experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout)
            )
            for _ in range(num_experts)
        ])
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        """
        x: [batch, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)  # [batch*seq_len, d_model]
        
        # Gating
        logits = self.gate(x_flat)  # [batch*seq_len, num_experts]
        weights = F.softmax(logits, dim=1)
        
        # Top-k selection
        top_weights, top_indices = torch.topk(weights, self.top_k, dim=1)
        top_weights = top_weights / (top_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        expert_capacity = int(x_flat.size(0) * self.expert_capacity_factor)
        
        # Process each expert
        for i, expert in enumerate(experts):
            expert_mask = (top_indices == i).any(dim=1)
            expert_output = expert(x_flat[expert_mask])
            output[expert_mask] += expert_output * top_weights[expert_mask, top_indices[expert_mask] == i].unsqueeze(1)
        
        output = self.norm(output)
        return output.view(batch_size, seq_len, d_model)


class TransformerEncoderLayerMoE(nn.Module):
    def __init__(self, d_model, nhead, num_experts=4, top_k=2, dim_feedforward=None, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.moe = FeedForwardMoE(d_model, num_experts, top_k, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src, src_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        src2 = self.moe(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerEncoderMoE(nn.Module):
    def __init__(self, num_layers, d_model, nhead, num_experts=4, top_k=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            TransformerEncoderLayerMoE(d_model, nhead, num_experts, top_k, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, src, src_mask=None):
        src = self.pos_encoder(src)
        for layer in self.layers:
            src = layer(src, src_mask)
        return self.norm(src)
