"""
GPU-Optimized Attention Module

Contains the GPUOptimizedAttention class with memory-efficient attention computation.
Optimized for Flash Attention and grouped-query attention (GQA).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import model_config


class GPUOptimizedAttention(nn.Module):
    """GPU-optimierte Attention mit Memory-Effizienz."""
    
    def __init__(self):
        super().__init__()
        self.hidden_size = model_config.hidden_size
        self.num_attention_heads = model_config.num_attention_heads
        self.num_key_value_heads = model_config.num_key_value_heads
        self.head_dim = model_config.hidden_size // model_config.num_attention_heads
        self.num_key_value_groups = model_config.num_attention_heads // model_config.num_key_value_heads
        self.scaling = 1.0 / math.sqrt(self.head_dim)

        # Projections - optimiert für GPU
        self.q_proj = nn.Linear(model_config.hidden_size, model_config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(model_config.hidden_size, model_config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(model_config.hidden_size, model_config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(model_config.num_attention_heads * self.head_dim, model_config.hidden_size, bias=False)

        # Dropout für Training (0.0 für Flash-Kompatibilität)
        self.attention_dropout = nn.Dropout(0.0)
    
    def _attention_forward(self, hidden_states, attention_mask=None):
        """Interne Attention-Berechnung für Memory-Optimierung."""
        batch_size, seq_len, _ = hidden_states.shape

        # Projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape für Multi-Head
        query_states = query_states.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # GQA: Repeat K/V heads
        if self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        #  FLASH ATTENTION - Optimiert für >95% Hit-Rate
        # KEINE ATTENTION_MASK für maximale Flash-Kompatibilität!
        # Padding wird über ignore_index=-100 im Loss behandelt
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=None,      # KEINE MASKE - Flash-Hit-Rate >95%!
            dropout_p=0.0,       # Dropout=0.0 für Flash-Pfad
            is_causal=True       # Nur kausale Maske
        )

        # Reshape und Output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

        return self.o_proj(attn_output)

    def forward(self, hidden_states, attention_mask=None):
        # Attention Layer macht KEIN Checkpointing - wird im Block gemacht
        return self._attention_forward(hidden_states, attention_mask)
