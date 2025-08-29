"""
Memory-Optimized Transformer Block

Contains the MemoryOptimizedTransformerBlock class with selective gradient checkpointing
and optimized feed-forward networks using SwiGLU activation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from config import model_config, training_config
from .attention import GPUOptimizedAttention


class MemoryOptimizedTransformerBlock(nn.Module):
    """Memory-optimierter Transformer Block mit Checkpointing."""

    def __init__(self):
        super().__init__()
        self.attention = GPUOptimizedAttention()

        # Feed Forward - SwiGLU für bessere Performance
        self.gate_proj = nn.Linear(model_config.hidden_size, model_config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(model_config.hidden_size, model_config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(model_config.intermediate_size, model_config.hidden_size, bias=False)

        # Layer Norms
        self.attention_norm = nn.LayerNorm(model_config.hidden_size, eps=1e-5)
        self.ffn_norm = nn.LayerNorm(model_config.hidden_size, eps=1e-5)

    def _ffn_forward(self, hidden_states):
        """Interne FFN-Berechnung für Checkpointing."""
        # SwiGLU Activation
        gate = torch.nn.functional.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        return self.down_proj(gate * up)

    def _block_forward(self, hidden_states, attention_mask=None):
        """Kompletter Block für Checkpointing."""
        # Pre-norm attention
        normed_states = self.attention_norm(hidden_states)
        attn_output = self.attention(normed_states, attention_mask)
        hidden_states = hidden_states + attn_output

        # Pre-norm feed forward
        normed_states = self.ffn_norm(hidden_states)
        ffn_output = self._ffn_forward(normed_states)
        hidden_states = hidden_states + ffn_output

        return hidden_states

    def forward(self, hidden_states, attention_mask=None):
        #  MEMORY OPTIMIZATION: Selective Gradient Checkpointing
        if training_config.use_activation_checkpointing and self.training:
            if training_config.selective_checkpointing == "attention":
                # Nur Attention checkpointen
                attn_out = checkpoint(self.attention, hidden_states, attention_mask, use_reentrant=True)
                attn_out = self.attention_norm(attn_out + hidden_states)

                # MLP normal
                mlp_out = self.gate_proj(attn_out)
                mlp_out = F.silu(mlp_out) * self.up_proj(attn_out)
                mlp_out = self.down_proj(mlp_out)
                return self.ffn_norm(mlp_out + attn_out)

            elif training_config.selective_checkpointing == "mlp":
                # Attention normal
                attn_out = self.attention(hidden_states, attention_mask)
                attn_out = self.attention_norm(attn_out + hidden_states)

                # Nur MLP checkpointen
                def mlp_forward(x):
                    mlp_out = self.gate_proj(x)
                    mlp_out = F.silu(mlp_out) * self.up_proj(x)
                    return self.down_proj(mlp_out)

                mlp_out = checkpoint(mlp_forward, attn_out, use_reentrant=True)
                return self.ffn_norm(mlp_out + attn_out)

            else:  # "full"
                return checkpoint(self._block_forward, hidden_states, attention_mask, use_reentrant=True)
        else:
            return self._block_forward(hidden_states, attention_mask)
