"""
Memory-Optimized LLM Model

Contains the main MemoryOptimizedLLM class - the complete language model
with embeddings, transformer layers, and output head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import model_config, training_config
from .transformer_block import MemoryOptimizedTransformerBlock


class MemoryOptimizedLLM(nn.Module):
    """1B Parameter LLM - REALISTISCH für RTX 4070 Ti (12GB)."""

    def __init__(self):
        super().__init__()

        # Embeddings
        self.token_embeddings = nn.Embedding(model_config.vocab_size, model_config.hidden_size)

        # GPT-5: Dropout Layers (für Pretraining auf 0.0 gesetzt)
        self.dropout = nn.Dropout(model_config.dropout_prob)

        # Transformer layers - verwende memory-optimierte Blöcke
        self.layers = nn.ModuleList([
            MemoryOptimizedTransformerBlock() for _ in range(model_config.num_layers)
        ])

        # Output
        self.norm = nn.LayerNorm(model_config.hidden_size, eps=1e-5)
        self.lm_head = nn.Linear(model_config.hidden_size, model_config.vocab_size, bias=False)

        # Weight tying für Memory-Effizienz
        if hasattr(model_config, 'tie_word_embeddings') and model_config.tie_word_embeddings:
            self.lm_head.weight = self.token_embeddings.weight

        # Weight initialization
        self.apply(self._init_weights)

        # Parameter count für konsistente Logs
        self.total_params = sum(p.numel() for p in self.parameters())
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, labels=None):
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        hidden_states = self.token_embeddings(input_ids)

        # GPT-5: Apply dropout after embeddings
        hidden_states = self.dropout(hidden_states)

        # Causal mask
        attention_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=input_ids.device),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)

        # Transformer layers - Memory-optimiert
        for layer in self.layers:
            # Gradient Checkpointing ist bereits in den Layern implementiert
            hidden_states = layer(hidden_states, attention_mask)

        # Output
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        # Loss calculation mit korrekter PAD-Token Maskierung
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Cross-Entropy in FP32 für numerische Stabilität
            # Logits zu FP32 konvertieren für CE-Berechnung
            shift_logits_fp32 = shift_logits.float()

            # Fused Cross-Entropy falls verfügbar
            if training_config.use_fused_cross_entropy:
                try:
                    from flash_attn.losses.cross_entropy import CrossEntropyLoss as FusedCrossEntropyLoss
                    loss_fct = FusedCrossEntropyLoss(ignore_index=-100)
                    loss = loss_fct(shift_logits_fp32.view(-1, model_config.vocab_size), shift_labels.view(-1))
                except ImportError:
                    # Fallback: Standard CrossEntropy in FP32
                    loss = F.cross_entropy(
                        shift_logits_fp32.view(-1, model_config.vocab_size),
                        shift_labels.view(-1),
                        ignore_index=-100
                    )
            else:
                # Standard CrossEntropy in FP32
                loss = F.cross_entropy(
                    shift_logits_fp32.view(-1, model_config.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100
                )
        
        return {"loss": loss, "logits": logits}
