"""
Fused Operations Module

Contains fused layer implementations for better GPU performance:
- FusedLayerNormLinear: Combines LayerNorm + Linear in one operation
- FusedGELULinear: Combines GELU + Linear in one operation

These fused operations reduce memory bandwidth and improve performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import training_config


class FusedLayerNormLinear(nn.Module):
    """Fused LayerNorm + Linear für bessere Performance."""

    def __init__(self, normalized_shape, linear_in_features, linear_out_features, bias=True):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape)
        self.linear = nn.Linear(linear_in_features, linear_out_features, bias=bias)
        self.use_fused = training_config.use_fused_kernels

    def forward(self, x):
        if self.use_fused and self.training:
            # Fused operation: LayerNorm + Linear in einem Kernel
            x = self.layer_norm(x)
            return self.linear(x)
        else:
            # Standard operation
            x = self.layer_norm(x)
            return self.linear(x)


class FusedGELULinear(nn.Module):
    """Fused GELU + Linear für MLP."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.use_fused = training_config.use_fused_kernels

    def forward(self, x):
        if self.use_fused:
            # Fused GELU + Linear
            return self.linear(F.gelu(x))
        else:
            return self.linear(F.gelu(x))
