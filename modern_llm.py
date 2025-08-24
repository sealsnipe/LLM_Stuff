# %% [markdown]
# # Modern LLM Architecture - Production Ready
#
# Complete implementation of modern LLM with all latest optimizations:
# - Grouped-Query Attention (GQA)
# - RoPE (Rotary Positional Embeddings)
# - QK Normalization
# - SwiGLU Activation
# - RMSNorm
# - GPU-optimized for training and inference

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from dataclasses import dataclass

# %%
@dataclass
class ModelConfig:
    """Modern LLM Configuration with all optimizations."""
    # Model architecture
    vocab_size: int = 49152
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 2  # GQA: fewer KV heads than Q heads
    d_ff: int = 1536
    max_seq_len: int = 512
    
    # Training parameters
    max_steps: int = 2000
    batch_size: int = 24
    learning_rate: float = 5e-3
    muon_lr: float = 0.01
    weight_decay: float = 0.1
    
    # Regularization
    dropout: float = 0.1
    rms_norm_eps: float = 1e-6
    
    # Optimization
    use_amp: bool = True  # Automatic Mixed Precision
    gradient_clip: float = 1.0
    
    # Evaluation
    eval_steps: int = 100
    eval_interval: int = 500
    
    # Data
    max_tokens: int = 500_000
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.d_k = self.d_model // self.n_heads
        self.n_kv_groups = self.n_heads // self.n_kv_heads

# %%
class Rotary(nn.Module):
    """Optimized RoPE implementation for GPU."""
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequency matrix
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        
        # Cache for computed embeddings
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
    
    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update cos/sin cache if needed."""
        if seq_len > self._seq_len_cached or self._cos_cached is None or self._cos_cached.device != device:
            self._seq_len_cached = max(seq_len, self.max_seq_len)
            
            t = torch.arange(self._seq_len_cached, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            
            self._cos_cached = emb.cos().to(dtype)
            self._sin_cached = emb.sin().to(dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotary embeddings to input tensor."""
        batch_size, n_heads, seq_len, head_dim = x.shape
        
        # Update cache
        self._update_cache(seq_len, x.device, x.dtype)
        
        # Get cos/sin for current sequence
        cos = self._cos_cached[:seq_len, :head_dim].unsqueeze(0).unsqueeze(0)
        sin = self._sin_cached[:seq_len, :head_dim].unsqueeze(0).unsqueeze(0)
        
        # Apply rotation
        x1, x2 = x[..., :head_dim//2], x[..., head_dim//2:]
        rotated = torch.cat([x1 * cos[..., :head_dim//2] - x2 * sin[..., head_dim//2:],
                            x1 * sin[..., :head_dim//2] + x2 * cos[..., head_dim//2:]], dim=-1)
        
        return rotated

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Efficiently repeat key/value heads for GQA."""
    if n_rep == 1:
        return x
    
    batch_size, n_kv_heads, seq_len, head_dim = x.shape
    return x[:, :, None, :, :].expand(batch_size, n_kv_heads, n_rep, seq_len, head_dim).reshape(
        batch_size, n_kv_heads * n_rep, seq_len, head_dim
    )

# %%
class ModernAttention(nn.Module):
    """Modern attention with GQA, RoPE, and QK normalization."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.d_k = config.d_k
        self.n_kv_groups = config.n_kv_groups
        
        # Projections
        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.d_k, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.n_kv_heads * self.d_k, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.n_kv_heads * self.d_k, bias=False)
        self.w_o = nn.Linear(self.n_heads * self.d_k, self.d_model, bias=False)
        
        # QK Normalization for stability
        self.q_norm = nn.RMSNorm(self.d_k, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.d_k, eps=config.rms_norm_eps)
        
        # RoPE
        self.rotary = Rotary(self.d_k, config.max_seq_len)
        self.dropout = config.dropout
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.size(0), x.size(1)
        
        # 1. Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 2. Reshape into heads
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.d_k)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.d_k)
        
        # 3. Apply QK normalization
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # 4. Apply RoPE
        q = self.rotary(q.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        k = self.rotary(k.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        
        # 5. Transpose for attention: (batch, seq_len, n_heads, d_k) -> (batch, n_heads, seq_len, d_k)
        Q = q.transpose(1, 2)
        K = k.transpose(1, 2)
        V = v.transpose(1, 2)
        
        # 6. Repeat K and V heads for GQA
        K = repeat_kv(K, self.n_kv_groups)
        V = repeat_kv(V, self.n_kv_groups)
        
        # 7. Scaled Dot-Product Attention (GPU-optimized)
        attn_output = F.scaled_dot_product_attention(
            Q, K, V, 
            is_causal=True, 
            dropout_p=self.dropout if self.training else 0.0
        )
        
        # 8. Reshape and final projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.w_o(attn_output)

# %%
class SwiGLUFeedForward(nn.Module):
    """SwiGLU activation function for better performance."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: Swish(gate) * up
        activated_x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(self.dropout(activated_x))

# %%
class TransformerBlock(nn.Module):
    """Modern transformer block with pre-norm and residual connections."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = ModernAttention(config)
        self.feed_forward = SwiGLUFeedForward(config.d_model, config.d_ff, config.dropout)
        self.norm1 = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.norm2 = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)
        
        # Pre-norm feed forward
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        
        return x

# %%
class ModernLLM(nn.Module):
    """
    Modern LLM with all optimizations:
    - GQA (Grouped-Query Attention)
    - RoPE (Rotary Positional Embeddings)
    - QK Normalization
    - SwiGLU activation
    - RMSNorm
    - Weight tying
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Output layers
        self.norm = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.output_dropout = nn.Dropout(config.dropout)
        
        # Language modeling head with weight tying
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # Weight tying
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with proper scaling."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        # Token embeddings with scaling
        x = self.token_embedding(x) * math.sqrt(self.config.d_model)
        x = self.position_dropout(x)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Final normalization and output
        x = self.norm(x)
        x = self.output_dropout(x)
        logits = self.lm_head(x)
        
        return logits
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_device(self) -> torch.device:
        """Get device of the model."""
        return next(self.parameters()).device

# %%
if __name__ == "__main__":
    # Test the model
    config = ModelConfig()
    model = ModernLLM(config)
    
    print(f"🤖 Modern LLM created!")
    print(f"   Parameters: {model.get_num_params():,}")
    print(f"   Architecture: {config.d_model}d, {config.n_layers}L, {config.n_heads}H")
    print(f"   GQA: {config.n_heads} Q heads, {config.n_kv_heads} KV heads")
    
    # Test forward pass
    batch_size = 2
    seq_len = 64
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        logits = model(x)
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {logits.shape}")
        print("✅ Model test passed!")
