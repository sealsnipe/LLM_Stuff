# %% [markdown]
# # LLM Training Configuration
#
# Zentrale Konfigurationsdatei für alle Parameter des LLM-Training-Systems.
# Einfach die Werte hier ändern - werden automatisch von allen Komponenten verwendet.

# %%
import os
from dataclasses import dataclass
from typing import Optional, List

# %%
# === MODEL ARCHITECTURE ===
@dataclass
class ModelConfig:
    """Model Architecture Configuration."""

    # Basic Architecture
    vocab_size: int = 49152  # SmolLM-135M Tokenizer Vocabulary
    hidden_size: int = 1536
    num_layers: int = 12
    num_attention_heads: int = 24
    num_key_value_heads: int = 6  # GQA: 4:1 ratio (24:6)
    intermediate_size: int = 4096  # FFN hidden size
    max_position_embeddings: int = 2048

    # Modern Features
    use_gqa: bool = True  # Grouped-Query Attention
    use_rope: bool = True  # Rotary Position Embeddings
    use_swiglu: bool = True  # SwiGLU activation
    use_qk_norm: bool = True  # QK Normalization
    tie_word_embeddings: bool = True  # Weight tying

    # Regularization
    dropout_prob: float = 0.1
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    layer_norm_eps: float = 1e-6
    initializer_range: float = 0.02

    # Compatibility aliases for modern_llm.py
    @property
    def d_model(self) -> int:
        return self.hidden_size

    @property
    def n_layers(self) -> int:
        return self.num_layers

    @property
    def n_heads(self) -> int:
        return self.num_attention_heads

    @property
    def n_kv_heads(self) -> int:
        return self.num_key_value_heads

    @property
    def d_k(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def n_kv_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def d_ff(self) -> int:
        return self.intermediate_size

    @property
    def dropout(self) -> float:
        return self.dropout_prob

    @property
    def rms_norm_eps(self) -> float:
        return self.layer_norm_eps

    @property
    def max_seq_len(self) -> int:
        return self.max_position_embeddings

# === TRAINING CONFIGURATION ===
@dataclass
class TrainingConfig:
    """Training Configuration."""

    # NEW: Token-based Training Control
    target_tokens: int = 100_000_000  # 100M tokens (quick test)
    sequence_length: int = 384

    # Training Parameters
    batch_size: int = 5
    gradient_accumulation_steps: int = 8  # Effective batch size: 40
    learning_rate: float = 3e-4  # Lower for stability
    weight_decay: float = 0.1
    max_grad_norm: float = 0.5  # Tighter clipping

    # Calculated automatically from target_tokens
    @property
    def max_steps(self) -> int:
        """Calculate steps needed for target tokens"""
        effective_batch_size = self.batch_size * self.gradient_accumulation_steps
        tokens_per_step = effective_batch_size * self.sequence_length
        return self.target_tokens // tokens_per_step

    # Optimizer Settings
    optimizer_type: str = "adamw_fused"  # "adamw_fused", "adamw", "muon_hybrid"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8

    # Muon-specific (wenn optimizer_type = "muon_hybrid")
    muon_lr: float = 0.02
    muon_momentum: float = 0.95

    # Learning Rate Schedule
    lr_scheduler: str = "cosine"  # "cosine", "linear", "constant"
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1  # Minimum LR als Ratio der max LR

    # Performance Optimizations
    use_torch_compile: bool = True
    torch_compile_mode: str = "max-autotune"  # "default", "reduce-overhead", "max-autotune"
    use_mixed_precision: bool = True
    mixed_precision_dtype: str = "bfloat16"  # "bfloat16", "float16"
    use_activation_checkpointing: bool = True  # Memory vs Speed tradeoff

    # Advanced Speed Optimizations
    use_flash_attention: bool = True  # Faster attention computation (already active)
    use_fused_kernels: bool = True   # Fused LayerNorm+Linear operations
    use_optimized_attention: bool = True  # Optimized attention patterns

    # Advanced Speed Optimizations
    use_flash_attention: bool = True  # Faster attention computation
    use_fused_kernels: bool = True   # Fused operations where possible
    dataloader_prefetch_factor: int = 4  # Increased prefetching

    # Monitoring & Logging
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 1000
    max_checkpoints_to_keep: int = 3

    # Data Loading
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = True
    dataloader_prefetch_factor: int = 4  # Increased for better GPU utilization

# === INFERENCE CONFIGURATION ===
@dataclass
class InferenceConfig:
    """Text Generation Configuration."""

    # Generation Parameters
    max_length: int = 100
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    num_return_sequences: int = 1

    # Stopping Criteria
    stop_tokens: Optional[List[str]] = None
    max_time: Optional[float] = None  # Seconds

    # Performance
    use_torch_compile: bool = True
    use_mixed_precision: bool = True
    batch_size: int = 1

    # Tokenizer
    tokenizer_path: str = "HuggingFaceTB/SmolLM-135M"
    add_special_tokens: bool = False

# === HARDWARE CONFIGURATION ===
@dataclass
class HardwareConfig:
    """Hardware-specific Configuration."""

    # GPU Settings
    device: str = "auto"  # "auto", "cuda", "cpu", "cuda:0"
    gpu_memory_fraction: float = 0.95  # Max GPU memory to use

    # Memory Management
    empty_cache_interval: int = 100  # Steps between cache clearing
    max_memory_gb: Optional[float] = None  # Hard limit

    # Multi-GPU (Future)
    use_distributed: bool = False
    world_size: int = 1
    local_rank: int = 0

# === DATASET CONFIGURATION ===
@dataclass
class DatasetConfig:
    """Dataset Configuration."""

    # Default Dataset Settings
    default_tokenizer: str = "HuggingFaceTB/SmolLM-135M"
    default_dataset: str = "fineweb-edu"  # "fineweb-edu", "custom"

    # FineWeb-Edu Settings
    # Verwendet HuggingFace Standard-Cache (plattformübergreifend)
    # Windows: C:\Users\{user}\.cache\huggingface\hub
    # Linux: ~/.cache/huggingface/hub
    fineweb_cache_dir: str = os.path.expanduser("~/.cache/huggingface/hub")
    fineweb_streaming: bool = True
    fineweb_num_samples: int = 10000  # Default sample size

    # Dataset Size Profile Selection
    # Verfügbare Profile:
    # - "tiny": 1k samples, ~5 min training (quick testing)
    # - "small": 10k samples, ~30 min training (development)
    # - "medium": 300k samples, ~15 hours training (serious training)
    # - "large": 1M samples, ~2 days training (production-ready)
    # - "production": 10M samples, ~1-2 weeks training (full production)
    # - "full": All samples, ~months training (complete dataset)
    default_dataset_size: str = "auto"  # Use new token-based system

    # Dataset Size Presets
    dataset_sizes: dict = None

    def __post_init__(self):
        if self.dataset_sizes is None:
            # NEW: Token-based dataset configuration
            # Based on target_tokens from TrainingConfig
            self.dataset_sizes = {
                "test": {
                    "target_tokens": 100_000_000,  # 100M tokens (quick test)
                    "description": "Quick test run",
                    "training_time": "~2 hours",
                    "steps": "~6.5K"
                },
                "small": {
                    "target_tokens": 1_000_000_000,  # 1B tokens
                    "description": "Small training run",
                    "training_time": "~20 hours",
                    "steps": "~65K"
                },
                "medium": {
                    "target_tokens": 5_000_000_000,  # 5B tokens
                    "description": "Medium training run",
                    "training_time": "~4 days",
                    "steps": "~325K"
                },
                "large": {
                    "target_tokens": 18_500_000_000,  # 18.5B tokens (minimum for 926M)
                    "description": "Proper training (minimum viable)",
                    "training_time": "~2 weeks",
                    "steps": "~1.2M"
                },
                "production": {
                    "target_tokens": 46_000_000_000,  # 46B tokens (optimal for 926M)
                    "description": "Production training (optimal)",
                    "training_time": "~1 month",
                    "steps": "~3M"
                }
            }

    def get_samples_for_tokens(self, target_tokens: int) -> int:
        """Calculate how many samples we need for target tokens"""
        avg_tokens_per_sample = 200  # Conservative estimate for FineWeb
        safety_margin = 1.5  # 50% extra for multiple epochs
        return int((target_tokens * safety_margin) // avg_tokens_per_sample)

# === SYSTEM CONFIGURATION ===
@dataclass
class SystemConfig:
    """System and Environment Configuration."""

    # Paths
    model_save_path: str = "checkpoints"
    final_model_name: str = "final_model.pt"
    log_dir: str = "logs"
    cache_dir: str = "cache"

    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    benchmark: bool = False  # cudnn.benchmark

    # Debugging
    debug_mode: bool = False
    profile_memory: bool = False
    detect_anomaly: bool = False  # autograd anomaly detection

# %%
# === GLOBAL CONFIG INSTANCES ===
# Diese werden von allen Komponenten verwendet
model_config = ModelConfig()
training_config = TrainingConfig()
inference_config = InferenceConfig()
hardware_config = HardwareConfig()
system_config = SystemConfig()
dataset_config = DatasetConfig()
