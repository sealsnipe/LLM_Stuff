# %% [markdown]
# # LLM Training Configuration
#
# Zentrale Konfigurationsdatei für alle Parameter des LLM-Training-Systems.
# Einfach die Werte hier ändern - werden automatisch von allen Komponenten verwendet.

# %%
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

# === TRAINING CONFIGURATION ===
@dataclass
class TrainingConfig:
    """Training Configuration."""

    # Training Parameters
    max_steps: int = 2000
    batch_size: int = 5
    gradient_accumulation_steps: int = 8  # Effective batch size: 40
    sequence_length: int = 384
    learning_rate: float = 5e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0

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
    use_activation_checkpointing: bool = False  # Memory vs Speed tradeoff

    # Monitoring & Logging
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 1000
    max_checkpoints_to_keep: int = 3

    # Data Loading
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = True
    dataloader_prefetch_factor: int = 2

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
    fineweb_cache_dir: str = "cache/fineweb"
    fineweb_streaming: bool = True
    fineweb_num_samples: int = 10000  # Default sample size

    # Dataset Size Presets
    dataset_sizes: dict = None

    def __post_init__(self):
        if self.dataset_sizes is None:
            self.dataset_sizes = {
                "tiny": {
                    "num_samples": 1000,
                    "description": "Tiny sample for quick testing",
                    "estimated_tokens": "~200K tokens",
                    "training_time": "~5 minutes"
                },
                "small": {
                    "num_samples": 10000,
                    "description": "Small sample for development",
                    "estimated_tokens": "~2M tokens",
                    "training_time": "~30 minutes"
                },
                "medium": {
                    "num_samples": 100000,
                    "description": "Medium sample (FineWeb-Edu sample-10BT)",
                    "estimated_tokens": "~20M tokens",
                    "training_time": "~5 hours",
                    "dataset_size": "~27GB"
                },
                "large": {
                    "num_samples": 1000000,
                    "description": "Large sample (FineWeb-Edu sample-100BT)",
                    "estimated_tokens": "~200M tokens",
                    "training_time": "~2 days",
                    "dataset_size": "~277GB"
                },
                "production": {
                    "num_samples": 10000000,
                    "description": "Production training (FineWeb-Edu sample-350BT)",
                    "estimated_tokens": "~2B tokens",
                    "training_time": "~1-2 weeks",
                    "dataset_size": "~388GB"
                },
                "full": {
                    "num_samples": None,
                    "description": "Full FineWeb-Edu dataset",
                    "estimated_tokens": "~1.3T tokens",
                    "training_time": "~months",
                    "dataset_size": "~10.4TB"
                }
            }

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
