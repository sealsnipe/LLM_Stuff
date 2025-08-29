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
    intermediate_size: int = int(1536 * 8/3)  # SwiGLU standard expansion (8/3 ≈ 2.67 → 4096)
    max_position_embeddings: int = 2048

    # Modern Features
    use_gqa: bool = True  # Grouped-Query Attention
    use_rope: bool = True  # Rotary Position Embeddings
    use_swiglu: bool = True  # SwiGLU activation
    use_qk_norm: bool = True  # QK Normalization
    tie_word_embeddings: bool = True  # Weight tying

    # Dropout Settings (GPT-5: Reduziert für Pretraining)
    dropout_prob: float = 0.0  # GPT-5: 0.0-0.05 für Pretraining
    attention_dropout: float = 0.0  # GPT-5: 0.0 für Flash-Attention Pfad
    hidden_dropout: float = 0.0  # GPT-5: 0.0-0.05 für Pretraining
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

    # TRAINING CONTROL - Epoch-based or Token-based
    use_epoch_based_training: bool = True  # True: epoch-based, False: token-based
    target_epochs: int = 5  # Anzahl Epochs (≥3, nur ganze Zahlen)
    epoch_dataset_fraction: float = 0.8  # Anteil des Datasets pro Epoch (4/5 = 0.8)
    target_tokens: int = 2_200_000_000  # 2.2B Tokens (nur wenn use_epoch_based_training=False)
    sequence_length: int = 512  # Cache-kompatibel (wird vom Cache überschrieben)

    # Training Parameters - OPTIMIZED for better GPU utilization
    batch_size: int = 12  # Increased from 8 (push GPU harder)
    gradient_accumulation_steps: int = 6  # Reduced from 8 (effective batch: 12×6=72)
    learning_rate: float = 2.55e-4  # Reduziert von 3e-4 für Clip-Rate 10-15%
    weight_decay: float = 0.1
    max_grad_norm: float = 0.5  # Tighter clipping

    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate epoch-based training
        if self.use_epoch_based_training:
            if not isinstance(self.target_epochs, int):
                raise ValueError("target_epochs must be an integer")
            if self.target_epochs < 3:
                raise ValueError("target_epochs must be ≥3 for proper training")
            if self.target_epochs > 50:
                raise ValueError("target_epochs must be ≤50 (practical limit)")
            if not (0.1 <= self.epoch_dataset_fraction <= 1.0):
                raise ValueError("epoch_dataset_fraction must be between 0.1 and 1.0")

        # Validate token-based training
        if not self.use_epoch_based_training:
            if self.target_tokens < 1_000_000:
                raise ValueError("target_tokens must be ≥1M for meaningful training")

    # Calculated automatically from target_tokens or target_epochs
    @property
    def max_steps(self) -> int:
        """Calculate steps needed for target tokens or epochs"""
        effective_batch_size = self.batch_size * self.gradient_accumulation_steps
        tokens_per_step = effective_batch_size * self.sequence_length

        if self.use_epoch_based_training:
            # Epoch-based: calculate dynamically with dataset size
            try:
                from core.utils.dataset_utils import get_dynamic_max_steps
                return get_dynamic_max_steps()
            except ImportError:
                # Fallback if dataset_utils not available
                return self.target_tokens // tokens_per_step
        else:
            # Token-based: use target_tokens
            return self.target_tokens // tokens_per_step

    @property
    def dynamic_warmup_steps(self) -> int:
        """Calculate dynamic warmup steps based on max_steps"""
        try:
            from core.utils.dataset_utils import get_dynamic_warmup_steps
            return get_dynamic_warmup_steps()
        except ImportError:
            # Fallback to static warmup_steps
            return self.warmup_steps

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
    warmup_steps: int = 1000  # GPT-5: Erhöht von 100 (1.5% von ~65k steps)
    min_lr_ratio: float = 0.05  # GPT-5: Reduziert von 0.1 für aggressivere Decay

    # Performance Optimizations
    use_torch_compile: bool = True  # Intelligent: Nur wenn Triton verfügbar
    torch_compile_mode: str = "reduce-overhead"  # Besser für statische Shapes
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

    # Debug System
    debug_mode: bool = False  # Enable detailed debug logs
    debug_log_file: str = "debug_training.log"  # Debug log file
    error_log_file: str = "error_training.log"  # Error log file

    # Validation Settings (GPT-5: Für Generalisierung überwachen)
    use_validation: bool = True
    validation_split: float = 0.02  # 2% für Validation
    validation_interval: int = 500  # Validation alle 500 Steps

    # Performance Optimizations
    use_sequence_packing: bool = True  # Sequence Packing für +20-40% Durchsatz (AKTIVIERT für bessere Effizienz)
    fast_loading_mode: bool = True  # Schnelles Loading ohne komplexes Packing
    use_length_bucketing: bool = True  # Length Bucketing für weniger Padding
    use_cuda_graphs: bool = True  # CUDA Graphs für statische Shapes
    use_tf32: bool = True  # TF32 für RTX 3090/4090

    # VRAM Optimizations
    use_8bit_adam: bool = True  # 8-bit Adam States für >1GB VRAM Ersparnis
    selective_checkpointing: str = "attention"  # "full", "attention", "mlp", "none"

    # Advanced Optimizations
    use_fused_cross_entropy: bool = True  # Fused CrossEntropy für Speed
    log_padding_efficiency: bool = True  # Logge Padding-Quote pro Batch

    # Production Features
    use_ema: bool = True  # Exponential Moving Average für glatte Eval-Kurven
    ema_decay: float = 0.999  # EMA Decay Rate

    # Logging
    log_every: int = 10  # JSON Logging-Intervall
    validation_interval: int = 200  # Validation alle N Steps
    production_monitoring: bool = True  # Advanced Metrics & Monitoring

    # Token-basierte Schedule (präziser als Steps)
    use_token_based_schedule: bool = True  # LR-Schedule basierend auf echten Tokens
    target_real_tokens: int = 1_000_000_000  # 1B echte Tokens (ohne Padding)

    # Data Loading (Optimiert für RTX 3090/4090)
    dataloader_num_workers: int = 8  # Erhöht für bessere I/O
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = True
    dataloader_prefetch_factor: int = 8  # Erhöht für weniger GPU-Idle

    # 🚀 PACKED CACHE SETTINGS
    use_packed_cache: bool = True  # Verwende Packed Cache wenn verfügbar
    packed_cache_dir: str = "cache/packed_sequences"  # Pfad zum Packed Cache
    fallback_to_parquet: bool = True  # Fallback zu Parquet wenn Cache nicht da
    packed_cache_validation: bool = True  # Validiere Cache-Integrität beim Laden

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
    fineweb_num_samples: int = 11002672  # 11M samples = ALLE verfügbaren Samples (exakte Anzahl)

    # Dataset Size Profile Selection
    # Verfügbare Profile:
    # - "tiny": 1k samples, ~5 min training (quick testing)
    # - "small": 10k samples, ~30 min training (development)
    # - "medium": 300k samples, ~15 hours training (serious training)
    # - "large": 1M samples, ~2 days training (production-ready)
    # - "production": 10M samples, ~1-2 weeks training (full production)
    # - "full": All samples, ~months training (complete dataset)
    default_dataset_size: str = "full"  # GANZES FineWeb Dataset!

    # Dataset Size Presets
    dataset_sizes: dict = None

    def __post_init__(self):
        if self.dataset_sizes is None:
            # NEW: Token-based dataset configuration
            # Based on target_tokens from TrainingConfig
            self.dataset_sizes = {
                "test": {
                    "target_tokens": 10_000_000,  # 10M tokens (RAM-sicher!)
                    "description": "RAM-safe test run",
                    "training_time": "~20 minutes",
                    "steps": "~650"
                },
                "small": {
                    "target_tokens": 50_000_000,  # 50M tokens (RAM-sicher)
                    "description": "Small RAM-safe run",
                    "training_time": "~2 hours",
                    "steps": "~3.2K"
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
                },
                "full": {
                    "target_tokens": 2_200_000_000,  # 2.2B tokens = GANZES FineWeb Dataset
                    "description": "Complete FineWeb dataset (11M samples)",
                    "training_time": "~1 week",
                    "steps": "~143K"
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
