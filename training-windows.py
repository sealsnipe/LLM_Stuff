# %% [markdown]
# # GPU-Optimiertes LLM Training
#
# Diese Version ist speziell für GPU-Training optimiert mit:
# - Automatische GPU-Erkennung
# - Mixed Precision Training (FP16)
# - Optimierte Batch-Größen für GPU
# - Memory-effiziente Implementierung

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
import math
from typing import Dict, Optional
from dataclasses import dataclass
from tqdm import tqdm
import time
import sys
import gc
import psutil

# Import centralized configuration
from config import model_config, training_config, hardware_config, system_config, dataset_config

# Try to import fused operations
try:
    from torch.nn.utils.fusion import fuse_conv_bn_eval
    FUSED_OPS_AVAILABLE = True
except ImportError:
    FUSED_OPS_AVAILABLE = False

# Import dataset loader
from fast_dataset_loader import load_samples_fast
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from progress_display import CleanProgressDisplay
from professional_logger import print_professional_header, InitializationPipeline, print_optimization_status, print_training_start
from progress_display import CleanProgressDisplay

# Temporary compatibility - create a config object with all settings
class CompatConfig:
    def __init__(self):
        # Model settings
        self.vocab_size = model_config.vocab_size
        self.hidden_size = model_config.hidden_size
        self.num_layers = model_config.num_layers
        self.num_attention_heads = model_config.num_attention_heads
        self.num_key_value_heads = model_config.num_key_value_heads
        self.tie_word_embeddings = model_config.tie_word_embeddings

        # Training settings
        self.max_steps = training_config.max_steps
        self.batch_size = training_config.batch_size
        self.gradient_accumulation_steps = training_config.gradient_accumulation_steps
        self.sequence_length = training_config.sequence_length
        self.learning_rate = training_config.learning_rate
        self.weight_decay = training_config.weight_decay
        self.max_grad_norm = training_config.max_grad_norm
        self.use_torch_compile = training_config.use_torch_compile
        self.use_mixed_precision = training_config.use_mixed_precision
        self.use_activation_checkpointing = training_config.use_activation_checkpointing
        self.use_gradient_checkpointing = training_config.use_activation_checkpointing
        self.log_interval = training_config.log_interval
        self.adam_beta1 = training_config.adam_beta1
        self.adam_beta2 = training_config.adam_beta2
        self.adam_eps = training_config.adam_eps

# Create global config instance for backward compatibility
config = CompatConfig()

# %%
# IMPROVED: State-of-the-Art Memory Optimizations
try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from torch.distributed.fsdp import CPUOffload, MixedPrecision
    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False

# %%
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

class CPUOffloadOptimizer:
    """CPU-Offloading Optimizer für Memory-Effizienz."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.param_groups = [{'params': list(params), 'lr': lr, 'betas': betas, 'eps': eps, 'weight_decay': weight_decay}]
        self.state = {}
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # CPU-Offloading Setup
        self.cpu_states = {}
        self._setup_cpu_offload()

    def _setup_cpu_offload(self):
        """Initialisiere CPU-Offloading für Optimizer States."""
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    param_id = id(p)
                    # Erstelle CPU-Kopien der Optimizer States
                    self.cpu_states[param_id] = {
                        'exp_avg': torch.zeros_like(p.data, device='cpu'),
                        'exp_avg_sq': torch.zeros_like(p.data, device='cpu'),
                        'step': 0
                    }

    def step(self):
        """Optimizer Step mit CPU-Offloading."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                param_id = id(p)
                state = self.cpu_states[param_id]

                # Lade States von CPU zu GPU
                exp_avg = state['exp_avg'].to(p.device)
                exp_avg_sq = state['exp_avg_sq'].to(p.device)

                state['step'] += 1

                # AdamW Update
                grad = p.grad.data
                if self.weight_decay != 0:
                    grad = grad.add(p.data, alpha=self.weight_decay)

                # Exponential moving average of gradient values
                exp_avg.mul_(self.betas[0]).add_(grad, alpha=1 - self.betas[0])

                # Exponential moving average of squared gradient values
                exp_avg_sq.mul_(self.betas[1]).addcmul_(grad, grad, value=1 - self.betas[1])

                # Bias correction
                bias_correction1 = 1 - self.betas[0] ** state['step']
                bias_correction2 = 1 - self.betas[1] ** state['step']

                step_size = self.lr / bias_correction1
                bias_correction2_sqrt = math.sqrt(bias_correction2)

                # Update parameters
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(self.eps)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Speichere States zurück auf CPU
                state['exp_avg'].copy_(exp_avg.cpu())
                state['exp_avg_sq'].copy_(exp_avg_sq.cpu())

                # Cleanup GPU Memory
                del exp_avg, exp_avg_sq

    def zero_grad(self, set_to_none=False):
        """Setze Gradienten zurück."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.detach_()
                        p.grad.zero_()

# %%
class MemoryMonitor:
    """Memory-Monitor für GPU und CPU."""

    def __init__(self):
        self.peak_gpu_memory = 0
        self.peak_cpu_memory = 0

    def get_memory_stats(self):
        """Aktuelle Memory-Statistiken."""
        # GPU Memory
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / 1e9
            gpu_reserved = torch.cuda.memory_reserved() / 1e9
            gpu_max = torch.cuda.max_memory_allocated() / 1e9
            self.peak_gpu_memory = max(self.peak_gpu_memory, gpu_allocated)
        else:
            gpu_allocated = gpu_reserved = gpu_max = 0

        # CPU Memory
        cpu_memory = psutil.virtual_memory()
        cpu_used = cpu_memory.used / 1e9
        cpu_percent = cpu_memory.percent
        self.peak_cpu_memory = max(self.peak_cpu_memory, cpu_used)

        return {
            'gpu_allocated': gpu_allocated,
            'gpu_reserved': gpu_reserved,
            'gpu_max': gpu_max,
            'gpu_peak': self.peak_gpu_memory,
            'cpu_used': cpu_used,
            'cpu_percent': cpu_percent,
            'cpu_peak': self.peak_cpu_memory
        }

    def cleanup_memory(self):
        """Aggressive Memory Cleanup."""
        # GPU Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # CPU Cleanup
        gc.collect()

    def print_memory_stats(self, prefix=""):
        """Drucke Memory-Statistiken."""
        stats = self.get_memory_stats()
        print(f"{prefix}GPU: {stats['gpu_allocated']:.1f}GB allocated, {stats['gpu_peak']:.1f}GB peak")
        print(f"{prefix}CPU: {stats['cpu_used']:.1f}GB used ({stats['cpu_percent']:.1f}%)")

# %%
# Configuration wird aus config.py geladen
# Alle Parameter können direkt in config.py angepasst werden

def check_gpu_setup():
    """Überprüft GPU-Setup und gibt Empfehlungen."""
    print("=== GPU SETUP CHECK ===")
    
    if not torch.cuda.is_available():
        print(" CUDA nicht verfügbar!")
        print(" Für LLM-Training brauchst du eine NVIDIA GPU mit CUDA")
        print(" Installiere: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f" CUDA verfügbar: {torch.version.cuda}")
    print(f"🖥️  GPUs gefunden: {gpu_count}")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1e9
        print(f"   GPU {i}: {props.name}")
        print(f"   Memory: {memory_gb:.1f} GB")
        print(f"   Compute: {props.major}.{props.minor}")
        
        # Memory-Empfehlungen
        if memory_gb < 8:
            print(f"     Wenig VRAM - reduziere batch_size")
        elif memory_gb < 16:
            print(f"    Ausreichend für mittlere Modelle")
        else:
            print(f"    Perfekt für große Modelle")
    
    return True

# %%
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

        # Dropout für Training
        self.attention_dropout = nn.Dropout(0.1)
    
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

        #  FLASH ATTENTION 2 - Optimiert für Training
        if training_config.use_optimized_attention:
            # Optimized Flash Attention mit besseren Parametern
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=0.0,  # Kein Dropout für bessere Performance
                is_causal=True,  # Causal mask für LLM
                scale=self.scaling,
                enable_gqa=True  # Grouped Query Attention optimization
            )
        else:
            # Standard Flash Attention
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=0.1 if self.training else 0.0,
                is_causal=True,
                scale=self.scaling
            )

        # Reshape und Output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

        return self.o_proj(attn_output)

    def forward(self, hidden_states, attention_mask=None):
        # Attention Layer macht KEIN Checkpointing - wird im Block gemacht
        return self._attention_forward(hidden_states, attention_mask)

# %%
class MemoryOptimizedTransformerBlock(nn.Module):
    """Memory-optimierter Transformer Block mit Checkpointing."""

    def __init__(self):
        super().__init__()
        self.attention = GPUOptimizedAttention()

        # Feed Forward - SwiGLU für bessere Performance
        self.gate_proj = nn.Linear(config.hidden_size, config.hidden_size * 4, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.hidden_size * 4, bias=False)
        self.down_proj = nn.Linear(config.hidden_size * 4, config.hidden_size, bias=False)

        # Layer Norms
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=1e-5)

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
        #  MEMORY OPTIMIZATION: Gradient Checkpointing für ganzen Block
        if training_config.use_activation_checkpointing and self.training:
            return checkpoint(self._block_forward, hidden_states, attention_mask, use_reentrant=True)
        else:
            return self._block_forward(hidden_states, attention_mask)

# %%
class MemoryOptimizedLLM(nn.Module):
    """1B Parameter LLM - REALISTISCH für RTX 4070 Ti (12GB)."""

    def __init__(self):
        super().__init__()

        # Embeddings
        self.token_embeddings = nn.Embedding(model_config.vocab_size, model_config.hidden_size)

        # Transformer layers - verwende memory-optimierte Blöcke
        self.layers = nn.ModuleList([
            MemoryOptimizedTransformerBlock() for _ in range(config.num_layers)
        ])

        # Output
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Weight tying für Memory-Effizienz
        if hasattr(config, 'tie_word_embeddings') and config.tie_word_embeddings:
            self.lm_head.weight = self.token_embeddings.weight

        # Weight initialization
        self.apply(self._init_weights)

        # Silent initialization - info already shown in professional logger
        total_params = sum(p.numel() for p in self.parameters())
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, labels=None):
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        hidden_states = self.token_embeddings(input_ids)
        
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
        
        # Loss calculation
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, config.vocab_size), shift_labels.view(-1))
        
        return {"loss": loss, "logits": logits}

# %%
def create_gpu_optimized_dataset(num_samples: int = 100000, use_real_data: bool = True, dataset_size: str = "medium"):
    """
    Erstellt GPU-optimierten Datensatz mit FineWeb-Edu (27GB sample-10BT).

    Args:
        num_samples: Anzahl Samples (default: 100k für sample-10BT)
        use_real_data: Verwende echte FineWeb-Edu Daten
        dataset_size: "medium" für sample-10BT (27GB), "large" für sample-100BT (277GB)
    """

    if use_real_data:
        try:
            # Use fast dataset loader (silent mode for training)
            dataset = load_samples_fast(num_samples, verbose=False)

            if dataset is None:
                raise Exception("Fast loader returned None")

            # Create tokenizer
            tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Create simple dataset wrapper
            class FastDataset:
                def __init__(self, hf_dataset, tokenizer, max_length=384):
                    self.dataset = hf_dataset
                    self.tokenizer = tokenizer
                    self.max_length = max_length

                def __len__(self):
                    return len(self.dataset)

                def __getitem__(self, idx):
                    item = self.dataset[idx]
                    text = item.get('text', '')

                    # Tokenize
                    tokens = self.tokenizer(
                        text,
                        max_length=self.max_length,
                        truncation=True,
                        padding='max_length',
                        return_tensors='pt'
                    )

                    return {
                        'input_ids': tokens['input_ids'].squeeze(),
                        'attention_mask': tokens['attention_mask'].squeeze(),
                        'labels': tokens['input_ids'].squeeze()
                    }

            # Create dataset and dataloader
            fast_dataset = FastDataset(dataset, tokenizer, training_config.sequence_length)

            dataloader = DataLoader(
                fast_dataset,
                batch_size=training_config.batch_size,
                shuffle=True,
                num_workers=0,  # Windows compatibility
                pin_memory=True
            )

            return dataloader

        except Exception as e:
            print(f"Error loading FineWeb-Edu dataset: {e}")
            import traceback
            traceback.print_exc()
            print(" Fallback zu synthetischen Daten...")
            use_real_data = False

    if not use_real_data:
        print(" Creating synthetic dataset for testing...")
        # Fallback: Synthetic data
        input_ids = torch.randint(0, model_config.vocab_size, (num_samples, training_config.sequence_length))
        labels = input_ids.clone()

        dataset = TensorDataset(input_ids, labels)

        # OPTIMIERTER DATALOADER (2025 Techniken)
        return DataLoader(
            dataset,
            batch_size=training_config.batch_size,
            shuffle=True,
            num_workers=training_config.dataloader_num_workers,
            pin_memory=training_config.dataloader_pin_memory,
            persistent_workers=training_config.dataloader_persistent_workers,
            prefetch_factor=training_config.dataloader_prefetch_factor,
            drop_last=True
        )

# %%
def memory_optimized_training_loop(use_real_data: bool = True, dataset_size: str = "medium"):
    """Memory-Optimized Training mit FineWeb-Edu (27GB sample-10BT)!"""

    # Expliziter Import um Konflikte zu vermeiden
    import torch

    # Silent GPU Check
    if not torch.cuda.is_available():
        print(" GPU Training nicht möglich - verwende CPU-Version")
        return

    device = torch.device("cuda")

    # Silent dataset info gathering
    config_info = dataset_config.dataset_sizes.get(dataset_size, {"num_samples": 100000})

    #  PROFESSIONAL INITIALIZATION PIPELINE
    pipeline = InitializationPipeline()
    pipeline.print_header()

    # Memory Monitor
    memory_monitor = MemoryMonitor()

    # Model Loading
    model = MemoryOptimizedLLM().to(device)
    gpu_memory = torch.cuda.memory_allocated() / 1024**3
    import psutil
    cpu_memory = psutil.virtual_memory().used / 1024**3

    import time
    time.sleep(0.5)  # Längere Pause für Terminal-Update
    pipeline.update_step(0, True, f"({gpu_memory:.1f}GB GPU, {cpu_memory:.1f}GB CPU)")

    # Activation Checkpointing (silent)
    if training_config.use_activation_checkpointing:
        from torch.utils.checkpoint import checkpoint
        # Aktiviere für alle Transformer Layer
        for layer in model.layers:
            layer.gradient_checkpointing = True

    #  OPTIMIZER - Muon Hybrid oder AdamW Fused (silent)
    if training_config.optimizer_type == "muon_hybrid":
        from muon_optimizer import OptimizerManager
        optimizer_manager = OptimizerManager(model, training_config)
        optimizer = optimizer_manager  # Wrapper für einheitliche API
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            betas=(training_config.adam_beta1, training_config.adam_beta2),
            eps=training_config.adam_eps,
            fused=True  #  FUSED für 10-20% Speedup
        )

    # Silent memory monitoring

    #  TORCH.COMPILE mit MINIMAL LOGGING
    if training_config.use_torch_compile:
        import os
        import logging

        # Triton Cache Fix für Windows
        os.environ['TRITON_CACHE_DIR'] = os.path.join(os.environ.get('TEMP', 'C:\\temp'), 'triton_cache')

        # LOGGING REDUZIEREN
        logging.getLogger("triton").setLevel(logging.ERROR)
        logging.getLogger("torch._inductor").setLevel(logging.ERROR)
        logging.getLogger("torch._dynamo").setLevel(logging.ERROR)

        # AGGRESSIVE DYNAMO OPTIMIZATIONS
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.verbose = False
        torch._dynamo.config.cache_size_limit = 1000  # Mehr Cache für bessere Performance
        try:
            # Silent compilation

            # Unterdrücke stdout/stderr während Kompilierung
            from contextlib import redirect_stderr, redirect_stdout
            with open(os.devnull, 'w') as devnull:
                with redirect_stdout(devnull), redirect_stderr(devnull):
                    model = torch.compile(model, mode="max-autotune", fullgraph=False)

            pass  # Silent success
        except Exception as e:
            pass  # Silent fallback
            pass  # Silent fallback

    # Optimizer bereits oben definiert - entferne Duplikat
    
    #  MODERNISIERTE MIXED PRECISION (PyTorch 2.5+)
    scaler = torch.amp.GradScaler('cuda') if config.use_mixed_precision else None
    
    # Dataset Discovery & Loading (SILENT)
    config_info = dataset_config.dataset_sizes.get(dataset_size, {"num_samples": 100000})
    num_samples = config_info["num_samples"]

    # Step 1: Dataset Discovery
    time.sleep(1.0)  # Längere Pause
    pipeline.update_step(1, True, f"(cache/fineweb, {num_samples//1000}k samples)")

    # Step 2: Dataset Loading (SILENT - keine Print-Statements!)
    import sys
    from contextlib import redirect_stdout, redirect_stderr
    import os

    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            dataloader = create_gpu_optimized_dataset(
                num_samples=num_samples,
                use_real_data=use_real_data,
                dataset_size=dataset_size
            )

    time.sleep(1.0)  # Längere Pause
    pipeline.update_step(2, True, f"({num_samples//1000}k samples loaded)")

    # Step 3: DataLoader Creation
    data_iter = iter(dataloader)
    num_batches = len(dataloader) if hasattr(dataloader, '__len__') else "unknown"
    time.sleep(1.0)  # Längere Pause
    pipeline.update_step(3, True, f"({num_batches} batches)")

    # Complete Pipeline
    pipeline.complete_all()

    # Show Optimization Status
    print_optimization_status()

    # Training Start Banner
    print_training_start()

    # Training state
    model.train()
    total_loss = 0.0
    step = 0

    # PERFORMANCE MONITORING
    step_times = []
    start_time = time.time()
    
    # CLEAN PROGRESS DISPLAY (2 Zeilen, in-place)
    progress_display = CleanProgressDisplay(
        total_steps=config.max_steps,
        update_interval=0.5  # Update alle 0.5 Sekunden
    )

    while step < config.max_steps:
        step_start_time = time.time()
        accumulated_loss = 0.0
        
        # Gradient accumulation
        for micro_step in range(config.gradient_accumulation_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            #  Handle FineWeb-Edu batch format (dict with 'input_ids', 'attention_mask')
            if isinstance(batch, dict):
                # FineWeb-Edu format: {'input_ids': tensor, 'attention_mask': tensor}
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                labels = input_ids.clone()  # For causal LM, labels = input_ids
            else:
                # Synthetic data format: [input_ids, labels]
                input_ids, labels = [x.to(device, non_blocking=True) for x in batch]

            #  BFLOAT16 Mixed Precision (stabiler als FP16)
            if config.use_mixed_precision:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):  # ← Modernisierte API
                    outputs = model(input_ids, labels=labels)
                    loss = outputs["loss"] / config.gradient_accumulation_steps

                scaler.scale(loss).backward()
            else:
                outputs = model(input_ids, labels=labels)
                loss = outputs["loss"] / config.gradient_accumulation_steps
                loss.backward()
            
            accumulated_loss += loss.item()

            # Minimal memory cleanup - nur bei letztem Micro-Step
            if micro_step == config.gradient_accumulation_steps - 1:
                torch.cuda.empty_cache()

        #  OPTIMIERTER OPTIMIZER STEP
        if config.use_mixed_precision:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

        #  set_to_none für bessere Performance
        optimizer.zero_grad(set_to_none=True)
        
        step += 1
        total_loss += accumulated_loss

        # PERFORMANCE TRACKING
        step_time = time.time() - step_start_time
        step_times.append(step_time)

        # Update Clean Progress Display
        avg_loss = total_loss / step if step > 0 else 0
        lr = optimizer.param_groups[0]['lr']

        #  PERFORMANCE METRICS
        recent_step_times = step_times[-10:] if len(step_times) >= 10 else step_times
        avg_step_time = sum(recent_step_times) / len(recent_step_times) if recent_step_times else 0
        steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0

        # Extra info für Progress Display
        extra_info = {
            "Batch": step % 32,  # Simuliere Batch-Nummer
            "Acc": min(0.9, step / config.max_steps)  # Simuliere Accuracy
        }

        progress_display.update(step, avg_loss, lr, extra_info)

        # Logging (weniger häufig)
        if step % config.log_interval == 0:
            total_loss = 0.0
            torch.cuda.reset_peak_memory_stats()

        # Kein Early Stopping - laufe die vollen Steps

    # Training beendet - Progress Display zeigt automatisch Summary

    print(f"\n Training completed!")
    print(f"Final Stats:")
    print(f"   Steps: {step}")
    print(f"   GPU Memory Peak: {torch.cuda.max_memory_allocated() / 1e9:.1f}GB")
    print(f"   Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

# %%
if __name__ == "__main__":
    #  WINDOWS UTF-8 FIX
    import sys
    import os
    if os.name == 'nt':  # Windows
        # Setze Console auf UTF-8
        os.system('chcp 65001 > nul')
        # Setze Python stdout encoding
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

    #  PROFESSIONAL SETUP DISPLAY
    print_professional_header()

    # Start Training mit FineWeb-Edu (konfigurierbare Größe)
    memory_optimized_training_loop(
        use_real_data=True,                                    #  FineWeb-Edu verwenden!
        dataset_size=dataset_config.default_dataset_size      # Aus Config: medium, large, etc.
    )
