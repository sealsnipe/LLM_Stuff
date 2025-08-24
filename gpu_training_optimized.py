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
import math
from typing import Dict, Optional
from dataclasses import dataclass

# %%
@dataclass
class GPUTrainingConfig:
    """GPU-optimierte Training-Konfiguration."""
    # Model parameters
    vocab_size: int = 32000
    hidden_size: int = 2048
    num_attention_heads: int = 32
    num_key_value_heads: int = 8  # GQA für Memory-Effizienz
    num_layers: int = 24
    max_position_embeddings: int = 4096
    
    # Training parameters - GPU-optimiert
    max_steps: int = 50000  # ← HIER sind deine max_steps!
    batch_size: int = 16    # Größere Batch für GPU
    gradient_accumulation_steps: int = 8  # Effektive batch_size = 128
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    
    # GPU-spezifische Einstellungen
    use_mixed_precision: bool = True  # FP16 für GPU
    max_grad_norm: float = 1.0
    sequence_length: int = 1024  # Längere Sequenzen auf GPU
    
    # Memory-Optimierungen
    gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    
    # Logging
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000

def check_gpu_setup():
    """Überprüft GPU-Setup und gibt Empfehlungen."""
    print("=== GPU SETUP CHECK ===")
    
    if not torch.cuda.is_available():
        print("❌ CUDA nicht verfügbar!")
        print("💡 Für LLM-Training brauchst du eine NVIDIA GPU mit CUDA")
        print("🔧 Installiere: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"✅ CUDA verfügbar: {torch.version.cuda}")
    print(f"🖥️  GPUs gefunden: {gpu_count}")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1e9
        print(f"   GPU {i}: {props.name}")
        print(f"   Memory: {memory_gb:.1f} GB")
        print(f"   Compute: {props.major}.{props.minor}")
        
        # Memory-Empfehlungen
        if memory_gb < 8:
            print(f"   ⚠️  Wenig VRAM - reduziere batch_size")
        elif memory_gb < 16:
            print(f"   ✅ Ausreichend für mittlere Modelle")
        else:
            print(f"   🚀 Perfekt für große Modelle")
    
    return True

# %%
class GPUOptimizedAttention(nn.Module):
    """GPU-optimierte Attention mit Memory-Effizienz."""
    
    def __init__(self, config: GPUTrainingConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = 1.0 / math.sqrt(self.head_dim)
        
        # Projections - optimiert für GPU
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        
        # Dropout für Training
        self.attention_dropout = nn.Dropout(0.1)
    
    def forward(self, hidden_states, attention_mask=None):
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
        
        # Attention computation - GPU-optimiert
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scaling
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.attention_dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape und Output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        
        return self.o_proj(attn_output)

# %%
class GPUOptimizedTransformerBlock(nn.Module):
    """GPU-optimierter Transformer Block."""
    
    def __init__(self, config: GPUTrainingConfig):
        super().__init__()
        self.attention = GPUOptimizedAttention(config)
        
        # Feed Forward - größer für bessere GPU-Auslastung
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4, bias=False),
            nn.GELU(),  # GELU ist GPU-freundlicher als SiLU
            nn.Linear(config.hidden_size * 4, config.hidden_size, bias=False),
            nn.Dropout(0.1)
        )
        
        # Layer Norms
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=1e-5)
    
    def forward(self, hidden_states, attention_mask=None):
        # Pre-norm attention
        normed_states = self.attention_norm(hidden_states)
        attn_output = self.attention(normed_states, attention_mask)
        hidden_states = hidden_states + attn_output
        
        # Pre-norm feed forward
        normed_states = self.ffn_norm(hidden_states)
        ffn_output = self.feed_forward(normed_states)
        hidden_states = hidden_states + ffn_output
        
        return hidden_states

# %%
class GPUOptimizedLLM(nn.Module):
    """GPU-optimiertes LLM für Training."""
    
    def __init__(self, config: GPUTrainingConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            GPUOptimizedTransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Output
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Gradient checkpointing für Memory-Effizienz
        if config.gradient_checkpointing:
            self.gradient_checkpointing_enable()
        
        # Weight initialization
        self.apply(self._init_weights)
    
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
        
        # Transformer layers
        for layer in self.layers:
            if self.config.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, attention_mask, use_reentrant=False
                )
            else:
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
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        
        return {"loss": loss, "logits": logits}

# %%
def create_gpu_optimized_dataset(config: GPUTrainingConfig, num_samples: int = 10000):
    """Erstellt GPU-optimierten Datensatz."""
    # Größere Sequenzen für GPU
    input_ids = torch.randint(0, config.vocab_size, (num_samples, config.sequence_length))
    labels = input_ids.clone()
    
    dataset = TensorDataset(input_ids, labels)
    return DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=4,  # Parallel data loading
        pin_memory=True  # Faster GPU transfer
    )

# %%
def gpu_training_loop(config: GPUTrainingConfig):
    """GPU-optimierte Training-Loop."""
    
    # GPU Setup Check
    if not check_gpu_setup():
        print("🚫 GPU Training nicht möglich - verwende CPU-Version")
        return
    
    device = torch.device("cuda")
    print(f"\n🚀 Starting GPU Training on {torch.cuda.get_device_name()}")
    print(f"📊 Max Steps: {config.max_steps}")
    print(f"🔄 Effective Batch Size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"💾 Mixed Precision: {config.use_mixed_precision}")
    print()
    
    # Model
    model = GPUOptimizedLLM(config).to(device)
    
    # Optimizer - GPU-optimiert
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),  # Bessere Werte für LLMs
        eps=1e-8
    )
    
    # Mixed Precision Scaler
    scaler = torch.cuda.amp.GradScaler() if config.use_mixed_precision else None
    
    # Dataset
    dataloader = create_gpu_optimized_dataset(config)
    data_iter = iter(dataloader)
    
    # Training state
    model.train()
    total_loss = 0.0
    step = 0
    
    print("🎯 GPU Training gestartet...")
    
    # TRAINING LOOP mit max_steps
    while step < config.max_steps:
        optimizer.zero_grad()
        accumulated_loss = 0.0
        
        # Gradient accumulation
        for micro_step in range(config.gradient_accumulation_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            input_ids, labels = [x.to(device, non_blocking=True) for x in batch]
            
            # Mixed precision forward pass
            if config.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids, labels=labels)
                    loss = outputs["loss"] / config.gradient_accumulation_steps
                
                scaler.scale(loss).backward()
            else:
                outputs = model(input_ids, labels=labels)
                loss = outputs["loss"] / config.gradient_accumulation_steps
                loss.backward()
            
            accumulated_loss += loss.item()
        
        # Optimizer step
        if config.use_mixed_precision:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
        
        step += 1
        total_loss += accumulated_loss
        
        # Logging
        if step % config.log_interval == 0:
            avg_loss = total_loss / config.log_interval
            gpu_memory = torch.cuda.max_memory_allocated() / 1e9
            print(f"Step {step:5d}/{config.max_steps} | Loss: {avg_loss:.4f} | GPU Memory: {gpu_memory:.1f}GB")
            total_loss = 0.0
            torch.cuda.reset_peak_memory_stats()
        
        # Early stopping für Demo
        if step >= 200:  # Nur 200 Steps für Demo
            print(f"🛑 Demo beendet nach {step} Steps")
            break
    
    print(f"✅ Training completed!")

# %%
if __name__ == "__main__":
    # GPU-optimierte Konfiguration
    config = GPUTrainingConfig(
        max_steps=10000,     # ← max_steps für GPU Training
        batch_size=8,        # GPU-optimiert
        gradient_accumulation_steps=4,
        hidden_size=1024,    # Kleineres Modell für Demo
        num_layers=12,
        sequence_length=512,
        use_mixed_precision=True,
        gradient_checkpointing=True
    )
    
    print("=== GPU-OPTIMIERTE TRAINING KONFIGURATION ===")
    print(f"Max Steps: {config.max_steps}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Gradient Accumulation: {config.gradient_accumulation_steps}")
    print(f"Effective Batch Size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"Mixed Precision: {config.use_mixed_precision}")
    print(f"Gradient Checkpointing: {config.gradient_checkpointing}")
    print()
    
    # Start GPU training
    gpu_training_loop(config)
