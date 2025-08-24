# %% [markdown]
# # Training Loop für Llama 4 Attention
#
# Diese Datei zeigt, wie die Attention-Mechanismen in einem echten Training-Loop
# verwendet werden, inklusive max_steps, batch_size, learning rate scheduling, etc.

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math
from typing import Dict, List
from dataclasses import dataclass

# Import unsere professionelle Attention-Implementierung
from attention_code_professional import ProfessionalLlama4Attention, AttentionConfig

# %%
@dataclass
class TrainingConfig:
    """Training-spezifische Konfiguration."""
    # Model parameters
    vocab_size: int = 32000
    hidden_size: int = 1024
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    num_layers: int = 12
    max_position_embeddings: int = 2048
    
    # Training parameters
    max_steps: int = 10000  # Hier sind deine max_steps!
    batch_size: int = 8     # Batch size pro GPU
    gradient_accumulation_steps: int = 4  # Effektive batch_size = batch_size * grad_accum
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    
    # Sequence parameters
    sequence_length: int = 512
    
    # Logging
    log_interval: int = 100
    eval_interval: int = 500
    save_interval: int = 1000

# %%
class SimpleLlamaBlock(nn.Module):
    """Vereinfachter Llama Block mit unserer Attention."""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        
        # Attention configuration
        attn_config = AttentionConfig(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            use_flash_attention=True,
            use_kv_cache=False  # Kein Cache während Training
        )
        
        self.attention = ProfessionalLlama4Attention(attn_config)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.SiLU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size)
        )
        
        # Layer normalization
        self.attention_norm = nn.RMSNorm(config.hidden_size)
        self.ffn_norm = nn.RMSNorm(config.hidden_size)
    
    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        # Pre-norm attention
        normed_hidden_states = self.attention_norm(hidden_states)
        attn_output, _, _ = self.attention(
            normed_hidden_states, 
            attention_mask=attention_mask, 
            position_ids=position_ids
        )
        hidden_states = hidden_states + attn_output
        
        # Pre-norm feed forward
        normed_hidden_states = self.ffn_norm(hidden_states)
        ffn_output = self.feed_forward(normed_hidden_states)
        hidden_states = hidden_states + ffn_output
        
        return hidden_states

# %%
class SimpleLlamaModel(nn.Module):
    """Vereinfachtes Llama Modell für Training-Demo."""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            SimpleLlamaBlock(config) for _ in range(config.num_layers)
        ])
        
        # Output
        self.norm = nn.RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size, seq_len = input_ids.shape
        
        # Create position ids
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Token embeddings
        hidden_states = self.token_embeddings(input_ids)
        
        # Create causal mask if not provided
        if attention_mask is None:
            attention_mask = torch.triu(
                torch.full((seq_len, seq_len), float('-inf'), device=input_ids.device), 
                diagonal=1
            ).unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
        
        # Pass through transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_ids)
        
        # Final norm and projection
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        
        return {"loss": loss, "logits": logits}

# %%
def create_dummy_dataset(config: TrainingConfig, num_samples: int = 1000):
    """Erstellt einen Dummy-Datensatz für Training."""
    # Generiere zufällige Token-Sequenzen
    input_ids = torch.randint(0, config.vocab_size, (num_samples, config.sequence_length))
    
    # Labels sind die gleichen wie input_ids (für Language Modeling)
    labels = input_ids.clone()
    
    dataset = TensorDataset(input_ids, labels)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

# %%
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Cosine learning rate schedule mit warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# %%
def train_model(config: TrainingConfig):
    """Haupttraining-Loop mit max_steps."""
    # GPU Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    if device.type == "cpu":
        print("⚠️  WARNING: Training auf CPU ist sehr langsam!")
        print("💡 Für echtes LLM-Training solltest du eine GPU verwenden.")
        print("🔧 Reduziere batch_size und max_steps für CPU-Demo...")
        # CPU-optimierte Einstellungen
        config.batch_size = min(config.batch_size, 2)
        config.max_steps = min(config.max_steps, 100)
        config.sequence_length = min(config.sequence_length, 128)
    else:
        print(f"🚀 GPU Training aktiviert: {torch.cuda.get_device_name()}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"🚀 Starting Training with max_steps = {config.max_steps}")
    print(f"📊 Batch size: {config.batch_size}")
    print(f"🔄 Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"📈 Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print()

    # Model und Optimizer
    model = SimpleLlamaModel(config).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        config.warmup_steps, 
        config.max_steps
    )
    
    # Dataset
    dataloader = create_dummy_dataset(config)
    data_iter = iter(dataloader)
    
    # Training state
    model.train()
    total_loss = 0.0
    step = 0
    
    print("🎯 Training Loop gestartet...")
    
    # HIER IST DER WICHTIGE TEIL: Training bis max_steps
    while step < config.max_steps:
        
        # Gradient accumulation loop
        optimizer.zero_grad()
        accumulated_loss = 0.0
        
        for micro_step in range(config.gradient_accumulation_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                # Dataset durchlaufen, neu starten
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            input_ids, labels = batch
            
            # Forward pass
            outputs = model(input_ids, labels=labels)
            loss = outputs["loss"] / config.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            accumulated_loss += loss.item()
        
        # Optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Update counters
        step += 1
        total_loss += accumulated_loss
        
        # Logging
        if step % config.log_interval == 0:
            avg_loss = total_loss / config.log_interval
            current_lr = scheduler.get_last_lr()[0]
            print(f"Step {step:5d}/{config.max_steps} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")
            total_loss = 0.0
        
        # Evaluation (vereinfacht)
        if step % config.eval_interval == 0:
            print(f"📊 Evaluation at step {step} (simplified)")
        
        # Checkpoint saving (vereinfacht)
        if step % config.save_interval == 0:
            print(f"💾 Checkpoint saved at step {step}")
    
    print(f"✅ Training completed! Reached max_steps = {config.max_steps}")

# %%
if __name__ == "__main__":
    # Training configuration
    config = TrainingConfig(
        max_steps=2000,      # Hier definierst du max_steps!
        batch_size=4,        # Kleinere batch_size für Demo
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        sequence_length=256,
        num_layers=6,        # Kleineres Modell für Demo
        log_interval=50,
        eval_interval=200,
        save_interval=500
    )
    
    print("=== TRAINING CONFIGURATION ===")
    print(f"Max Steps: {config.max_steps}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Gradient Accumulation: {config.gradient_accumulation_steps}")
    print(f"Effective Batch Size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Sequence Length: {config.sequence_length}")
    print(f"Model Layers: {config.num_layers}")
    print()
    
    # Start training
    train_model(config)
