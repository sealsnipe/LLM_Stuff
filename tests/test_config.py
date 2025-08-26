#!/usr/bin/env python3
# %% [markdown]
# # Configuration Test Script
#
# Testet die neue zentrale Konfiguration.

# %%
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import model_config, training_config, dataset_config, hardware_config, system_config

def test_config():
    """Teste die zentrale Konfiguration."""

    print("🔧 CONFIGURATION TEST")
    print("=" * 60)

    # Berechne Parameter-Anzahl (grobe Schätzung)
    vocab_size = model_config.vocab_size
    hidden_size = model_config.hidden_size
    num_layers = model_config.num_layers
    intermediate_size = model_config.intermediate_size

    # Embedding parameters
    embedding_params = vocab_size * hidden_size

    # Attention parameters per layer
    attention_params_per_layer = hidden_size * hidden_size * 4

    # FFN parameters per layer
    ffn_params_per_layer = hidden_size * intermediate_size * 3

    # Layer norm parameters per layer
    ln_params_per_layer = hidden_size * 2

    # Total parameters
    total_params = (
        embedding_params +
        num_layers * (attention_params_per_layer + ffn_params_per_layer + ln_params_per_layer) +
        hidden_size
    )

    # Effective batch size
    effective_batch_size = training_config.batch_size * training_config.gradient_accumulation_steps

    print(f"📊 Model Architecture:")
    print(f"   Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"   Hidden Size: {hidden_size}")
    print(f"   Layers: {num_layers}")
    print(f"   Attention Heads: {model_config.num_attention_heads} (Q) / {model_config.num_key_value_heads} (KV)")
    print(f"   Sequence Length: {training_config.sequence_length}")
    print()
    print(f"🚀 Training Setup:")
    print(f"   Batch Size: {training_config.batch_size} (effective: {effective_batch_size})")
    print(f"   Learning Rate: {training_config.learning_rate}")
    print(f"   Max Steps: {training_config.max_steps}")
    print(f"   torch.compile: {training_config.use_torch_compile}")
    print(f"   Mixed Precision: {training_config.use_mixed_precision}")
    print()
    print(f"📊 Dataset Config:")
    print(f"   Default Size: {dataset_config.default_dataset_size}")
    print(f"   Available Profiles: {list(dataset_config.dataset_sizes.keys())}")
    print()
    print(f"🖥️  Hardware Config:")
    print(f"   Device: {hardware_config.device}")
    print(f"   GPU Memory Fraction: {hardware_config.gpu_memory_fraction}")
    print()
    print(f"⚙️  System Config:")
    print(f"   Seed: {system_config.seed}")
    print(f"   Model Save Path: {system_config.model_save_path}")

    print("\n✅ Configuration loaded successfully!")
    print("=" * 60)

def show_config_structure():
    """Zeige die Struktur der Konfiguration."""
    
    print("\n📋 CONFIGURATION STRUCTURE")
    print("=" * 60)
    
    # Lade Standard-Config
    model_config, training_config, inference_config, hardware_config, system_config = get_config("rtx_4070_ti_12gb")
    
    print("🏗️  ModelConfig:")
    for field, value in model_config.__dict__.items():
        print(f"   {field}: {value}")
    
    print("\n🚀 TrainingConfig:")
    for field, value in training_config.__dict__.items():
        print(f"   {field}: {value}")
    
    print("\n🎯 InferenceConfig:")
    for field, value in inference_config.__dict__.items():
        print(f"   {field}: {value}")
    
    print("\n🖥️  HardwareConfig:")
    for field, value in hardware_config.__dict__.items():
        print(f"   {field}: {value}")
    
    print("\n⚙️  SystemConfig:")
    for field, value in system_config.__dict__.items():
        print(f"   {field}: {value}")

def compare_presets():
    """Vergleiche verschiedene Presets."""
    
    print("\n📊 PRESET COMPARISON")
    print("=" * 80)
    
    presets = ["rtx_3060_12gb", "rtx_4070_ti_12gb", "rtx_4090_24gb"]
    
    print(f"{'Preset':<20} {'Parameters':<12} {'Hidden':<8} {'Layers':<8} {'Batch':<8} {'Seq Len':<8}")
    print("-" * 80)
    
    for preset in presets:
        model_config, training_config, _, _, _ = get_config(preset)
        
        # Berechne Parameter (grobe Schätzung)
        vocab_size = model_config.vocab_size
        hidden_size = model_config.hidden_size
        num_layers = model_config.num_layers
        intermediate_size = model_config.intermediate_size
        
        embedding_params = vocab_size * hidden_size
        attention_params_per_layer = hidden_size * hidden_size * 4
        ffn_params_per_layer = hidden_size * intermediate_size * 3
        ln_params_per_layer = hidden_size * 2
        
        total_params = (
            embedding_params +
            num_layers * (attention_params_per_layer + ffn_params_per_layer + ln_params_per_layer) +
            hidden_size
        )
        
        effective_batch = training_config.batch_size * training_config.gradient_accumulation_steps
        
        print(f"{preset:<20} {total_params/1e6:>8.1f}M {hidden_size:>8} {num_layers:>8} {effective_batch:>8} {training_config.sequence_length:>8}")

def memory_estimation():
    """Schätze Memory-Verbrauch für verschiedene Presets."""
    
    print("\n💾 MEMORY ESTIMATION")
    print("=" * 60)
    
    presets = ["rtx_3060_12gb", "rtx_4070_ti_12gb", "rtx_4090_24gb"]
    
    print(f"{'Preset':<20} {'Model (GB)':<12} {'Activations (GB)':<16} {'Total (GB)':<12}")
    print("-" * 60)
    
    for preset in presets:
        model_config, training_config, _, _, _ = get_config(preset)
        
        # Grobe Memory-Schätzung
        vocab_size = model_config.vocab_size
        hidden_size = model_config.hidden_size
        num_layers = model_config.num_layers
        batch_size = training_config.batch_size
        seq_len = training_config.sequence_length
        
        # Model parameters (FP16)
        total_params = vocab_size * hidden_size + num_layers * hidden_size * hidden_size * 8
        model_memory = total_params * 2 / 1e9  # FP16 = 2 bytes
        
        # Activations (grobe Schätzung)
        activation_memory = batch_size * seq_len * hidden_size * num_layers * 2 / 1e9
        
        total_memory = model_memory + activation_memory
        
        print(f"{preset:<20} {model_memory:>8.1f} {activation_memory:>12.1f} {total_memory:>8.1f}")

if __name__ == "__main__":
    print("🧪 TESTING LLM CONFIGURATION SYSTEM")
    print("=" * 60)

    # Teste Konfiguration
    test_config()

    print("\n🎉 Configuration test completed successfully!")
    print("\n💡 To change configuration:")
    print("   Edit values directly in config.py")
    print("   All components will automatically use the new values")
