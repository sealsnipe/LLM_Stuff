#!/usr/bin/env python3
"""
🎯 SIMPLE TRAINING CONFIGURATION
Einfache, token-basierte Trainingskonfiguration ohne komplizierte Presets
"""

from config import training_config, model_config, dataset_config

def set_training_target(target_tokens_billions: float):
    """
    Setzt das Training-Ziel in Milliarden Tokens.
    
    Args:
        target_tokens_billions: Ziel in Milliarden Tokens (z.B. 18.5 für 18.5B)
    
    Examples:
        set_training_target(0.1)    # 100M tokens (quick test)
        set_training_target(1.0)    # 1B tokens (small run)  
        set_training_target(5.0)    # 5B tokens (medium run)
        set_training_target(18.5)   # 18.5B tokens (minimum viable)
        set_training_target(46.0)   # 46B tokens (optimal)
    """
    
    target_tokens = int(target_tokens_billions * 1_000_000_000)
    
    # Update training config
    training_config.target_tokens = target_tokens
    
    # Calculate derived values
    effective_batch_size = training_config.batch_size * training_config.gradient_accumulation_steps
    tokens_per_step = effective_batch_size * training_config.sequence_length
    max_steps = target_tokens // tokens_per_step
    
    # Calculate samples needed
    samples_needed = dataset_config.get_samples_for_tokens(target_tokens)
    
    # Calculate estimated time
    steps_per_hour = 200  # Conservative estimate for consumer GPU
    estimated_hours = max_steps / steps_per_hour
    
    print("🎯 TRAINING TARGET SET")
    print("=" * 50)
    print(f"Target Tokens:     {target_tokens:,} ({target_tokens_billions:.1f}B)")
    print(f"Training Steps:    {max_steps:,}")
    print(f"Samples Needed:    {samples_needed:,}")
    print(f"Estimated Time:    {estimated_hours:.1f} hours ({estimated_hours/24:.1f} days)")
    print(f"Tokens per Step:   {tokens_per_step:,}")
    print(f"Effective Batch:   {effective_batch_size}")
    print("=" * 50)
    
    return {
        'target_tokens': target_tokens,
        'max_steps': max_steps,
        'samples_needed': samples_needed,
        'estimated_hours': estimated_hours
    }

def get_recommended_targets():
    """Zeigt empfohlene Training-Targets für verschiedene Zwecke"""
    
    print("🎯 RECOMMENDED TRAINING TARGETS")
    print("=" * 60)
    print("For 926M parameter model:")
    print()
    
    targets = [
        (0.1, "Quick Test", "2 hours", "Verify everything works"),
        (1.0, "Development", "20 hours", "Basic language patterns"),
        (5.0, "Medium Run", "4 days", "Decent coherence"),
        (18.5, "Minimum Viable", "2 weeks", "Proper language model"),
        (46.0, "Optimal", "1 month", "GPT-2 Large quality")
    ]
    
    for tokens_b, name, time, description in targets:
        print(f"{tokens_b:4.1f}B tokens - {name:15} ({time:8}) - {description}")
    
    print()
    print("💡 RECOMMENDATION:")
    print("   Start with 1.0B tokens to verify training works")
    print("   Then scale to 18.5B+ for production model")
    print("=" * 60)

def update_model_architecture(hidden_size: int = 1536, num_layers: int = 24):
    """
    Updates model architecture for better performance.
    
    Args:
        hidden_size: Hidden dimension (default: 1536, slightly wider)
        num_layers: Number of layers (default: 24, deeper)
    """
    
    # Calculate other dimensions
    num_heads = hidden_size // 64  # 64 dims per head
    if num_heads % 2 != 0:
        num_heads -= 1  # Make even for GQA
    
    kv_heads = num_heads // 2  # GQA: half KV heads
    intermediate_size = int(hidden_size * 8/3)  # SwiGLU expansion
    
    # Update config
    model_config.hidden_size = hidden_size
    model_config.num_layers = num_layers
    model_config.num_attention_heads = num_heads
    model_config.num_key_value_heads = kv_heads
    model_config.intermediate_size = intermediate_size
    
    # Calculate parameters
    params = calculate_model_parameters()
    
    print("🏗️  MODEL ARCHITECTURE UPDATED")
    print("=" * 40)
    print(f"Hidden Size:       {hidden_size}")
    print(f"Layers:            {num_layers}")
    print(f"Attention Heads:   {num_heads}")
    print(f"KV Heads:          {kv_heads}")
    print(f"Intermediate:      {intermediate_size}")
    print(f"Total Parameters:  {params:,} ({params/1e6:.0f}M)")
    print("=" * 40)
    
    return params

def calculate_model_parameters():
    """Calculate total model parameters"""
    
    h = model_config.hidden_size
    n = model_config.num_layers
    v = model_config.vocab_size
    i = model_config.intermediate_size
    
    # Embedding parameters
    embedding_params = v * h
    
    # Transformer block parameters (per layer)
    # Attention: q_proj, k_proj, v_proj, o_proj
    attention_params = 4 * h * h
    
    # Feed forward: gate_proj, up_proj, down_proj
    ff_params = 3 * h * i
    
    # Layer norms (RMSNorm has no bias)
    norm_params = 2 * h  # attention_norm + ffn_norm
    
    # Per layer total
    layer_params = attention_params + ff_params + norm_params
    
    # Total model
    total_params = embedding_params + (n * layer_params) + h  # +h for final norm
    
    return total_params

def show_current_config():
    """Shows current training configuration"""
    
    print("📋 CURRENT CONFIGURATION")
    print("=" * 50)
    print("MODEL:")
    print(f"  Parameters:      {calculate_model_parameters():,}")
    print(f"  Hidden Size:     {model_config.hidden_size}")
    print(f"  Layers:          {model_config.num_layers}")
    print(f"  Attention Heads: {model_config.num_attention_heads}")
    print(f"  Vocab Size:      {model_config.vocab_size}")
    print()
    print("TRAINING:")
    print(f"  Target Tokens:   {training_config.target_tokens:,}")
    print(f"  Max Steps:       {training_config.max_steps:,}")
    print(f"  Batch Size:      {training_config.batch_size}")
    print(f"  Grad Accum:      {training_config.gradient_accumulation_steps}")
    print(f"  Learning Rate:   {training_config.learning_rate}")
    print(f"  Sequence Length: {training_config.sequence_length}")
    print("=" * 50)

if __name__ == "__main__":
    print("🎯 Simple Training Configuration Tool")
    print()
    
    # Show recommendations
    get_recommended_targets()
    print()
    
    # Show current config
    show_current_config()
    print()
    
    print("💡 Usage Examples:")
    print("   from simple_training_config import set_training_target")
    print("   set_training_target(1.0)    # 1B tokens")
    print("   set_training_target(18.5)   # 18.5B tokens")
