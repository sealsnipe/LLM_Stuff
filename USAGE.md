# 📖 Usage Guide

## Overview

This guide covers the complete workflow from cache generation to model training, including advanced features and optimization strategies.

## 🔄 The Complete Workflow

### Phase 1: Cache Generation

The cache generation system processes raw datasets into optimized, compressed chunks for efficient training.

#### Normal Mode (Forward Processing)
```bash
cd scripts
python sequence_packing_cache.py
```

**What happens:**
1. **Dataset Download**: Automatically downloads FineWeb-Edu dataset
2. **Tokenization**: Converts text to tokens using HuggingFace tokenizer
3. **Sequence Packing**: Combines multiple sequences into 512-token chunks
4. **Compression**: Applies LZ4 compression (60-70% size reduction)
5. **Metadata Generation**: Creates registry and validation data

**Output Structure:**
```
cache/packed_sequences/512/FineWeb/
├── packed_chunk_000000.pt    # Compressed data chunks
├── packed_chunk_000001.pt
├── ...
├── cache_metadata.json       # Dataset metadata
└── cache_registry.json       # Global registry
```

#### Inverse Mode (Gap Filling)
```bash
python sequence_packing_cache.py --mode inverse
```

**Use Cases:**
- Fill gaps in existing cache
- Resume interrupted cache generation
- Add new data to existing cache

**Process:**
1. **Gap Detection**: Scans for missing chunk numbers
2. **Targeted Processing**: Only processes missing sequences
3. **Seamless Integration**: New chunks integrate with existing cache

### Phase 2: Training

#### Starting Training
```bash
python training-windows.py
```

**Interactive Flow:**

1. **Environment Setup**
   ```
   🚀 MODERN LLM TRAINING FRAMEWORK
   ==================================================
   Initializing system environment                [COMPLETE]
   Configuring GPU settings                       [COMPLETE]
   ```

2. **Training Mode Selection**
   ```
   📋 AVAILABLE CHECKPOINTS
   ================================================================================
    1. 415M_FineWeb_512_checkpoint_762_run_1
        Step: 762 | Loss: 5.3716 | Time: 2025-08-29 02:36
   
   N.  Start New Training
   X.  Checkpoint Management (Delete)
   ================================================================================
   ```

3. **Model Configuration** (New Training)
   ```
   🏷️  MODEL NAME FOR NEW TRAINING
   ==================================================
   📋 SUGGESTIONS:
   1. 415M_FineWeb_512 (415M parameters, FineWeb dataset, 512 seq_len)
   2. experiment_415M_FineWeb_512 (Experimental run)
   3. modern_llm_415M (Modern LLM architecture)
   4. test_415M (Test run)
   ```

4. **Dataset Loading & Validation**
   ```
   📦 Cache aus Checkpoint: FineWeb (seq_len: 512)
   🔍 Validating cache: FineWeb at cache/packed_sequences/512/FineWeb
   ✅ Cache validated: 59 chunks
   📈 Dataset expanded: 352,217 → 472,240 sequences
   ```

5. **Training Execution**
   ```
   INFO: Dataset: Packed Cache
   INFO: Samples: 472,240 | Batches: 39,353 | Seq Length: 512
   INFO: Training Mode: EPOCH-BASED (5 epochs)
   INFO: Total steps: 26,235
   INFO: Dynamic warmup steps: 649 (2.5%)
   
   Training started - Target steps: 26,235
   Step   770/26,235 ( 2.9%) | Loss: 5.583 | 106,212 tok/s | ETA: 04:23h
   ```

## 🎯 Advanced Features

### Dynamic Cache Expansion

The system automatically handles growing datasets:

**Scenario**: Cache grows from 352K to 472K sequences during training

**Automatic Handling:**
1. **Detection**: Training detects cache expansion on resume
2. **Recalculation**: Updates total steps (19,567 → 26,235)
3. **Seamless Resume**: Continues from last checkpoint
4. **Consistent Warmup**: Maintains warmup based on full dataset estimate

**Example Output:**
```
📈 Dataset expanded: 352,217 → 472,240 sequences
🔧 Trainer: Cache expanded 352,217 → 472,240 sequences
INFO: Using dynamic total steps: 26,235 (with current dataset)
```

### Smart Warmup Strategy

**Problem**: Warmup should be consistent regardless of current cache size

**Solution**: Calculate warmup based on estimated full dataset size

**Implementation:**
```python
# Estimate full FineWeb dataset: ~2.5M sequences
full_dataset_steps = estimate_full_dataset_steps()  # ~138,888 steps
warmup_steps = full_dataset_steps * 0.025           # ~3,472 steps (2.5%)

# Current cache might only support 26,235 steps
if warmup_steps > current_training_steps:
    warn_about_partial_cache()
```

**Benefits:**
- Consistent warmup across different cache sizes
- Prepares model for eventual full dataset training
- Prevents warmup inconsistencies during cache expansion

### Checkpoint Management

**Automatic Checkpointing:**
- Saves every 100 steps (configurable)
- Includes model, optimizer, and scheduler state
- Preserves cache metadata for resume compatibility

**Checkpoint Structure:**
```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'step': current_step,
    'loss': current_loss,
    'cache_info': {
        'dataset_name': 'FineWeb',
        'total_sequences': 472240,
        'sequence_length': 512
    }
}
```

**Resume Process:**
1. **Cache Validation**: Ensures cache still exists and is valid
2. **Expansion Detection**: Compares checkpoint cache size with current
3. **State Restoration**: Loads model, optimizer, and scheduler states
4. **Step Recalculation**: Updates training plan if cache expanded

## 🔧 Configuration

### Model Architecture (`config.py`)

```python
class ModelConfig:
    # Core architecture
    hidden_size = 1536              # Model dimension
    num_layers = 24                 # Transformer layers
    num_heads = 24                  # Attention heads
    intermediate_size = 6144        # FFN dimension
    
    # Attention configuration
    max_position_embeddings = 2048  # Maximum sequence length
    attention_dropout = 0.1         # Attention dropout rate
    hidden_dropout = 0.1            # Hidden layer dropout
    
    # Vocabulary
    vocab_size = 50257              # GPT-2 vocabulary size
```

### Training Configuration

```python
class TrainingConfig:
    # Batch configuration
    batch_size = 12                 # Per-device batch size
    gradient_accumulation_steps = 4 # Effective batch = 48
    
    # Learning rate schedule
    learning_rate = 3e-4           # Peak learning rate
    warmup_ratio = 0.025           # 2.5% warmup
    weight_decay = 0.1             # AdamW weight decay
    
    # Training duration
    use_epoch_based_training = True # Use epochs instead of tokens
    target_epochs = 5              # Number of epochs
    epoch_dataset_fraction = 0.8   # Use 80% of data per epoch
    
    # Optimization
    use_mixed_precision = True     # FP16 training
    use_torch_compile = True       # Torch compilation
    use_flash_attention = True     # Flash attention
    gradient_checkpointing = False # Memory vs speed tradeoff
    
    # Monitoring
    log_interval = 10              # Log every N steps
    save_interval = 100            # Save checkpoint every N steps
    eval_interval = 500            # Evaluation every N steps
```

### Dataset Configuration

```python
class DatasetConfig:
    # Cache settings
    use_packed_cache = True        # Use compressed cache
    packed_cache_validation = True # Validate cache integrity
    
    # Sequence processing
    sequence_length = 512          # Token sequence length
    pack_sequences = True          # Enable sequence packing
    
    # Dataset selection
    dataset_name = "FineWeb"       # Primary dataset
    num_samples = "auto"           # Auto-detect or specify number
```

## 📊 Monitoring & Analysis

### Real-time Monitoring

**Progress Display:**
```
Step   770/26,235 ( 2.9%) | Loss: 5.583 | 106,212 tok/s | ETA: 04:23h
```

**Components:**
- **Step Progress**: Current step / Total steps (Percentage)
- **Loss**: Current training loss
- **Throughput**: Tokens processed per second
- **ETA**: Estimated time to completion

### JSON Logging

**Log Structure** (`training_logs/model_name_run_X.json`):
```json
{
  "metadata": {
    "model_name": "415M_FineWeb_512",
    "start_time": "2025-08-29T02:20:27",
    "total_steps": 26235,
    "target_tokens": 967147520
  },
  "training_data": [
    {
      "step": 770,
      "timestamp": "2025-08-29T02:36:15",
      "loss": 5.583,
      "lr": 0.0003,
      "tokens_per_sec": 106212,
      "gpu_memory_gb": 22.1,
      "progress_pct": 2.9
    }
  ]
}
```

### Performance Analysis

**Generate Training Plots:**
```python
from core.monitoring.training_plotter import plot_training_progress
plot_training_progress("training_logs/415M_FineWeb_512_run_1.json")
```

**Metrics Tracked:**
- Training loss over time
- Learning rate schedule
- Throughput (tokens/second)
- GPU memory usage
- Training progress

## 🚀 Optimization Strategies

### Memory Optimization

**For Limited VRAM:**
```python
# Reduce batch size, increase accumulation
training_config.batch_size = 8
training_config.gradient_accumulation_steps = 6

# Enable gradient checkpointing
training_config.gradient_checkpointing = True

# Reduce sequence length
training_config.sequence_length = 256
```

**For Large VRAM:**
```python
# Increase batch size
training_config.batch_size = 16
training_config.gradient_accumulation_steps = 3

# Disable gradient checkpointing
training_config.gradient_checkpointing = False
```

### Speed Optimization

**Enable All Optimizations:**
```python
training_config.use_mixed_precision = True
training_config.use_torch_compile = True
training_config.use_flash_attention = True
```

**Disable for Compatibility:**
```python
# If encountering compilation errors
training_config.use_torch_compile = False
training_config.use_flash_attention = False
```

### Dataset Optimization

**Cache Generation Strategy:**
1. **Start Small**: Generate initial cache with subset of data
2. **Begin Training**: Start training while cache generation continues
3. **Expand Gradually**: Add more chunks using inverse mode
4. **Resume Seamlessly**: Training adapts to larger cache automatically

**Benefits:**
- Faster time to first training
- Continuous cache expansion
- No training interruption
- Optimal resource utilization

## 🔍 Troubleshooting

### Common Issues

**1. Cache Corruption**
```bash
# Regenerate cache
python scripts/sequence_packing_cache.py --force-rebuild
```

**2. Memory Issues**
```python
# Reduce memory usage
training_config.batch_size = 6
training_config.gradient_accumulation_steps = 8
```

**3. Slow Training**
```python
# Check optimizations
training_config.use_mixed_precision = True
training_config.use_torch_compile = True
```

**4. Checkpoint Issues**
```bash
# Clean restart
rm -rf current_training/checkpoints/*
python training-windows.py
```

### Performance Tuning

**Monitor GPU Utilization:**
```bash
nvidia-smi -l 1  # Monitor every second
```

**Optimal Settings by GPU:**

| GPU | Batch Size | Grad Accum | Mixed Precision | Flash Attention |
|-----|------------|------------|-----------------|-----------------|
| RTX 4090 | 16 | 3 | ✅ | ✅ |
| RTX 4080 | 12 | 4 | ✅ | ✅ |
| RTX 3080 | 10 | 5 | ✅ | ✅ |
| RTX 3070 | 8 | 6 | ✅ | ❌ |

## 📈 Best Practices

### Training Strategy
1. **Start with small cache** for quick iteration
2. **Monitor loss curves** for training stability
3. **Use checkpoints frequently** for safety
4. **Expand cache gradually** during training
5. **Validate on held-out data** periodically

### Resource Management
1. **Monitor GPU memory** usage
2. **Use mixed precision** for speed
3. **Enable optimizations** gradually
4. **Clean up old checkpoints** regularly
5. **Backup important models** externally

### Development Workflow
1. **Test with small dataset** first
2. **Validate configuration** before long runs
3. **Monitor initial steps** for issues
4. **Scale up gradually** after validation
5. **Document successful configurations**
