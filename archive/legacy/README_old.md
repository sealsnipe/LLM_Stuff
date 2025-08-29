# 🚀 LLM Training System

Professional GPU-optimized LLM training system with advanced features.

## 📁 Project Structure

```
LLM_Stuff/
├── 🎯 CORE TRAINING FILES
│   ├── training-windows.py          # Main training script
│   ├── config.py                    # Central configuration
│   ├── professional_logger.py       # Professional logging system
│   ├── debug_logger.py             # Debug & error logging
│   ├── training_logger.py          # Training metrics logging
│   ├── fast_dataset_loader.py      # Optimized dataset loading
│   ├── sequence_packing_cache.py   # Sequence packing & caching
│   ├── optimized_sequence_packing.py # Sequence packing algorithms
│   └── muon_optimizer.py           # Advanced optimizers
│
├── 📂 ORGANIZED DIRECTORIES
│   ├── tests/                      # All test files
│   ├── scripts/                    # Utility scripts & .bat files
│   ├── reports/                    # Performance reports & documentation
│   ├── archive/                    # Old/deprecated files
│   ├── docs/                       # Technical documentation
│   ├── cache/                      # Dataset & sequence caches
│   ├── training_logs/              # Training run logs
│   └── logs/                       # Analysis plots & charts
```

## 🚀 Quick Start

### 1. Start Training
```bash
python training-windows.py
```

### 2. Configuration
Edit `config.py` for:
- Model architecture (372M parameters)
- Training parameters (batch size, learning rate)
- Dataset settings (FineWeb-Edu)
- Hardware optimizations

### 3. Debug Mode
```python
# In config.py:
debug_mode: bool = True  # Enable detailed debug logs
```

## ⚡ Key Features

### 🎯 **Training System**
- **GPU-Optimized**: RTX 3090/4090 optimized
- **Mixed Precision**: BFLOAT16 training
- **Flash Attention**: Memory-efficient attention
- **Torch Compile**: JIT compilation for speed
- **Sequence Packing**: 99% token utilization

### 📊 **Monitoring**
- **Professional Logs**: Clean console output
- **Debug System**: Detailed debug files
- **Training Metrics**: Real-time performance tracking
- **Progress Display**: ETA, tokens/sec, loss tracking

### 💾 **Data Pipeline**
- **Packed Cache**: Instant loading (no tokenization)
- **Intelligent Training**: Auto-adjusts steps based on data
- **FineWeb-Edu**: High-quality training data
- **Resume Support**: Crash-resistant training

### 🔧 **Advanced Features**
- **Checkpoint System**: Auto-save & resume
- **Memory Optimization**: Gradient checkpointing
- **Multi-Worker**: Parallel data loading
- **Error Handling**: Robust error recovery

## 📈 Performance

**Typical Performance (RTX 3090):**
- **Speed**: ~9,000-12,000 tokens/sec
- **Memory**: ~20-22GB VRAM usage
- **Efficiency**: 99% token utilization
- **Stability**: <1% gradient clipping

## 🛠️ Directory Details

### `/tests/`
- Unit tests for all components
- Integration tests
- Performance benchmarks
- Interactive chat tests

### `/scripts/`
- Dataset download scripts
- Cache creation utilities
- Analysis tools
- Batch files for automation

### `/reports/`
- Performance analysis reports
- Optimization documentation
- Sequence packing studies

### `/archive/`
- Deprecated code
- Old implementations
- Legacy training scripts

## 🔍 Troubleshooting

### Common Issues
1. **GPU Memory**: Reduce batch_size in config.py
2. **Slow Training**: Enable debug_mode to check bottlenecks
3. **Cache Issues**: Delete cache/ and regenerate
4. **Import Errors**: Ensure all core files are in root directory

### Debug Files
- `debug_run_XXX.log` - Detailed debug information
- `error_run_XXX.log` - Error logs and stack traces
- `training_logs/` - Training metrics and plots

## 📝 Configuration

Key settings in `config.py`:

```python
# Model Architecture
hidden_size: int = 1536          # Model dimension
num_layers: int = 28             # Transformer layers
num_attention_heads: int = 12    # Attention heads

# Training Parameters
batch_size: int = 12             # GPU-optimized batch size
gradient_accumulation_steps: int = 6  # Effective batch: 72
sequence_length: int = 768       # Context length
learning_rate: float = 2.55e-4   # Optimized learning rate

# Performance
use_torch_compile: bool = True   # JIT compilation
use_mixed_precision: bool = True # BFLOAT16
use_flash_attention: bool = True # Memory efficiency
debug_mode: bool = False         # Debug logging
```

## 🎯 Next Steps

1. **Start Training**: Run `python training-windows.py`
2. **Monitor Progress**: Watch console output
3. **Check Logs**: Review `training_logs/` for metrics
4. **Optimize**: Adjust config.py based on performance
5. **Scale Up**: Increase batch_size if GPU allows

---

**Built for professional LLM training with maximum efficiency and reliability.** 🚀
