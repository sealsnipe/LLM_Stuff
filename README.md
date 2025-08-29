# 🚀 Modern LLM Training Framework

**Professional Edition - Clean Architecture**

A production-ready LLM training framework with intelligent caching, dynamic dataset expansion, and robust checkpoint management. Designed for efficient training on consumer GPUs with enterprise-grade features.

## 🎯 Key Features

- **🏗️ Modular Architecture**: Clean separation of concerns with core components
- **📦 Intelligent Caching**: LZ4-compressed sequence packing with dynamic expansion
- **🔄 Seamless Resume**: Robust checkpoint system with automatic cache detection
- **📊 Dynamic Scaling**: Automatic handling of growing datasets during training
- **🎯 Smart Warmup**: Warmup steps calculated for full dataset, not current cache
- **📈 Professional Monitoring**: Real-time metrics, JSON logging, and progress tracking
- **🖥️ Windows Optimized**: Special handling for Triton cache and Windows-specific optimizations

## 📁 Project Structure

```
LLM_Stuff/
├── training-windows.py          # 🎯 Main entry point
├── config.py                    # ⚙️ Central configuration
├── core/                        # 🏗️ Modular architecture
│   ├── models/                  # 🧠 Model components
│   ├── training/                # 🎓 Training infrastructure  
│   ├── data/                    # 📊 Data management
│   ├── checkpoints/             # 💾 Checkpoint system
│   ├── monitoring/              # 📈 Progress tracking
│   ├── utils/                   # 🔧 Utilities
│   └── interfaces/              # 🎛️ High-level APIs
├── cache/                       # 📦 Dataset cache
├── scripts/                     # 🛠️ Utility scripts
│   └── sequence_packing_cache.py # 📦 Cache generation
├── tests/                       # 🧪 Test suite
├── archive/                     # 📚 Legacy code & tests
└── docs/                        # 📚 Documentation
```

## 🛠️ Installation

### Prerequisites

- **Python 3.8+**
- **CUDA 11.8+ or 12.x** (for GPU training)
- **16GB+ RAM** (32GB+ recommended)
- **50GB+ free disk space** (for cache and models)

### Step 1: Clone Repository

```bash
git clone https://github.com/your-username/llm-training-framework.git
cd llm-training-framework/LLM_Stuff
```

### Step 2: Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate
pip install lz4 psutil GPUtil
pip install matplotlib seaborn  # For plotting
```

### Step 3: Windows-Specific Setup (IMPORTANT!)

**Triton Cache Configuration:**
```bash
# Create Triton cache directory (Windows compatibility)
mkdir .triton_cache

# Set environment variables (add to your system or .env file)
set TRITON_CACHE_DIR=%CD%\.triton_cache
set TRITON_INTERPRET=1
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

**For PowerShell:**
```powershell
$env:TRITON_CACHE_DIR = "$PWD\.triton_cache"
$env:TRITON_INTERPRET = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:128"
```

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import lz4; print('LZ4 compression available')"
```

## 🚀 Quick Start

### Basic Training
```bash
python training-windows.py
```

### Cache Generation
```bash
python scripts/sequence_packing_cache.py
```

## 📖 Detailed Usage Guide

### 🎯 Training Windows - The Main Flywheel

The `training-windows.py` script is the central orchestrator that provides a complete training experience:

#### **User Experience Flow:**

1. **🚀 Startup & Environment**
   - Automatic GPU detection and optimization
   - Triton cache setup for Windows
   - System resource validation

2. **📋 Training Mode Selection**
   ```
   🚀 TRAINING MODE SELECTION
   ==================================================
   
   📋 AVAILABLE CHECKPOINTS
   ================================================================================
    1. 415M_FineWeb_512_checkpoint_762_run_1
        Step: 762 | Loss: 5.3716 | Time: 2025-08-29 02:36
   
   N.  Start New Training
   X.  Checkpoint Management (Delete)
   ================================================================================
   Selection [1-1, N or X]: 
   ```

3. **🏷️ Model Naming (New Training)**
   ```
   🏷️  MODEL NAME FOR NEW TRAINING
   ==================================================
   📋 SUGGESTIONS (Choose number or enter custom name):
   --------------------------------------------------
   1. 415M_FineWeb_512 (415M parameters, FineWeb dataset, 512 seq_len)
   2. experiment_415M_FineWeb_512 (Experimental run)
   3. modern_llm_415M (Modern LLM architecture)
   4. test_415M (Test run)
   
   Choose 1-4 or enter custom name: 
   ```

4. **📦 Dataset Loading & Cache Detection**
   - Automatic cache validation and expansion detection
   - Dynamic dataset size calculation
   - Intelligent fallback to FineWeb-Edu if no cache

5. **🎓 Training Execution**
   - Real-time progress monitoring
   - Automatic checkpointing
   - Memory optimization
   - JSON logging for analysis

#### **Input Options & Flows:**

**Checkpoint Resume:**
- Select existing checkpoint → Automatic cache detection → Resume training
- Handles cache expansion automatically (e.g., 352K → 472K sequences)
- Preserves optimizer and scheduler state

**New Training:**
- Choose model name → Dataset selection → Fresh training start
- Automatic run ID assignment
- Clean slate initialization

**Checkpoint Management:**
- View all checkpoints with metadata
- Safe deletion with confirmation
- Orphaned log cleanup

### 📦 Cache Generation Flywheel

The cache generation system has two complementary modes:

#### **🔄 Normal Mode (Forward Processing)**

```bash
python scripts/sequence_packing_cache.py
```

**Process Flow:**
1. **📥 Data Ingestion**
   - Downloads FineWeb-Edu dataset
   - Streams data in chunks to manage memory
   - Tokenizes with HuggingFace tokenizer

2. **📦 Sequence Packing**
   - Packs multiple sequences into 512-token chunks
   - Optimizes token utilization (>98% efficiency)
   - Handles padding and attention masks

3. **💾 Compression & Storage**
   - LZ4 compression for 60-70% size reduction
   - Atomic file operations for safety
   - Metadata tracking for integrity

4. **📊 Progress Monitoring**
   ```
   🚀 SEQUENCE PACKING CACHE GENERATION
   ====================================
   📊 Progress: 1,250/10,000 samples (12.5%)
   💾 Cache: 15 chunks created
   ⚡ Speed: 125 samples/sec
   💿 Compression: 65% size reduction
   ```

#### **🔄 Inverse Mode (Gap Filling)**

```bash
python scripts/sequence_packing_cache.py --mode inverse
```

**Process Flow:**
1. **🔍 Gap Detection**
   - Scans existing cache for missing chunks
   - Identifies sequence ranges to fill
   - Calculates optimal processing order

2. **📥 Targeted Processing**
   - Processes only missing data
   - Maintains chunk numbering consistency
   - Avoids duplicate work

3. **🔗 Seamless Integration**
   - New chunks integrate with existing cache
   - Metadata updates automatically
   - Training can continue without interruption

#### **📈 Cache Growth & Training Integration**

**Dynamic Cache Expansion:**
```
Initial Cache:    352,217 sequences → 19,567 training steps
Expanded Cache:   472,240 sequences → 26,235 training steps
Full Dataset:   2,500,000 sequences → 138,888 training steps (estimated)
```

**Smart Warmup Calculation:**
- **Problem**: Warmup should be consistent across cache sizes
- **Solution**: Calculate warmup based on FULL dataset estimate
- **Implementation**: 
  ```python
  # Estimate full dataset: ~2.5M sequences
  full_dataset_steps = estimate_full_dataset_steps()
  warmup_steps = full_dataset_steps * 0.025  # 2.5%
  
  # Warning if current cache is too small
  if warmup_steps > current_training_steps:
      warn_about_small_cache()
  ```

**Training Adaptation:**
1. **Cache Detection**: Training automatically detects cache expansion
2. **Step Recalculation**: Updates total steps based on new cache size
3. **Warmup Consistency**: Maintains warmup based on full dataset
4. **Resume Compatibility**: Seamlessly resumes with larger cache

### 🎛️ Configuration Options

Edit `config.py` for customization:

```python
# Model Architecture
model_config.hidden_size = 1536        # Model dimension
model_config.num_layers = 24           # Transformer layers
model_config.num_heads = 24            # Attention heads

# Training Parameters
training_config.batch_size = 12        # Per-device batch size
training_config.gradient_accumulation_steps = 4  # Effective batch = 48
training_config.learning_rate = 3e-4   # Peak learning rate
training_config.target_epochs = 5      # Training epochs

# Dataset Configuration
training_config.use_packed_cache = True     # Use compressed cache
training_config.epoch_dataset_fraction = 0.8  # Use 80% per epoch
training_config.sequence_length = 512       # Token sequence length

# System Optimization
training_config.use_mixed_precision = True  # FP16 training
training_config.use_torch_compile = True    # Torch compilation
training_config.use_flash_attention = True  # Flash attention
```

### 📊 Monitoring & Logging

**Real-time Display:**
```
Step   770/26,235 ( 2.9%) | Loss: 5.583 | 106,212 tok/s | ETA: 04:23h
```

**JSON Logs:** `training_logs/model_name_run_X.json`
- Structured data for analysis
- Step-by-step metrics
- Resumable training state

**Checkpoints:** `current_training/checkpoints/`
- Model state preservation
- Optimizer state included
- Scheduler state maintained

## 🔧 Advanced Usage

### Custom Dataset Integration

```python
# In core/data/dataset_factory.py
def create_custom_dataset(self, data_path):
    # Implement your dataset loading logic
    pass
```

### Model Architecture Modification

```python
# In core/models/transformer.py
class CustomTransformer(MemoryOptimizedTransformer):
    # Extend base architecture
    pass
```

### Training Loop Customization

```python
# In core/training/trainer.py
def custom_training_step(self, batch):
    # Implement custom training logic
    pass
```

## 🐛 Troubleshooting

### Common Issues

**CUDA Out of Memory:**
```bash
# Reduce batch size in config.py
training_config.batch_size = 8
training_config.gradient_accumulation_steps = 6
```

**Triton Compilation Errors (Windows):**
```bash
# Set fallback mode
set TRITON_INTERPRET=1
```

**Cache Corruption:**
```bash
# Regenerate cache
python scripts/sequence_packing_cache.py --force-rebuild
```

### Performance Optimization

**For RTX 4090:**
- Batch size: 12-16
- Mixed precision: Enabled
- Flash attention: Enabled

**For RTX 3080:**
- Batch size: 8-10
- Gradient accumulation: 6-8
- Memory optimization: Enabled

## 📈 Performance Benchmarks

| GPU | Batch Size | Tokens/sec | Memory Usage |
|-----|------------|------------|--------------|
| RTX 4090 | 12 | 106,000 | 22GB |
| RTX 3080 | 8 | 78,000 | 9.5GB |
| RTX 3070 | 6 | 65,000 | 7.2GB |

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- HuggingFace for transformers and datasets
- PyTorch team for the framework
- OpenAI for inspiration and research
- Community contributors and testers
