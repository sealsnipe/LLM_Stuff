# 🛠️ Installation Guide

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, Linux (Ubuntu 18.04+), macOS 10.15+
- **Python**: 3.8 or higher
- **RAM**: 16GB (32GB recommended for large models)
- **Storage**: 50GB free space (100GB+ for full dataset)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3070 or better)

### Recommended Setup
- **GPU**: RTX 4090 (24GB VRAM) or RTX 3080/4080 (12-16GB VRAM)
- **RAM**: 32GB+ DDR4/DDR5
- **Storage**: NVMe SSD with 200GB+ free space
- **CPU**: 8+ cores (Intel i7/i9 or AMD Ryzen 7/9)

## Step-by-Step Installation

### 1. Environment Setup

#### Windows (PowerShell as Administrator)
```powershell
# Check Python version
python --version  # Should be 3.8+

# Create virtual environment (recommended)
python -m venv llm_env
.\llm_env\Scripts\Activate.ps1

# If execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Linux/macOS
```bash
# Check Python version
python3 --version  # Should be 3.8+

# Create virtual environment
python3 -m venv llm_env
source llm_env/bin/activate
```

### 2. CUDA Installation (GPU Training)

#### Check CUDA Compatibility
```bash
nvidia-smi  # Check driver version
```

#### Install CUDA Toolkit (if needed)
- **CUDA 11.8**: For older GPUs (RTX 20/30 series)
- **CUDA 12.x**: For newer GPUs (RTX 40 series)

Download from: https://developer.nvidia.com/cuda-downloads

### 3. PyTorch Installation

#### For CUDA 11.8
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### For CUDA 12.1
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### CPU Only (not recommended for training)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 4. Core Dependencies

```bash
# Essential packages
pip install transformers>=4.30.0
pip install datasets>=2.12.0
pip install accelerate>=0.20.0

# Compression and utilities
pip install lz4>=4.3.0
pip install psutil>=5.9.0
pip install GPUtil>=1.4.0

# Visualization and analysis
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0
pip install pandas>=2.0.0

# Optional: Advanced optimizations
pip install flash-attn>=2.0.0  # May require compilation
pip install triton>=2.0.0      # For advanced kernels
```

### 5. Windows-Specific Configuration

#### Triton Cache Setup (CRITICAL for Windows)
```powershell
# Create cache directory
mkdir .triton_cache

# Set environment variables permanently
[Environment]::SetEnvironmentVariable("TRITON_CACHE_DIR", "$PWD\.triton_cache", "User")
[Environment]::SetEnvironmentVariable("TRITON_INTERPRET", "1", "User")
[Environment]::SetEnvironmentVariable("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128", "User")

# For current session
$env:TRITON_CACHE_DIR = "$PWD\.triton_cache"
$env:TRITON_INTERPRET = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:128"
```

#### Alternative: Create .env file
```bash
# Create .env file in project root
echo TRITON_CACHE_DIR=./.triton_cache > .env
echo TRITON_INTERPRET=1 >> .env
echo PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 >> .env
```

### 6. Verification

#### Test GPU Setup
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
```

#### Test Dependencies
```python
# Test core imports
import transformers
import datasets
import lz4
import psutil
import GPUtil

print("✅ All dependencies installed successfully!")
```

#### Test Framework
```bash
# Quick system check
python -c "from core.utils.gpu_utils import check_gpu_setup; check_gpu_setup()"
```

## Troubleshooting

### Common Installation Issues

#### 1. CUDA Version Mismatch
```bash
# Check CUDA version
nvcc --version
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 2. Triton Compilation Errors (Windows)
```bash
# Enable interpretation mode
set TRITON_INTERPRET=1

# Or disable torch.compile
# In config.py: training_config.use_torch_compile = False
```

#### 3. Out of Memory Errors
```python
# In config.py, reduce batch size
training_config.batch_size = 8  # Reduce from 12
training_config.gradient_accumulation_steps = 6  # Increase to maintain effective batch size
```

#### 4. Flash Attention Installation Issues
```bash
# If flash-attn fails to install
pip install flash-attn --no-build-isolation

# Or disable flash attention
# In config.py: training_config.use_flash_attention = False
```

#### 5. Permission Errors (Windows)
```powershell
# Run PowerShell as Administrator
# Or change execution policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Performance Optimization

#### For Different GPU Configurations

**RTX 4090 (24GB VRAM):**
```python
training_config.batch_size = 16
training_config.gradient_accumulation_steps = 3
training_config.use_mixed_precision = True
training_config.use_flash_attention = True
```

**RTX 3080/4080 (10-16GB VRAM):**
```python
training_config.batch_size = 12
training_config.gradient_accumulation_steps = 4
training_config.use_mixed_precision = True
training_config.use_flash_attention = True
```

**RTX 3070 (8GB VRAM):**
```python
training_config.batch_size = 8
training_config.gradient_accumulation_steps = 6
training_config.use_mixed_precision = True
training_config.use_flash_attention = False  # May cause OOM
```

#### Memory Optimization
```python
# Enable gradient checkpointing
training_config.gradient_checkpointing = True

# Reduce sequence length if needed
training_config.sequence_length = 256  # Instead of 512

# Enable CPU offloading for large models
training_config.cpu_offload = True
```

## Next Steps

After successful installation:

1. **Test the framework**: `python training-windows.py`
2. **Generate cache**: `python scripts/sequence_packing_cache.py`
3. **Read the usage guide**: See README.md for detailed usage
4. **Configure for your setup**: Edit `config.py` as needed

## Getting Help

If you encounter issues:

1. Check this troubleshooting section
2. Search existing GitHub issues
3. Create a new issue with:
   - Your system specifications
   - Python and CUDA versions
   - Complete error message
   - Steps to reproduce

## Hardware Recommendations

### Budget Setup ($800-1200)
- **GPU**: RTX 3070 (8GB) or RTX 4060 Ti (16GB)
- **RAM**: 32GB DDR4
- **Storage**: 1TB NVMe SSD

### Enthusiast Setup ($1500-2500)
- **GPU**: RTX 4080 (16GB) or RTX 3080 (12GB)
- **RAM**: 32GB DDR4/DDR5
- **Storage**: 2TB NVMe SSD

### Professional Setup ($3000+)
- **GPU**: RTX 4090 (24GB) or multiple GPUs
- **RAM**: 64GB+ DDR5
- **Storage**: 4TB+ NVMe SSD
- **CPU**: Intel i9 or AMD Ryzen 9
