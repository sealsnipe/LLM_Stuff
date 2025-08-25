# 🐧 Linux Setup für RunPod - LLM Training

## 🚀 RunPod Template Empfehlung

**Beste Option:** `Runpod PyTorch 2.4.0` (py3.11-cuda12.4.1-devel-ubuntu22.04)

**Warum diese Wahl:**
- ✅ Ubuntu 22.04 LTS (sehr stabil)
- ✅ CUDA 12.4.1 (aktuell und kompatibel)
- ✅ Python 3.11 (optimal für PyTorch)
- ✅ PyTorch 2.4.0 (unterstützt alle Features)

## 📦 Schnelle Installation

### 1. Pod starten und verbinden
```bash
# SSH in deinen RunPod
# Oder verwende das Web Terminal
```

### 2. Repository klonen
```bash
cd /workspace
git clone https://github.com/sealsnipe/LLM_Stuff.git
cd LLM_Stuff
```

### 3. Dependencies installieren
```bash
# PyTorch sollte bereits installiert sein, aber upgraden:
pip install --upgrade torch torchvision torchaudio

# Projekt-Dependencies
pip install transformers>=4.40.0 tqdm psutil numpy

# Optional für Monitoring
pip install wandb tensorboard
```

### 4. GPU Setup validieren
```bash
python -c "
import torch
print('=== GPU SETUP VALIDATION ===')
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU Count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {props.name}')
        print(f'  Memory: {props.total_memory/1e9:.1f} GB')
        print(f'  Compute: {props.major}.{props.minor}')
    print('✅ GPU Setup erfolgreich!')
else:
    print('❌ CUDA nicht verfügbar')
"
```

### 5. Erstes Training testen
```bash
# Kurzer Test (10 Steps)
python gpu_training_optimized.py
```

## ⚙️ Linux-spezifische Optimierungen

### Triton Cache (Linux ist besser als Windows!)
```bash
# Linux hat bessere Triton-Unterstützung
export TRITON_CACHE_DIR=/tmp/triton_cache
mkdir -p $TRITON_CACHE_DIR
```

### Memory Management
```bash
# Bessere Memory-Performance auf Linux
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
```

### GPU Monitoring
```bash
# GPU Status live anzeigen
watch -n 1 nvidia-smi

# Oder für detaillierte Infos:
nvidia-smi -l 1
```

## 🔧 Konfiguration für Cloud GPUs

### Für RTX 4090 (24GB) - RunPod High-End
```python
# config.py anpassen:
model_config.hidden_size = 2048
model_config.num_layers = 20
training_config.batch_size = 12
training_config.sequence_length = 512
training_config.gradient_accumulation_steps = 4

# Erwartete Performance:
# - ~1.1B Parameter Model
# - ~1.0-1.5 Steps/sec
# - VRAM: 18-22GB
```

### Für RTX 3090 (24GB) - RunPod Standard
```python
# config.py anpassen:
model_config.hidden_size = 1792
model_config.num_layers = 16
training_config.batch_size = 10
training_config.sequence_length = 448

# Erwartete Performance:
# - ~800M Parameter Model
# - ~1.2-1.8 Steps/sec
# - VRAM: 16-20GB
```

### Für A100 (40GB/80GB) - RunPod Premium
```python
# config.py für große Modelle:
model_config.hidden_size = 3072
model_config.num_layers = 32
training_config.batch_size = 16
training_config.sequence_length = 1024
training_config.gradient_accumulation_steps = 2

# Erwartete Performance:
# - ~3-7B Parameter Model
# - ~0.8-1.2 Steps/sec
# - VRAM: 30-70GB
```

## 🎯 3-7B Parameter Training

### Für 3B Parameter Model (A100 40GB)
```python
# config.py für 3B:
model_config.vocab_size = 32000
model_config.hidden_size = 2560
model_config.num_layers = 32
model_config.num_attention_heads = 32
model_config.num_key_value_heads = 8  # GQA 4:1

training_config.batch_size = 8
training_config.gradient_accumulation_steps = 4  # Effective batch: 32
training_config.sequence_length = 2048
training_config.max_steps = 50000
```

### Für 7B Parameter Model (A100 80GB)
```python
# config.py für 7B:
model_config.hidden_size = 4096
model_config.num_layers = 32
model_config.num_attention_heads = 32
model_config.num_key_value_heads = 8

training_config.batch_size = 4
training_config.gradient_accumulation_steps = 8  # Effective batch: 32
training_config.sequence_length = 4096
training_config.max_steps = 100000
```

## 🚀 Performance-Vorteile auf Linux

**Linux vs. Windows Performance:**
- ✅ **20-30% bessere GPU-Performance** (bessere CUDA-Integration)
- ✅ **Stabileres torch.compile** (weniger Triton-Probleme)
- ✅ **Bessere Memory-Management** (kein Windows-Overhead)
- ✅ **Schnellere I/O** für Checkpoints und Logs
- ✅ **Bessere Multi-GPU Unterstützung** (falls verfügbar)

## 📊 Monitoring auf RunPod

### GPU Monitoring Script
```bash
# Erstelle monitoring.sh
cat > monitoring.sh << 'EOF'
#!/bin/bash
while true; do
    clear
    echo "=== GPU STATUS ==="
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
    echo ""
    echo "=== TRAINING PROGRESS ==="
    tail -n 5 training.log 2>/dev/null || echo "No training log yet"
    sleep 2
done
EOF

chmod +x monitoring.sh
./monitoring.sh
```

## 🔄 Automatisches Training Script

```bash
# Erstelle auto_train.sh für kontinuierliches Training
cat > auto_train.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting automated LLM training on RunPod..."

# Aktiviere alle Optimierungen
export CUDA_LAUNCH_BLOCKING=0
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Starte Training mit Logging
python gpu_training_optimized.py 2>&1 | tee training.log

echo "✅ Training completed!"
EOF

chmod +x auto_train.sh
```

## 💾 Checkpoint Management

```bash
# Automatisches Backup der Checkpoints
mkdir -p /workspace/backups
cp checkpoints/*.pt /workspace/backups/ 2>/dev/null || true

# Für persistente Speicherung (RunPod Network Storage)
# Kopiere wichtige Checkpoints in persistent storage
```

## 🎯 Nächste Schritte

1. **RunPod Pod starten** mit PyTorch 2.4.0 Template
2. **Repository klonen** und Dependencies installieren
3. **GPU validieren** mit dem Test-Script
4. **Konfiguration anpassen** für deine gewünschte Model-Größe
5. **Training starten** und Performance monitoren
6. **Checkpoints sichern** für spätere Verwendung

**Erwartete Trainingszeiten für 3-7B Parameter:**
- **3B Model**: ~2-4 Stunden für 10K Steps (A100 40GB)
- **7B Model**: ~6-12 Stunden für 10K Steps (A100 80GB)

Das Projekt ist perfekt für Linux optimiert und wird deutlich besser performen als auf Windows! 🚀
