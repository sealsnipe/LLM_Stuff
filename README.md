# 🚀 Modern LLM Training System

Ein vollständiges, GPU-optimiertes System zum Training und Inference von Large Language Models mit modernsten Techniken.

## 📋 Überblick

Dieses System implementiert ein komplettes LLM-Training-Pipeline mit:
- **GPU-optimiertes Training** mit torch.compile und Mixed Precision
- **Moderne Transformer-Architektur** mit GQA, RoPE, SwiGLU
- **State-of-the-Art Optimizers** (AdamW Fused + Muon)
- **Production-ready Inference** mit verschiedenen Sampling-Strategien

## 🏗️ System-Architektur

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM Training System                      │
├─────────────────────────────────────────────────────────────┤
│  gpu_training_optimized.py  │  Training Loop & Optimierung  │
│  modern_llm.py             │  Model-Architektur            │
│  muon_optimizer.py         │  Advanced Optimizer           │
│  text_generator.py         │  Inference & Generation       │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Dateien-Übersicht

### 🔥 `gpu_training_optimized.py` - Haupttraining
**Zweck:** GPU-optimierte Training-Loop mit modernsten Performance-Techniken

**Key Features:**
- **torch.compile** mit `max-autotune` Mode für 20-40% Speedup
- **Mixed Precision (BF16)** für Memory-Effizienz
- **Fused AdamW** für 10-20% Optimizer-Speedup
- **Flash Attention 2** automatisch via PyTorch 2.5+
- **Gradient Accumulation** für effektive große Batch Sizes
- **Memory Monitoring** mit detailliertem GPU/CPU Tracking

**Konfiguration:**
```python
@dataclass
class GPUTrainingConfig:
    # Model parameters (1B Parameter für RTX 4070 Ti)
    vocab_size: int = 32000
    hidden_size: int = 1536
    num_layers: int = 12           # Angepasst für 12GB VRAM
    num_attention_heads: int = 24
    num_key_value_heads: int = 6   # GQA 4:1 ratio
    
    # Training parameters
    max_steps: int = 2000
    batch_size: int = 5            # Optimiert für VRAM
    gradient_accumulation_steps: int = 8
    sequence_length: int = 384     # Balance Performance/Memory
    
    # Performance optimizations
    use_torch_compile: bool = True
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = False  # AUS für Performance
```

### 🏛️ `modern_llm.py` - Model-Architektur
**Zweck:** State-of-the-Art Transformer-Architektur mit allen modernen Optimierungen

**Architektur-Features:**
- **Grouped-Query Attention (GQA)** - 4:1 Q:KV Ratio für Memory-Effizienz
- **Rotary Position Embeddings (RoPE)** - Bessere Positionscodierung
- **SwiGLU Activation** - Bessere Performance als ReLU/GELU
- **Pre-Norm Architecture** - Stabileres Training
- **QK Normalization** - Verhindert Attention-Kollaps
- **Weight Tying** - Shared Embeddings für Effizienz

**Model-Komponenten:**
```python
class ModernLLM(nn.Module):
    ├── TokenEmbedding          # Vocab → Hidden
    ├── TransformerBlocks       # N × Transformer Layer
    │   ├── GroupedQueryAttention  # GQA mit RoPE
    │   ├── SwiGLU_MLP            # Feed Forward
    │   └── LayerNorms            # Pre-Norm
    ├── OutputNorm              # Final LayerNorm
    └── LMHead                  # Hidden → Vocab (tied weights)
```

### ⚡ `muon_optimizer.py` - Advanced Optimizer
**Zweck:** State-of-the-Art Optimizer mit Newton-Schulz Orthogonalization

**Muon vs. AdamW:**
- **Bessere Konvergenz** für 2D Weight Matrices (Linear Layers)
- **Newton-Schulz Orthogonalization** für stabilere Updates
- **Momentum-basiert** mit Nesterov Acceleration
- **Speziell für Transformer** optimiert

**Verwendung:**
```python
# Für Linear Layers (bessere Performance)
muon_params = [p for n, p in model.named_parameters() if len(p.shape) == 2]
adamw_params = [p for n, p in model.named_parameters() if len(p.shape) != 2]

optimizer = Muon(muon_params, lr=0.02)
optimizer_adamw = AdamW(adamw_params, lr=1e-4)
```

### 🎯 `text_generator.py` - Inference System
**Zweck:** Production-ready Text Generation mit GPU-Optimierung

**Generation-Features:**
- **Multiple Sampling-Strategien:**
  - Greedy Decoding (deterministisch)
  - Temperature Sampling (kontrollierte Randomness)
  - Top-p (Nucleus) Sampling (qualitativ hochwertig)
  - Top-k Sampling (begrenzte Auswahl)
- **Batched Generation** für mehrere Sequenzen parallel
- **Interactive CLI** mit Echtzeit-Generation
- **Automatic Device Detection** (Multi-GPU Support)
- **torch.compile** für Inference-Optimierung

**Sampling-Parameter:**
```python
generator.generate(
    prompt="The future of AI is",
    max_length=100,
    temperature=0.8,      # Höher = kreativer
    top_p=0.9,           # Nucleus sampling
    top_k=50,            # Top-k filtering
    do_sample=True,      # Sampling vs. Greedy
    num_return_sequences=1
)
```

## 🛠️ Installation & Setup (Windows)

### Voraussetzungen
- **Windows 10/11** (64-bit)
- **NVIDIA GPU** mit mindestens 8GB VRAM (empfohlen: 12GB+)
- **Python 3.9-3.11** (Python 3.12+ kann Kompatibilitätsprobleme haben)
- **Git** für Repository-Kloning

### 1. Repository klonen
```bash
# Repository klonen
git clone https://github.com/YOUR_USERNAME/LLM-CODING.git
cd LLM-CODING
```

### 2. Python Environment Setup

#### Option A: Conda (Empfohlen)
```bash
# Conda installieren (falls nicht vorhanden)
# Download: https://docs.conda.io/en/latest/miniconda.html

# Neue Conda Environment erstellen
conda create -n llm_cuda python=3.10 -y
conda activate llm_cuda

# CUDA Toolkit installieren (für PyTorch)
conda install nvidia/label/cuda-12.1::cuda-toolkit -y
```

#### Option B: Python venv
```bash
# Virtual Environment erstellen
python -m venv llm_env

# Environment aktivieren
llm_env\Scripts\activate

# pip upgraden
python -m pip install --upgrade pip
```

### 3. PyTorch mit CUDA installieren
```bash
# Aktiviere deine Environment (conda oder venv)
conda activate llm_cuda  # oder: llm_env\Scripts\activate

# PyTorch 2.5+ mit CUDA 12.1 installieren
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA Installation testen
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"
```

**Erwartete Ausgabe:**
```
CUDA available: True
CUDA version: 12.1
GPU: NVIDIA GeForce RTX 4070 Ti
```

### 4. Dependencies installieren
```bash
# Hauptabhängigkeiten
pip install transformers>=4.40.0
pip install tqdm
pip install psutil
pip install numpy

# Optional: Für erweiterte Features
pip install wandb  # Für Experiment Tracking
pip install tensorboard  # Für Visualisierung
```

### 5. GPU Setup validieren
```bash
# GPU Training Test
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
    print('❌ CUDA nicht verfügbar - prüfe GPU-Treiber')
"
```

### 6. Triton Cache Setup (Windows Fix)
```bash
# Triton Cache Ordner erstellen (verhindert torch.compile Fehler)
mkdir %TEMP%\triton_cache 2>nul

# Environment Variable setzen (optional, wird automatisch gesetzt)
set TRITON_CACHE_DIR=%TEMP%\triton_cache
```

### 7. Erstes Training testen
```bash
# Kurzer Test-Lauf (10 Steps)
python gpu_training_optimized.py
```

**Erwartete Ausgabe:**
```
=== GPU SETUP CHECK ===
✅ CUDA verfügbar: 12.1
🖥️  GPUs gefunden: 1
   GPU 0: NVIDIA GeForce RTX 4070 Ti
   Memory: 12.9 GB
   ✅ Ausreichend für mittlere Modelle

🔧 GPU-Optimized LLM initialisiert:
   Hidden Size: 1536
   Layers: 12
   Total Parameters: 733,354,496 (0.73B)

🚀 torch.compile wird aktiviert...
✅ torch.compile aktiviert

🎯 Training gestartet...
🚀 Training: 5/50 [██▌...] 10% | Loss: 8.2847 | VRAM: 8.2/12.9GB | GPU%: 67% | Step/s: 1.23
```

## 🚀 Schnellstart

### 1. Training starten
```bash
# GPU-optimiertes Training
python gpu_training_optimized.py
```

**Was passiert:**
1. **GPU-Setup Check** - Erkennt verfügbare Hardware
2. **Model Initialization** - Lädt moderne LLM-Architektur
3. **torch.compile** - Optimiert Model für Performance
4. **Training Loop** - Mit Real-time Monitoring
5. **Model Saving** - Speichert Checkpoint

### 2. Text Generation
```bash
# Interactive Text Generation
python text_generator.py
```

**Befehle:**
- `sample The quick brown fox` - Sampling-basierte Generation
- `greedy Once upon a time` - Deterministische Generation  
- `batch Tell me about AI` - Mehrere Varianten
- `settings` - Aktuelle Einstellungen anzeigen
- `help` - Hilfe anzeigen
- `quit` - Beenden

## ⚙️ Performance-Optimierungen

### Training-Performance
- **torch.compile** mit `max-autotune`: 20-40% Speedup
- **Fused AdamW**: 10-20% Optimizer-Speedup
- **Mixed Precision BF16**: 50% Memory-Reduktion
- **Flash Attention 2**: 3-5x schnellere Attention
- **Optimized DataLoader**: pin_memory, non_blocking

### Memory-Optimierungen
- **Gradient Checkpointing**: Optional für Memory vs. Speed
- **Micro-Batching**: Kleine Batches mit Gradient Accumulation
- **GQA (4:1 Ratio)**: 25% weniger KV-Cache Memory
- **BF16 Mixed Precision**: Halbierter Memory-Verbrauch

### Hardware-Anforderungen & Konfigurationen

#### **RTX 3060 (12GB) - Budget Setup**
```python
# gpu_training_optimized.py anpassen:
batch_size: int = 3
hidden_size: int = 1024
num_layers: int = 10
sequence_length: int = 256

# Erwartete Performance:
# - ~500M Parameter Model
# - ~1.5-2.0 Steps/sec
# - VRAM: 8-10GB
```

#### **RTX 4070 Ti (12GB) - Empfohlen (Standard)**
```python
# Aktuelle Konfiguration (optimal):
batch_size: int = 5
hidden_size: int = 1536
num_layers: int = 12
sequence_length: int = 384

# Erwartete Performance:
# - ~730M Parameter Model
# - ~1.2-1.5 Steps/sec
# - VRAM: 10-12GB
```

#### **RTX 4080/4090 (16-24GB) - High-End**
```python
# Für größere Modelle:
batch_size: int = 8
hidden_size: int = 2048
num_layers: int = 20
sequence_length: int = 512

# Erwartete Performance:
# - ~1.5B Parameter Model
# - ~0.8-1.2 Steps/sec
# - VRAM: 14-20GB
```

#### **Multi-GPU Setup (Fortgeschritten)**
```python
# Für 2x RTX 4090 oder ähnlich:
# Distributed Training aktivieren
# (Erfordert zusätzliche Konfiguration)
```

## 📊 Monitoring & Metriken

Das System bietet Real-time Monitoring:
```
🚀 Training: 45/2000 [██▌...] 2% | Loss: 3.2847 | VRAM: 8.2/12.9GB | GPU%: 67% | Step/s: 1.23
```

**Metriken:**
- **Loss**: Training Loss (sollte sinken)
- **VRAM**: GPU Memory Usage
- **GPU%**: GPU Utilization (Ziel: 60-80%)
- **Step/s**: Training Speed

## 🔧 Konfiguration & Tuning

### Für verschiedene GPUs anpassen:

**RTX 3060 (12GB):**
```python
batch_size = 3
hidden_size = 1024
num_layers = 10
```

**RTX 4070 Ti (12GB):**
```python
batch_size = 5          # Aktuell
hidden_size = 1536
num_layers = 12
```

**RTX 4090 (24GB):**
```python
batch_size = 12
hidden_size = 2048
num_layers = 20
```

## 🎯 Nächste Schritte

1. **Training durchführen** mit `gpu_training_optimized.py`
2. **Model testen** mit `text_generator.py`
3. **Hyperparameter tunen** für bessere Performance
4. **Größere Modelle** auf besserer Hardware trainieren
5. **Custom Datasets** für spezifische Anwendungen

## 🔬 Technische Details

### Training-Pipeline Schritt-für-Schritt

#### 1. **Initialization Phase**
```python
# 1. GPU Setup & Validation
check_gpu_setup()  # Erkennt Hardware, prüft CUDA

# 2. Model Creation
model = GPUOptimizedLLM(config)  # Lädt moderne Architektur
model.to(device)  # GPU Transfer

# 3. torch.compile Optimization
model = torch.compile(model, mode="max-autotune")  # Graph-Optimierung
```

#### 2. **Training Loop**
```python
for step in range(max_steps):
    # Micro-Batching für Memory-Effizienz
    for micro_step in range(gradient_accumulation_steps):
        # Mixed Precision Forward Pass
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(input_ids, labels=labels)
            loss = outputs["loss"] / gradient_accumulation_steps

        # Mixed Precision Backward Pass
        scaler.scale(loss).backward()

    # Optimizer Step mit Gradient Clipping
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
```

#### 3. **Memory Management**
- **Gradient Accumulation**: Simuliert große Batches ohne Memory-Overhead
- **Mixed Precision**: BF16 für Activations, FP32 für kritische Berechnungen
- **torch.compile**: Automatische Memory-Layout Optimierung

### Model-Architektur Deep Dive

#### **Grouped-Query Attention (GQA)**
```python
# Standard Multi-Head Attention: Q, K, V haben gleiche Anzahl Heads
num_q_heads = 24
num_k_heads = 24  # Gleich wie Q
num_v_heads = 24  # Gleich wie Q

# GQA: Weniger K/V Heads für Memory-Effizienz
num_q_heads = 24
num_k_heads = 6   # 4:1 Ratio
num_v_heads = 6   # 4:1 Ratio

# Memory-Ersparnis: ~25% weniger KV-Cache
```

#### **Rotary Position Embeddings (RoPE)**
```python
# Traditionell: Additive Position Embeddings
x = token_embeddings + position_embeddings

# RoPE: Rotational Position Encoding
def apply_rope(q, k, position):
    cos, sin = get_rope_cache(position)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot
```

#### **SwiGLU Activation**
```python
# Standard FFN mit ReLU/GELU
def standard_ffn(x):
    return relu(linear1(x)) @ linear2

# SwiGLU: Bessere Performance
def swiglu_ffn(x):
    gate = silu(gate_proj(x))      # SiLU Activation
    up = up_proj(x)                # Linear ohne Activation
    return down_proj(gate * up)    # Element-wise Multiplikation
```

### Optimizer-Strategien

#### **Fused AdamW vs. Standard AdamW**
```python
# Standard AdamW: Separate Kernel für jeden Parameter
for param in parameters:
    # Momentum Update (separater Kernel)
    exp_avg = beta1 * exp_avg + (1 - beta1) * grad
    # Variance Update (separater Kernel)
    exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad**2
    # Parameter Update (separater Kernel)
    param -= lr * exp_avg / (sqrt(exp_avg_sq) + eps)

# Fused AdamW: Ein Kernel für alle Updates
fused_adamw_kernel(parameters, gradients, lr, beta1, beta2, eps)
# → 10-20% Speedup durch weniger Kernel-Launches
```

#### **Muon Optimizer**
```python
# Newton-Schulz Orthogonalization für 2D Matrices
def newton_schulz_step(W, num_steps=5):
    """Orthogonalisiert Weight Matrix für stabilere Updates."""
    Y = W
    for _ in range(num_steps):
        Y = 1.5 * Y - 0.5 * Y @ Y.T @ Y
    return Y

# Bessere Konvergenz für Linear Layers in Transformern
```

### Inference-Optimierungen

#### **Sampling-Strategien Vergleich**
```python
# Greedy: Immer bestes Token (deterministisch)
next_token = torch.argmax(logits, dim=-1)

# Temperature: Kontrollierte Randomness
probs = softmax(logits / temperature)
next_token = torch.multinomial(probs, 1)

# Top-p (Nucleus): Dynamische Auswahl
sorted_probs, indices = torch.sort(probs, descending=True)
cumsum = torch.cumsum(sorted_probs, dim=-1)
mask = cumsum <= top_p
filtered_probs = probs * mask
next_token = torch.multinomial(filtered_probs, 1)
```

#### **Batched Generation**
```python
# Parallel Generation für mehrere Prompts
batch_prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
batch_input_ids = tokenizer(batch_prompts, padding=True)

# Gleichzeitige Generation aller Sequenzen
for step in range(max_length):
    with torch.amp.autocast('cuda'):
        logits = model(batch_input_ids)
    next_tokens = sample(logits, temperature, top_p)
    batch_input_ids = torch.cat([batch_input_ids, next_tokens], dim=1)
```

## 🐛 Troubleshooting

### Installation-Probleme

#### **CUDA nicht verfügbar**
```bash
# Problem: torch.cuda.is_available() = False

# Lösung 1: GPU-Treiber aktualisieren
# Download: https://www.nvidia.com/drivers/
# Mindestens: 526.98+ für CUDA 12.1

# Lösung 2: PyTorch neu installieren
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Lösung 3: CUDA Toolkit prüfen
nvidia-smi  # Sollte GPU und CUDA Version anzeigen
```

#### **torch.compile Fehler**
```bash
# Problem: "Triton not found" oder Compile-Fehler

# Lösung 1: Triton Cache Fix
mkdir %TEMP%\triton_cache
set TRITON_CACHE_DIR=%TEMP%\triton_cache

# Lösung 2: torch.compile deaktivieren (Fallback)
# In gpu_training_optimized.py:
use_torch_compile: bool = False
```

#### **Import Errors**
```bash
# Problem: ModuleNotFoundError

# Lösung: Dependencies neu installieren
pip install --upgrade transformers tqdm psutil numpy

# Für spezifische Fehler:
pip install --upgrade torch  # PyTorch Update
pip install --upgrade setuptools wheel  # Build Tools
```

#### **Memory Errors beim Start**
```bash
# Problem: "CUDA out of memory" beim Model Loading

# Lösung: Kleinere Model-Konfiguration
# In gpu_training_optimized.py anpassen:
batch_size: int = 3           # Reduziert von 5
hidden_size: int = 1024       # Reduziert von 1536
num_layers: int = 10          # Reduziert von 12
```

#### **Langsame Performance**
```bash
# Problem: <0.5 Steps/sec

# Checks:
1. GPU Utilization prüfen: nvidia-smi
2. torch.compile aktiviert? use_torch_compile = True
3. Mixed Precision aktiviert? use_mixed_precision = True
4. Batch Size zu klein? Erhöhe auf 6-8 (wenn VRAM erlaubt)
```

### Training-Probleme

#### **CUDA Out of Memory**
```bash
# Lösungen:
1. Batch Size reduzieren: batch_size = 3
2. Sequence Length reduzieren: sequence_length = 256
3. Gradient Checkpointing aktivieren: gradient_checkpointing = True
4. Model Size reduzieren: hidden_size = 1024, num_layers = 10
```

#### **Langsames Training**
```bash
# Optimierungen:
1. torch.compile aktivieren: use_torch_compile = True
2. Fused AdamW verwenden: fused=True
3. Gradient Checkpointing deaktivieren: gradient_checkpointing = False
4. Batch Size erhöhen: batch_size = 8 (wenn VRAM erlaubt)
```

#### **Instabile Loss**
```bash
# Fixes:
1. Learning Rate reduzieren: learning_rate = 5e-5
2. Gradient Clipping: max_grad_norm = 1.0
3. Warmup Steps hinzufügen
4. Mixed Precision prüfen: use_mixed_precision = True
```

## 📋 Quick Reference

### Wichtige Befehle
```bash
# Environment aktivieren
conda activate llm_cuda

# Training starten
python gpu_training_optimized.py

# Text Generation
python text_generator.py

# GPU Status prüfen
nvidia-smi

# CUDA Test
python -c "import torch; print(torch.cuda.is_available())"
```

### Wichtige Dateien bearbeiten
```bash
# Training-Konfiguration anpassen
notepad gpu_training_optimized.py  # Zeile 183-208 (GPUTrainingConfig)

# Model-Architektur ändern
notepad modern_llm.py  # Zeile 22-35 (ModelConfig)

# Generation-Parameter anpassen
notepad text_generator.py  # Zeile 390-398 (generate() Aufruf)
```

### Performance-Monitoring
```bash
# GPU Utilization live anzeigen
nvidia-smi -l 1

# Training Progress
# Achte auf: Loss (sollte sinken), VRAM (nicht >95%), GPU% (60-80% optimal)
```

### Backup & Sharing
```bash
# Model Checkpoint sichern
copy final_model.pt backup_model_YYYY-MM-DD.pt

# Konfiguration dokumentieren
echo "Training Config: batch_size=5, hidden_size=1536, steps=2000" > training_log.txt
```

## 🔗 Weiterführende Links

- **PyTorch Documentation**: https://pytorch.org/docs/stable/
- **Transformers Library**: https://huggingface.co/docs/transformers/
- **CUDA Toolkit**: https://developer.nvidia.com/cuda-toolkit
- **GPU Drivers**: https://www.nvidia.com/drivers/

## 📞 Support

Bei Problemen:
1. **Prüfe Troubleshooting-Sektion** oben
2. **Validiere GPU Setup** mit den Test-Befehlen
3. **Reduziere Model-Größe** bei Memory-Problemen
4. **Deaktiviere torch.compile** bei Compile-Fehlern

---

**🎉 Viel Erfolg beim Training deines eigenen LLMs!**

*Dieses System implementiert State-of-the-Art Techniken für effizientes LLM-Training auf Consumer-Hardware.*

**⭐ Wenn dir dieses Projekt geholfen hat, gib ihm einen Star auf GitHub!**
