# 🔧 torch.compile CUDAGraphs Fix - Arbeitsanleitung

## 📋 **Projekt Status:**
- **Original Projekt:** Funktioniert auf Windows, aber torch.compile Probleme auf Linux
- **Aktueller Ordner:** `LLM-CODING-linux` 
- **Ziel:** torch.compile CUDAGraphs Fehler beheben für Linux/RunPod
- **GPU:** RTX 3090 (25GB VRAM) auf RunPod

## 🚨 **Das Problem:**
```
RuntimeError: Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run.
Stack trace: File "modern_llm.py", line 403, in forward
    hidden_states = self.token_embeddings(input_ids)
To prevent overwriting, clone the tensor outside of torch.compile() or call torch.compiler.cudagraph_mark_step_begin() before each model invocation.
```

## 🎯 **Was zu tun ist:**

### **1. Problem-Dateien identifiziert:**
- `modern_llm.py` - Model Forward Pass überschreibt Tensoren
- `gpu_training_optimized.py` - Training Loop ohne CUDAGraphs Markierung
- `config.py` - torch.compile Einstellungen zu aggressiv

### **2. Fixes implementieren:**

#### **Fix A: modern_llm.py - Tensor Cloning**
**Problem:** Token Embeddings und Attention Tensoren werden überschrieben

**Lösung:** Tensoren vor Verwendung klonen
```python
# In ModernLLM.forward():
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # HINZUFÜGEN: Clone input to prevent CUDAGraphs overwriting
    x = x.clone()
    x = self.token_embedding(x) * math.sqrt(self.config.d_model)
    # ... rest unchanged
```

#### **Fix B: GroupedQueryAttention - Q,K,V Cloning**
```python
# In GroupedQueryAttention.forward():
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # HINZUFÜGEN: Clone attention inputs
    x = x.clone()
    
    # ÄNDERN: Clone Q, K, V projections
    q = self.q_proj(x).clone()
    k = self.k_proj(x).clone() 
    v = self.v_proj(x).clone()
    # ... rest unchanged
```

#### **Fix C: RoPE Cache Cloning**
```python
# In Rotary.forward():
def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int):
    # ÄNDERN: Clone RoPE cache
    cos = self._cos_cached[:seq_len].clone()
    sin = self._sin_cached[:seq_len].clone()
    # ... rest unchanged
```

#### **Fix D: Training Loop CUDAGraphs Marking**
```python
# In gpu_training_optimized.py training loop:
for step in range(max_steps):
    # HINZUFÜGEN: Mark CUDAGraphs step beginning
    if training_config.use_torch_compile:
        torch.compiler.cudagraph_mark_step_begin()
    
    for micro_step in range(gradient_accumulation_steps):
        # ... existing training code
```

#### **Fix E: torch.compile Mode weniger aggressiv**
```python
# In gpu_training_optimized.py:
# ÄNDERN von:
model = torch.compile(model, mode="max-autotune")

# ZU:
model = torch.compile(
    model, 
    mode="reduce-overhead",  # Weniger aggressiv
    dynamic=True,           # Dynamische Shapes
    fullgraph=False         # Partial Graphs OK
)
```

### **3. Test-Strategie:**

#### **Phase 1: Minimaler Test**
```python
# config.py anpassen für schnellen Test:
training_config = TrainingConfig(
    max_steps=10,           # Nur 10 Steps
    batch_size=2,           # Kleine Batches
    use_torch_compile=True, # Mit Fixes testen
)
```

#### **Phase 2: Performance Test**
```python
# Nach erfolgreichem Fix:
training_config = TrainingConfig(
    max_steps=100,          # 100 Steps
    batch_size=5,           # Normale Batches
    use_torch_compile=True,
)
```

#### **Phase 3: Vollständiges Training**
```python
# Finale Konfiguration für RTX 3090:
model_config = ModelConfig(
    hidden_size=2048,       # Größeres Model
    num_layers=20,
)

training_config = TrainingConfig(
    max_steps=10000,
    batch_size=8,
    use_torch_compile=True,
)
```

## 🔍 **Debugging Tools:**

### **CUDAGraphs Status prüfen:**
```python
# In Jupyter/Python:
import torch
print(f"CUDAGraphs available: {torch.backends.cuda.is_built()}")
print(f"torch.compile available: {hasattr(torch, 'compile')}")
print(f"Triton available: {torch.backends.cuda.is_built()}")
```

### **Memory Debugging aktivieren:**
```python
# Für detaillierte Memory-Analyse:
torch.cuda.memory._record_memory_history(True)

# Nach Training:
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
```

### **torch.compile Logs:**
```python
# Detaillierte Compile-Logs:
import logging
logging.getLogger("torch._inductor").setLevel(logging.INFO)
torch._dynamo.config.verbose = True
```

## 📊 **Erwartete Performance nach Fix:**

### **Ohne torch.compile (aktuell funktionierend):**
- Speed: ~1.0-1.2 Steps/sec
- VRAM: ~8-12GB (460M Model)

### **Mit torch.compile (nach Fix):**
- Speed: ~1.4-1.8 Steps/sec (+40% Speedup)
- VRAM: ~9-13GB (+10% Memory)
- Stabilität: ✅ Keine Crashes

### **Größeres Model (1.5B Parameter):**
- Speed: ~1.0-1.4 Steps/sec
- VRAM: ~16-21GB
- Training: ~2-3h für 10K Steps

## 🎯 **Arbeitsschritte für VS Code:**

### **1. Projekt öffnen:**
```bash
# VS Code öffnen in:
C:\Users\Matthias\Documents\augment-projects\LLM-CODING-linux
```

### **2. Dateien bearbeiten (Reihenfolge):**
1. **modern_llm.py** - Tensor Cloning Fixes
2. **gpu_training_optimized.py** - Training Loop Fixes  
3. **config.py** - torch.compile Einstellungen
4. **Test ausführen** - Minimaler Test

### **3. Test-Kommandos:**
```bash
# Schneller Test (10 Steps):
python gpu_training_optimized.py

# Performance Test (100 Steps):
# config.py: max_steps=100

# Vollständiges Training:
# config.py: max_steps=10000
```

## ✅ **Erfolgskriterien:**

### **Fix erfolgreich wenn:**
- ✅ Keine CUDAGraphs RuntimeError
- ✅ Training läuft durch ohne Crash
- ✅ torch.compile Speedup sichtbar
- ✅ Memory Usage stabil

### **Erwartete Ausgabe:**
```
🚀 torch.compile wird aktiviert...
✅ torch.compile aktiviert (mit CUDAGraphs Fixes)
🎯 Training gestartet...
🚀 Training: 1/10 [██████████] 10% | Loss: 10.28 | VRAM: 9.2/25.3GB | GPU%: 87% | Step/s: 1.6
🚀 Training: 2/10 [████████████████████] 20% | Loss: 9.84 | VRAM: 9.2/25.3GB | GPU%: 89% | Step/s: 1.7
```

## 🚀 **Nach erfolgreichem Fix:**

### **Nächste Optimierungen:**
1. **Größeres Model** (1.5-2B Parameter)
2. **Längere Sequenzen** (1024-2048 Tokens)
3. **Batch Size Optimierung**
4. **Multi-GPU Support** (falls verfügbar)

### **Performance Monitoring:**
```bash
# GPU Monitoring während Training:
watch -n 1 nvidia-smi

# Detaillierte Performance-Logs:
python -m torch.profiler gpu_training_optimized.py
```

Das ist der Plan! Lass uns systematisch die Fixes implementieren und torch.compile zum Laufen bringen! 🎯
