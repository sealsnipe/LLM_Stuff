#!/usr/bin/env python3
"""
🐧 Linux-Optimized LLM Training with FineWeb-Edu
Vereint Windows- und Linux-Projekte für maximale Performance

Features:
- Linux-spezifische CUDA/Triton Optimierungen
- FineWeb-Edu Dataset Integration (27GB sample-10BT)
- Vocabulary-Fix (49152 für SmolLM-Kompatibilität)
- Memory-optimiertes Training
- torch.compile mit Linux-Optimierungen
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
import math
import time
import os
import platform
import sys
import gc
import psutil
from typing import Dict, Optional
from dataclasses import dataclass
from tqdm import tqdm

# Import unserer Module
from config import model_config, training_config, hardware_config, system_config, dataset_config
from dataset_loader import create_fineweb_dataloader
from modern_llm import ModernLLM

def setup_linux_optimizations():
    """Linux-spezifische Optimierungen für maximale Performance."""
    if platform.system() == "Linux":
        print("🐧 Linux-Optimierungen aktiviert...")
        
        # CUDA Optimierungen für Linux
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        
        # Triton Optimierungen
        triton_cache = '/tmp/triton_cache'
        os.environ['TRITON_CACHE_DIR'] = triton_cache
        os.makedirs(triton_cache, exist_ok=True)
        
        # PyTorch Backend Optimierungen
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Dynamo Optimierungen für torch.compile
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.verbose = False
        torch._dynamo.config.cache_size_limit = 2000
        torch._dynamo.config.automatic_dynamic_shapes = True
        torch._dynamo.config.assume_static_by_default = False
        
        print("✅ Linux GPU-Optimierungen aktiviert")
        print(f"   Triton Cache: {triton_cache}")
        print(f"   CUDA Memory: max_split_size_mb=512")
        return True
    else:
        print(f"⚠️  Läuft auf {platform.system()} - Linux-Optimierungen übersprungen")
        return False

def setup_windows_compatibility():
    """Windows-Kompatibilität für gemischte Umgebungen."""
    if platform.system() == "Windows":
        print("🪟 Windows-Kompatibilität aktiviert...")
        
        # Windows-spezifische CUDA Optimierungen
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
        
        # Weniger aggressive Optimierungen für Windows
        torch.backends.cudnn.benchmark = True
        
        print("✅ Windows-Optimierungen aktiviert")
        return True
    return False

class MemoryMonitor:
    """Memory-Monitoring für beide Plattformen."""
    
    def __init__(self):
        self.process = psutil.Process()
    
    def get_memory_stats(self):
        """Aktuelle Memory-Statistiken."""
        stats = {}
        
        # System Memory
        memory = psutil.virtual_memory()
        stats['system_total'] = memory.total / (1024**3)
        stats['system_used'] = memory.used / (1024**3)
        stats['system_free'] = memory.available / (1024**3)
        
        # Process Memory
        process_memory = self.process.memory_info()
        stats['process_rss'] = process_memory.rss / (1024**3)
        
        # GPU Memory (wenn verfügbar)
        if torch.cuda.is_available():
            stats['gpu_allocated'] = torch.cuda.memory_allocated() / (1024**3)
            stats['gpu_reserved'] = torch.cuda.memory_reserved() / (1024**3)
            stats['gpu_max_allocated'] = torch.cuda.max_memory_allocated() / (1024**3)
        
        return stats
    
    def print_memory_stats(self, prefix="📊 Memory: "):
        """Druckt Memory-Statistiken."""
        stats = self.get_memory_stats()
        
        print(f"{prefix}")
        print(f"   System: {stats['system_used']:.1f}GB / {stats['system_total']:.1f}GB")
        print(f"   Process: {stats['process_rss']:.1f}GB")
        
        if 'gpu_allocated' in stats:
            print(f"   GPU: {stats['gpu_allocated']:.1f}GB allocated, {stats['gpu_reserved']:.1f}GB reserved")

def check_gpu_setup():
    """Überprüft GPU-Setup für beide Plattformen."""
    if not torch.cuda.is_available():
        print("❌ CUDA nicht verfügbar")
        return False
    
    gpu_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(current_device)
    
    print(f"✅ GPU Setup:")
    print(f"   Devices: {gpu_count}")
    print(f"   Current: {current_device} ({gpu_name})")
    print(f"   Memory: {torch.cuda.get_device_properties(current_device).total_memory / (1024**3):.1f}GB")
    
    return True

def create_optimized_dataset(num_samples: int = 100000, use_real_data: bool = True, dataset_size: str = "medium"):
    """
    Erstellt optimierten Datensatz mit FineWeb-Edu (Linux/Windows kompatibel).
    """
    
    if use_real_data:
        print("🌐 Loading FineWeb-Edu Dataset (sample-10BT, ~27GB)...")
        print("   Nutzt gecachte Parquet-Dateien für schnelles Laden")
        
        try:
            # Verwende unser optimiertes Dataset-Loading
            dataloader = create_fineweb_dataloader(
                dataset_size=dataset_size,
                num_samples=num_samples,
                batch_size=training_config.batch_size,
                streaming=False  # Verwende gecachte Daten
            )
            
            print(f"✅ FineWeb-Edu Dataset geladen!")
            print(f"   Samples: {num_samples:,}")
            print(f"   Batch Size: {training_config.batch_size}")
            print(f"   Sequence Length: {training_config.sequence_length}")
            
            return dataloader
            
        except Exception as e:
            print(f"❌ Error loading FineWeb-Edu dataset: {e}")
            print("💡 Fallback zu synthetischen Daten...")
            use_real_data = False

    if not use_real_data:
        print("🔧 Creating synthetic dataset for testing...")
        # Fallback: Synthetic data
        input_ids = torch.randint(0, model_config.vocab_size, (num_samples, training_config.sequence_length))
        labels = input_ids.clone()

        dataset = TensorDataset(input_ids, labels)

        return DataLoader(
            dataset,
            batch_size=training_config.batch_size,
            shuffle=True,
            num_workers=training_config.dataloader_num_workers,
            pin_memory=training_config.dataloader_pin_memory,
            persistent_workers=training_config.dataloader_persistent_workers,
            prefetch_factor=training_config.dataloader_prefetch_factor,
            drop_last=True
        )

def linux_optimized_training_loop(use_real_data: bool = True, dataset_size: str = "medium"):
    """Linux-optimierte Training-Loop mit FineWeb-Edu."""
    
    # Platform-spezifische Optimierungen
    is_linux = setup_linux_optimizations()
    is_windows = setup_windows_compatibility()
    
    # GPU Setup Check
    if not check_gpu_setup():
        print("🚫 GPU Training nicht möglich")
        return
    
    device = torch.device("cuda")
    memory_monitor = MemoryMonitor()
    
    print(f"\n🚀 Linux-Optimized Training gestartet")
    print(f"   Platform: {platform.system()}")
    print(f"   Model: {model_config.hidden_size}d, {model_config.num_layers}L")
    print(f"   Vocabulary: {model_config.vocab_size:,} (SmolLM-kompatibel)")
    print(f"   Training: {training_config.max_steps} steps, batch {training_config.batch_size}")
    print(f"   Dataset: FineWeb-Edu {dataset_size}")
    print()
    
    # Dataset - FineWeb-Edu (27GB sample-10BT)
    # Verwende num_samples aus der Config für das gewählte dataset_size
    config_info = dataset_config.dataset_sizes.get(dataset_size, {"num_samples": 100000})
    num_samples = config_info["num_samples"]

    # Import intelligent dataloader from Windows training
    from training_windows import create_intelligent_dataloader

    dataloader = create_intelligent_dataloader(
        num_samples=num_samples,  # Aus Config: medium=300k, large=1M, etc.
        use_real_data=use_real_data,
        dataset_size=dataset_size
    )
    
    memory_monitor.print_memory_stats("📊 After Dataset Loading: ")
    
    # Model - mit korrekter Vocabulary-Größe
    print("🤖 Creating Model...")
    model = ModernLLM(model_config).to(device)
    
    # Parameter Count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    
    memory_monitor.print_memory_stats("📊 After Model Loading: ")
    
    # Optimizer
    print("🔧 Setting up Optimizer...")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8
    )
    
    # Mixed Precision
    scaler = torch.cuda.amp.GradScaler() if training_config.use_mixed_precision else None
    if scaler:
        print("⚡ Mixed Precision (FP16) aktiviert")
    
    # torch.compile (optimiert für Stabilität)
    if training_config.use_torch_compile and is_linux:
        print("🔥 torch.compile wird aktiviert (Linux-optimiert für Stabilität)...")
        try:
            # Linux: Stabile Optimierung mit weniger CUDAGraphs-Problemen
            model = torch.compile(
                model,
                mode="reduce-overhead",  # Stabiler als max-autotune
                dynamic=True,           # Dynamische Shapes erlauben
                fullgraph=False         # Partial Graphs OK
            )
            print("✅ torch.compile aktiviert (Linux)")
            print("   Mode: reduce-overhead (15-25% Speedup, stabil)")
            print("   Dynamic shapes: True")
        except Exception as e:
            print(f"⚠️  torch.compile Fehler: {e}")
            print("💡 Fallback: Standard PyTorch")
    elif training_config.use_torch_compile and is_windows:
        print("🔥 torch.compile wird aktiviert (Windows-kompatibel)...")
        try:
            # Windows: Konservative Optimierung
            model = torch.compile(
                model,
                mode="default",         # Sicher für Windows
                fullgraph=False
            )
            print("✅ torch.compile aktiviert (Windows)")
            print("   Mode: default (10-15% Speedup, sehr stabil)")
        except Exception as e:
            print(f"⚠️  torch.compile Fehler: {e}")
            print("💡 Fallback: Standard PyTorch")
    else:
        print("🔧 torch.compile deaktiviert")
    
    memory_monitor.print_memory_stats("📊 After Model Compilation: ")
    
    # Training Loop
    print("\n🚀 Training startet...")
    model.train()
    
    step = 0
    total_loss = 0.0
    start_time = time.time()
    
    progress_bar = tqdm(total=training_config.max_steps, desc="🚀 Training")
    
    try:
        for epoch in range(100):  # Viele Epochen
            for batch_idx, batch in enumerate(dataloader):
                if step >= training_config.max_steps:
                    break
                
                # CUDAGraphs Step Markierung für Linux (torch.compile Fix)
                if is_linux and training_config.use_torch_compile:
                    torch.compiler.cudagraph_mark_step_begin()

                # Handle FineWeb-Edu batch format
                if isinstance(batch, dict):
                    input_ids = batch['input_ids'].to(device, non_blocking=True)
                    labels = input_ids.clone()  # Clone für CUDAGraphs-Kompatibilität
                else:
                    input_ids, labels = [x.to(device, non_blocking=True) for x in batch]
                    # Clone inputs für CUDAGraphs-Kompatibilität
                    input_ids = input_ids.clone()
                    labels = labels.clone()
                
                # Forward Pass mit Mixed Precision
                with torch.cuda.amp.autocast(enabled=training_config.use_mixed_precision):
                    outputs = model(input_ids)
                    
                    # Causal LM Loss
                    shift_logits = outputs[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100
                    )
                
                # Backward Pass
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.max_grad_norm)
                    optimizer.step()
                
                optimizer.zero_grad()
                
                # Statistiken
                total_loss += loss.item()
                step += 1
                
                # Enhanced Progress Update
                if step % 10 == 0:
                    avg_loss = total_loss / step
                    elapsed = time.time() - start_time
                    steps_per_sec = step / elapsed

                    # GPU Stats
                    gpu_memory_used = torch.cuda.memory_allocated() / (1024**3)
                    gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    gpu_utilization = f"{gpu_memory_used:.1f}/{gpu_memory_total:.1f}GB"

                    # ETA Calculation
                    remaining_steps = training_config.max_steps - step
                    eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                    eta_minutes = eta_seconds / 60

                    progress_bar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'step/s': f'{steps_per_sec:.2f}',
                        'GPU': gpu_utilization,
                        'ETA': f'{eta_minutes:.1f}min'
                    })
                    progress_bar.update(10)

                # Detailed Stats every 100 steps
                if step % 100 == 0:
                    memory_stats = memory_monitor.get_memory_stats()
                    print(f"\n📊 Step {step:,} Detailed Stats:")
                    print(f"   Loss: {avg_loss:.6f}")
                    print(f"   Speed: {steps_per_sec:.2f} steps/sec")
                    print(f"   GPU Memory: {memory_stats.get('gpu_allocated', 0):.1f}GB allocated")
                    print(f"   System Memory: {memory_stats.get('process_rss', 0):.1f}GB")
                    print(f"   ETA: {eta_minutes:.1f} minutes remaining")
                    if is_linux:
                        print(f"   Platform: Linux (torch.compile optimized)")
                    else:
                        print(f"   Platform: Windows (compatible mode)")
                    print()
                
                if step >= training_config.max_steps:
                    break
            
            if step >= training_config.max_steps:
                break
    
    except KeyboardInterrupt:
        print("\n⏹️  Training unterbrochen")
    except Exception as e:
        print(f"\n❌ Training Fehler: {e}")
        import traceback
        traceback.print_exc()
    finally:
        progress_bar.close()
    
    # Final Statistics
    elapsed = time.time() - start_time
    avg_loss = total_loss / max(step, 1)
    
    print(f"\n🎯 Training abgeschlossen!")
    print(f"   Steps: {step:,}")
    print(f"   Zeit: {elapsed/60:.1f} Minuten")
    print(f"   Durchschnittlicher Loss: {avg_loss:.4f}")
    print(f"   Steps/Sekunde: {step/elapsed:.2f}")
    
    memory_monitor.print_memory_stats("📊 Final Memory: ")

if __name__ == "__main__":
    print("=== 🐧 LINUX-OPTIMIZED FINEWEB-EDU TRAINING ===")
    print(f"📊 Model: {model_config.hidden_size}d, {model_config.num_layers} layers")
    print(f"🚀 Training: {training_config.max_steps} steps, batch {training_config.batch_size}")
    print(f"⚡ Optimizations: torch.compile={training_config.use_torch_compile}, mixed_precision={training_config.use_mixed_precision}")
    print(f"🎯 Dataset: FineWeb-Edu sample-10BT (~27GB, 100k samples)")
    print(f"🐧 Platform: {platform.system()}")
    print()

    # Start Linux-optimized Training
    linux_optimized_training_loop(
        use_real_data=True,                                    # 🌐 FineWeb-Edu verwenden!
        dataset_size=dataset_config.default_dataset_size      # Aus Config: medium, large, etc.
    )
