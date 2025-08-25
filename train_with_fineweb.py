#!/usr/bin/env python3
"""
🚀 LLM Training mit FineWeb-Edu Dataset
Kombiniert unser optimiertes Training mit echten Daten
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import platform
from tqdm import tqdm
import gc

# Import unserer Module
from config import model_config, training_config, hardware_config, system_config, dataset_config
from dataset_loader import create_fineweb_dataloader, show_dataset_configs
from gpu_training_optimized import MemoryOptimizedLLM, MemoryMonitor, check_gpu_setup

def setup_windows_optimizations():
    """Windows-spezifische Optimierungen."""
    if platform.system() == "Windows":
        print("🪟 Windows-Optimierungen aktiviert...")
        
        # CUDA Optimierungen für Windows
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        
        # Triton Cache für Windows
        triton_cache = os.path.join(os.environ.get('TEMP', 'C:\\temp'), 'triton_cache')
        os.environ['TRITON_CACHE_DIR'] = triton_cache
        os.makedirs(triton_cache, exist_ok=True)
        
        # PyTorch Optimierungen
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        print("✅ Windows GPU-Optimierungen aktiviert")

def train_with_fineweb(
    dataset_size: str = "small",  # "tiny", "small", "medium", "large"
    max_steps: int = None,
    save_every: int = 500,
    eval_every: int = 100
):
    """Training mit FineWeb-Edu Dataset."""
    
    # Windows Optimierungen
    setup_windows_optimizations()
    
    # GPU Setup Check
    if not check_gpu_setup():
        print("🚫 GPU Training nicht möglich")
        return
    
    device = torch.device("cuda")
    memory_monitor = MemoryMonitor()
    
    # Show dataset info
    if dataset_size in dataset_config.dataset_sizes:
        config_info = dataset_config.dataset_sizes[dataset_size]
        print(f"🎯 FineWeb-Edu Training gestartet ({dataset_size.upper()})")
        print(f"   Samples: {config_info['num_samples']:,}")
        print(f"   Description: {config_info['description']}")
        print(f"   Estimated Tokens: {config_info['estimated_tokens']}")
        print(f"   Estimated Time: {config_info['training_time']}")
    else:
        print(f"🎯 FineWeb-Edu Training gestartet (Custom: {dataset_size})")
    
    print(f"   Model: {model_config.hidden_size}d, {model_config.num_layers}L")
    print(f"   Max Steps: {max_steps or training_config.max_steps}")
    print()
    
    # DataLoader erstellen
    print("📦 Loading FineWeb-Edu Dataset...")
    try:
        dataloader = create_fineweb_dataloader(
            dataset_size=dataset_size,
            batch_size=training_config.batch_size,
            streaming=dataset_size in ["large", "full"]
        )
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("💡 Falling back to tiny dataset...")
        dataloader = create_fineweb_dataloader(
            dataset_size="tiny",
            batch_size=training_config.batch_size,
            streaming=False
        )
    
    memory_monitor.print_memory_stats("📊 After Dataset Loading: ")
    
    # Model erstellen
    print("🤖 Creating Model...")
    model = MemoryOptimizedLLM().to(device)
    
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
        betas=(training_config.adam_beta1, training_config.adam_beta2),
        eps=training_config.adam_eps,
        fused=True
    )
    
    # Mixed Precision Scaler
    scaler = torch.amp.GradScaler('cuda') if training_config.use_mixed_precision else None
    
    # torch.compile
    if training_config.use_torch_compile:
        print("🚀 Compiling model...")
        try:
            compile_mode = "max-autotune" if platform.system() == "Windows" else "reduce-overhead"
            model = torch.compile(model, mode=compile_mode, fullgraph=False)
            print(f"✅ torch.compile aktiviert (mode: {compile_mode})")
        except Exception as e:
            print(f"⚠️  torch.compile Fehler: {e}")
    
    memory_monitor.print_memory_stats("🚀 Ready for Training: ")
    
    # Training Loop
    model.train()
    step = 0
    max_steps = max_steps or training_config.max_steps
    
    # Checkpoints Ordner
    os.makedirs(system_config.model_save_path, exist_ok=True)
    
    print(f"\n🎯 Starting Training Loop...")
    print(f"   Target Steps: {max_steps}")
    print(f"   Batch Size: {training_config.batch_size}")
    print(f"   Gradient Accumulation: {training_config.gradient_accumulation_steps}")
    print(f"   Effective Batch Size: {training_config.batch_size * training_config.gradient_accumulation_steps}")
    print()
    
    data_iter = iter(dataloader)
    start_time = time.time()
    
    with tqdm(total=max_steps, desc="Training") as pbar:
        while step < max_steps:
            epoch_loss = 0.0
            
            # Gradient Accumulation Loop
            for micro_step in range(training_config.gradient_accumulation_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)
                
                # Move to GPU
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)
                
                # Forward Pass
                if training_config.use_mixed_precision:
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        outputs = model(input_ids, labels=labels)
                        loss = outputs["loss"] / training_config.gradient_accumulation_steps
                    
                    scaler.scale(loss).backward()
                else:
                    outputs = model(input_ids, labels=labels)
                    loss = outputs["loss"] / training_config.gradient_accumulation_steps
                    loss.backward()
                
                epoch_loss += loss.item()
            
            # Optimizer Step
            if training_config.use_mixed_precision:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.max_grad_norm)
                optimizer.step()
            
            optimizer.zero_grad()
            
            step += 1
            
            # Logging
            if step % training_config.log_interval == 0:
                elapsed = time.time() - start_time
                steps_per_sec = step / elapsed
                
                # Memory Stats
                memory_used = torch.cuda.memory_allocated() / 1e9
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                gpu_util = memory_used / memory_total * 100
                
                pbar.set_postfix({
                    'Loss': f'{epoch_loss:.4f}',
                    'Step/s': f'{steps_per_sec:.2f}',
                    'VRAM': f'{memory_used:.1f}/{memory_total:.1f}GB',
                    'GPU%': f'{gpu_util:.0f}%'
                })
            
            # Evaluation
            if step % eval_every == 0:
                print(f"\n📊 Step {step}/{max_steps} | Loss: {epoch_loss:.4f}")
            
            # Save Checkpoint
            if step % save_every == 0:
                checkpoint_path = os.path.join(
                    system_config.model_save_path, 
                    f"fineweb_checkpoint_step_{step}.pt"
                )
                
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,
                    'config': model_config,
                    'dataset_size': dataset_size
                }, checkpoint_path)
                
                print(f"💾 Checkpoint saved: {checkpoint_path}")
            
            # Memory Cleanup
            if step % 100 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            
            pbar.update(1)
    
    # Final Save
    final_path = os.path.join(system_config.model_save_path, f"fineweb_{dataset_size}_model.pt")
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
        'config': model_config,
        'dataset_size': dataset_size,
        'training_time': time.time() - start_time
    }, final_path)
    
    print(f"\n✅ Training completed!")
    print(f"   Total Steps: {step}")
    print(f"   Final Loss: {epoch_loss:.4f}")
    print(f"   Model saved: {final_path}")
    print(f"   Training Time: {(time.time() - start_time)/3600:.1f} hours")
    print(f"   Dataset: FineWeb-Edu ({dataset_size})")

if __name__ == "__main__":
    print("🎯 FineWeb-Edu LLM Training")
    print("=" * 50)
    
    # Zeige verfügbare Dataset-Größen
    print("📋 Available Dataset Sizes:")
    show_dataset_configs()
    
    # Starte Training mit kleinem Dataset für ersten Test
    print("\n🚀 Starting Training...")
    train_with_fineweb(
        dataset_size="small",  # 10K samples für Test
        max_steps=100,         # Kurzer Test
        save_every=50,
        eval_every=25
    )
