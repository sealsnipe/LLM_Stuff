#!/usr/bin/env python3
"""
 Professional Training Logger
Erstellt professionelle, strukturierte Console-Ausgaben für Training
"""

import torch
from config import model_config, training_config, dataset_config, hardware_config

def print_professional_header():
    """Zeigt professionellen Training-Header."""
    
    # Hardware Info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        cuda_version = torch.version.cuda
        compute_capability = torch.cuda.get_device_properties(0).major + torch.cuda.get_device_properties(0).minor / 10
    else:
        gpu_name = "Not available"
        gpu_memory = 0
        cuda_version = "N/A"
        compute_capability = 0
    
    # Dataset Info
    dataset_info = dataset_config.dataset_sizes.get(dataset_config.default_dataset_size, {})
    num_samples = dataset_info.get('num_samples', 0)
    
    # Model Parameter Count - KORREKTE SwiGLU + GQA Berechnung
    # Token Embeddings (tied)
    token_embeddings = model_config.vocab_size * model_config.hidden_size

    # Per Layer: Attention (GQA) + SwiGLU FFN + LayerNorms
    head_dim = model_config.hidden_size // model_config.num_attention_heads
    attention_per_layer = (
        model_config.hidden_size * (model_config.num_attention_heads * head_dim) +  # q_proj
        model_config.hidden_size * (model_config.num_key_value_heads * head_dim) +  # k_proj
        model_config.hidden_size * (model_config.num_key_value_heads * head_dim) +  # v_proj
        (model_config.num_attention_heads * head_dim) * model_config.hidden_size    # o_proj
    )

    # SwiGLU FFN (3 matrices)
    ffn_per_layer = 3 * model_config.hidden_size * model_config.intermediate_size

    # LayerNorms (2 per layer + 1 final)
    layernorm_per_layer = 2 * model_config.hidden_size
    final_norm = model_config.hidden_size

    total_params = (token_embeddings +
                   model_config.num_layers * (attention_per_layer + ffn_per_layer + layernorm_per_layer) +
                   final_norm)
    
    print(f"  Hardware: {gpu_name} ({gpu_memory:.1f}GB) | CUDA {cuda_version} | Compute {compute_capability}")
    print()
    
    print(" MODEL CONFIGURATION")
    print("┌─────────────────┬──────────────┬─────────────────┬──────────────────┐")
    print("│ Parameters      │ Architecture │ Attention       │ Tokenization     │")
    print("├─────────────────┼──────────────┼─────────────────┼──────────────────┤")
    print(f"│ {total_params//1000000:3d}M ({total_params/1000000000:.2f}B)    │ {model_config.hidden_size:4d}d        │ {model_config.num_attention_heads:2d} heads ({model_config.num_key_value_heads} KV) │ {model_config.vocab_size//1000:2d}k vocab        │")
    print(f"│ {model_config.num_layers:2d} layers       │ {training_config.sequence_length:3d} seq_len  │ RoPE            │ SmolLM tokenizer │")
    print("└─────────────────┴──────────────┴─────────────────┴──────────────────┘")
    print()
    
    print(" TRAINING CONFIGURATION")
    print("┌─────────────────┬──────────────┬─────────────────┬──────────────────┐")
    print("│ Steps & Batches │ Learning     │ Memory          │ Dataset          │")
    print("├─────────────────┼──────────────┼─────────────────┼──────────────────┤")
    total_batches = (num_samples // training_config.batch_size) if num_samples > 0 else 0
    effective_batch = training_config.batch_size * training_config.gradient_accumulation_steps
    optimizer_name = "Muon Hybrid" if training_config.optimizer_type == "muon_hybrid" else "Fused AdamW"
    lr_schedule = training_config.lr_scheduler.title()
    print(f"│ {training_config.max_steps:4d} steps      │ {optimizer_name:12s} │ Micro-batch: {training_config.batch_size:2d}  │ FineWeb-Edu      │")
    print(f"│ {total_batches//1000:2d}k batches     │ {lr_schedule:12s} │ Grad-accum: {training_config.gradient_accumulation_steps:2d}   │ {dataset_config.default_dataset_size.title():8s} ({num_samples//1000:3d}k)    │")
    print("└─────────────────┴──────────────┴─────────────────┴──────────────────┘")
    print()

class InitializationPipeline:
    """Verwaltet in-place Updates der Initialization Pipeline."""

    def __init__(self):
        self.steps = [
            "Model Loading",
            "Dataset Discovery",
            "Dataset Loading",
            "DataLoader Creation"
        ]
        self.status = [" Loading..." for _ in self.steps]
        self.printed = False

    def print_header(self):
        """Zeigt Pipeline Header."""
        print()  # Leerzeile vor Pipeline
        print("INITIALIZATION PIPELINE")  # KEIN EMOJI - kann Probleme verursachen
        for i, step in enumerate(self.steps):
            print(f"├─ {step:20s} {self.status[i]}")
        print()
        self.printed = True
        # Merke: Total 6 Zeilen (Leerzeile + Header + 4 Steps + Leerzeile)

    def update_step(self, step_index: int, success: bool = True, details: str = ""):
        """Aktualisiert einen Schritt mit korrekten ANSI-Codes."""
        if not self.printed:
            return

        if success:
            self.status[step_index] = f"✓ Completed {details}"
        else:
            self.status[step_index] = f" Failed {details}"

        # KORREKTE INLINE-UPDATE LÖSUNG
        import sys
        import os

        # Aktiviere ANSI-Support für Windows
        if os.name == 'nt':
            try:
                import colorama
                colorama.init()
            except ImportError:
                pass

        # KORREKTE Zeilen-Berechnung:
        # Struktur: [Leerzeile] + [Header] + [4 Steps] + [Leerzeile] = 6 Zeilen total
        total_lines = 6

        # Gehe alle Zeilen hoch
        for _ in range(total_lines):
            sys.stdout.write("\033[A")  # Eine Zeile hoch

        # Drucke alles neu (ohne die erste Leerzeile)
        print("INITIALIZATION PIPELINE")
        for i, step in enumerate(self.steps):
            print(f"├─ {step:20s} {self.status[i]}")
        print()  # Leerzeile am Ende

        sys.stdout.flush()

    def complete_all(self):
        """Markiert alle als abgeschlossen."""
        if not self.printed:
            return
        print("└─ All systems ready......... ✓ Initialization complete")
        print()  # Leerzeile nach Pipeline

def print_initialization_pipeline():
    """Zeigt Initialization Pipeline (Legacy-Funktion)."""
    pipeline = InitializationPipeline()
    pipeline.print_header()
    return pipeline

def _get_sequence_packing_status(dataloader_info):
    """Ermittelt den tatsächlichen Sequence Packing Status."""
    from config import training_config

    # Prüfe ob Sequence Packing konfiguriert ist
    if not training_config.use_sequence_packing:
        return {
            'active': False,
            'reason': 'Available but disabled (efficiency)',
            'method': None
        }

    # Prüfe ob DataLoader-Info verfügbar ist
    if not dataloader_info:
        return {
            'active': False,
            'reason': 'Status unknown (no dataloader info)',
            'method': None
        }

    # Prüfe den tatsächlichen DataLoader-Typ
    dataloader_type = dataloader_info.get('type', 'unknown')

    if dataloader_type == 'packed_cache':
        return {
            'active': True,
            'reason': 'Active via Packed Cache',
            'method': 'Packed Cache'
        }
    elif dataloader_type == 'sequence_packed':
        return {
            'active': True,
            'reason': 'Active via Live Packing',
            'method': 'Live Packing'
        }
    elif dataloader_type == 'fallback':
        error_reason = dataloader_info.get('error', 'unknown error')
        return {
            'active': False,
            'reason': f'Failed ({error_reason})',
            'method': None
        }
    else:
        return {
            'active': False,
            'reason': 'Available but not used (fallback active)',
            'method': None
        }

def print_optimization_status(dataloader_info=None):
    """Zeigt detaillierte Optimization-Status."""
    print("⚡ OPTIMIZATIONS")
    
    # Aktive Optimizations
    optimizer_name = "Muon Hybrid" if training_config.optimizer_type == "muon_hybrid" else "Fused AdamW"
    optimizer_type = "(speed + stability)" if training_config.optimizer_type == "muon_hybrid" else "(speed)"
    
    print(f"├─ Optimizer ({optimizer_name})... ✓ Activated {optimizer_type}")
    print(f"├─ torch.compile............. ✓ Activated (speed)")
    
    precision_type = training_config.mixed_precision_dtype.upper() if training_config.use_mixed_precision else "FP32"
    if training_config.use_mixed_precision:
        print(f"├─ Mixed Precision........... ✓ {precision_type} enabled (memory)")
    else:
        print(f"├─ Mixed Precision...........  Disabled")
    
    if training_config.use_activation_checkpointing:
        print(f"├─ Activation Checkpointing.. ✓ Enabled (memory)")
    else:
        print(f"├─ Activation Checkpointing..  Disabled")
    
    print(f"├─ Flash Attention........... ✓ Enabled (speed + memory)")
    print(f"├─ Fused Kernels............. ✓ LayerNorm+Linear (speed)")
    print(f"├─ Optimized Attention....... ✓ GQA enabled (speed)")
    print(f"├─ Gradient Clipping......... ✓ Max norm {training_config.max_grad_norm} (stability)")
    print(f"├─ Zero Grad Optimization.... ✓ set_to_none=True (speed)")
    print(f"├─ Gradient Accumulation..... ✓ {training_config.gradient_accumulation_steps} steps (memory)")
    print(f"├─ Non-blocking Transfer..... ✓ GPU transfer (speed)")
    print(f"├─ DataLoader Optimization... ✓ Pin memory + {training_config.dataloader_num_workers} workers + prefetch {training_config.dataloader_prefetch_factor} (speed)")

    # Neue Optimierungen - Dynamischer Sequence Packing Status
    sequence_packing_status = _get_sequence_packing_status(dataloader_info)
    if sequence_packing_status['active']:
        print(f"├─ Sequence Packing.......... ✓ {sequence_packing_status['method']} (efficiency)")
    if training_config.use_tf32:
        print(f"├─ TF32 Acceleration......... ✓ Enabled (speed)")
    print(f"├─ Weight Decay Groups....... ✓ Enabled (stability)")
    if training_config.use_length_bucketing:
        print(f"├─ Length Bucketing.......... ✓ Enabled (efficiency)")
    if training_config.use_cuda_graphs:
        print(f"├─ CUDA Graphs............... ✓ Enabled (speed)")
    if training_config.use_8bit_adam:
        try:
            import bitsandbytes
            print(f"├─ 8-bit Adam States......... ✓ Enabled (memory)")
        except ImportError:
            print(f"├─ 8-bit Adam States......... ⚠ Fallback to FP32 (bitsandbytes n/a)")
    if training_config.selective_checkpointing != "full":
        print(f"├─ Selective Checkpointing... ✓ {training_config.selective_checkpointing} (memory)")
    if training_config.use_fused_cross_entropy:
        print(f"├─ Fused Cross-Entropy....... ✓ Enabled (speed)")
    if training_config.use_ema:
        print(f"├─ EMA Weights............... ✓ Decay {training_config.ema_decay} (stability)")
    if training_config.production_monitoring:
        print(f"├─ Production Monitoring..... ✓ Advanced metrics (quality)")

    print(f"└─ Memory Reset.............. ✓ Peak stats reset (monitoring)")
    print()
    
    # Verfügbare aber nicht aktive Optimizations
    print("📋 AVAILABLE TRAINING OPTIMIZATIONS")

    # Optimizer Alternativen
    if training_config.optimizer_type != "muon_hybrid":
        print("├─ Muon Optimizer............  Available (speed + stability)")
    if training_config.optimizer_type == "muon_hybrid":
        print("├─ Fused AdamW...............  Alternative available (speed)")

    print("├─ Lion Optimizer............  Not implemented (memory)")
    print("├─ 8-bit Optimizers..........  Not implemented (memory)")

    # Memory Optimizations
    if not training_config.use_activation_checkpointing:
        print("├─ Activation Checkpointing..  Available but disabled (memory)")

    # Nur noch nicht implementierte Features anzeigen
    if not sequence_packing_status['active']:
        print(f"├─ Sequence Packing..........  {sequence_packing_status['reason']}")
    if not training_config.use_cuda_graphs:
        print("├─ CUDA Graphs...............  Available but disabled (speed)")
    if not training_config.use_length_bucketing:
        print("├─ Length Bucketing..........  Available but disabled (efficiency)")
    if not training_config.use_tf32:
        print("├─ TF32 Acceleration.........  Available but disabled (speed)")

    # Advanced Techniques
    print("├─ Quantization (8-bit)......  Not implemented (memory)")
    print("└─ DeepSpeed ZeRO............  Not implemented (multi-gpu)")

def print_training_start(override_steps=None):
    """Zeigt Training Start Banner."""
    from config import training_config

    # Use override steps if provided, otherwise check for dynamic steps
    if override_steps is not None:
        effective_steps = override_steps
    else:
        # Check for dynamic steps override
        try:
            from training_windows import _dynamic_max_steps
            effective_steps = _dynamic_max_steps if _dynamic_max_steps is not None else training_config.max_steps
        except ImportError:
            effective_steps = training_config.max_steps

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_info = f"  GPU: {gpu_name} ({gpu_memory:.1f}GB)"
    else:
        gpu_info = "  GPU: Not available"

    print("=" * 75)
    print(f" TRAINING STARTED | {gpu_info}")
    print(f" Total Steps: {effective_steps:,}")
    print()
    print("📈 TRAINING PROGRESS")
    print("=" * 75)

def print_complete_professional_setup():
    """Zeigt kompletten professionellen Setup-Log."""
    print_professional_header()
    pipeline = print_initialization_pipeline()

    print_optimization_status()
    print_training_start()

    return pipeline

def test_inplace_updates():
    """Testet die In-Place Updates der Pipeline."""
    import time

    print(" Testing In-Place Pipeline Updates")
    print()

    print_professional_header()
    pipeline = InitializationPipeline()
    pipeline.print_header()

    # Simuliere Loading-Schritte
    time.sleep(1)
    pipeline.update_step(0, True, "(1.9GB GPU)")

    time.sleep(1)
    pipeline.update_step(1, True, "(14 files, 30.4GB)")

    time.sleep(1)
    pipeline.update_step(2, True, "(300k samples, 0.1s)")

    time.sleep(1)
    pipeline.update_step(3, True, "(60k batches)")

    time.sleep(0.5)
    pipeline.complete_all()

    print()
    print_optimization_status()
    print_training_start()

if __name__ == "__main__":
    print(" Testing Professional Logger")
    print()
    test_inplace_updates()
