"""
GPU Utilities Module

Contains GPU-related utility functions and the GPUUtils class.
Handles GPU detection, setup, optimization, and compatibility checks.
"""

import torch
import os
from typing import Dict, List, Optional


class GPUUtils:
    """Utility-Klasse für GPU-Management und -Optimierung."""
    
    def __init__(self):
        self.gpu_info = self._detect_gpus()
    
    def _detect_gpus(self) -> Dict:
        """Erkennt verfügbare GPUs und sammelt Informationen."""
        if not torch.cuda.is_available():
            return {'available': False, 'count': 0, 'devices': []}
        
        gpu_count = torch.cuda.device_count()
        devices = []
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            device_info = {
                'id': i,
                'name': props.name,
                'memory_gb': props.total_memory / 1e9,
                'compute_capability': f"{props.major}.{props.minor}",
                'multiprocessor_count': props.multi_processor_count,
                'max_threads_per_block': getattr(props, 'max_threads_per_block', 1024),
                'max_shared_memory': getattr(props, 'max_shared_memory_per_block', 49152)
            }
            devices.append(device_info)
        
        return {
            'available': True,
            'count': gpu_count,
            'devices': devices,
            'cuda_version': torch.version.cuda
        }
    
    def check_gpu_setup(self) -> bool:
        """Überprüft GPU-Setup und gibt Empfehlungen."""
        if not self.gpu_info['available']:
            return False

        return True
    
    def optimize_gpu_settings(self):
        """Optimiert GPU-Einstellungen für Training."""
        if not self.gpu_info['available']:
            return

        # TF32 für RTX 3090/4090
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        # CUDA Environment Variables mit Windows Triton Fix
        optimizations = {
            "PYTORCH_SDPA_ENABLE_BACKEND": "flash",  # Force Flash Attention
            "TRITON_CACHE_DIR": os.path.abspath("./.triton_cache"),   # Windows-kompatible absolute Pfade
            "CUDA_LAUNCH_BLOCKING": "0",             # Async CUDA calls
            "TORCH_CUDNN_V8_API_ENABLED": "1",      # CuDNN v8 API
            "TRITON_INTERPRET": "1",                # Windows Triton Fallback
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"  # Memory fragmentation fix
        }

        for key, value in optimizations.items():
            os.environ.setdefault(key, value)

        # Erstelle Triton Cache Directory für Windows
        triton_cache_dir = os.path.abspath("./.triton_cache")
        os.makedirs(triton_cache_dir, exist_ok=True)
    
    def get_optimal_batch_size(self, model_size_gb: float, sequence_length: int = 512) -> int:
        """Schätzt optimale Batch-Größe basierend auf GPU Memory."""
        if not self.gpu_info['available']:
            return 1
        
        # Verwende erste GPU für Schätzung
        gpu_memory_gb = self.gpu_info['devices'][0]['memory_gb']
        
        # Reserviere 20% für System und Overhead
        available_memory = gpu_memory_gb * 0.8
        
        # Schätze Memory pro Sample (sehr grob)
        # Model + Gradients + Optimizer States ≈ 3x Model Size
        # Plus Activations ≈ batch_size * sequence_length * hidden_size * layers * 4 bytes
        
        model_overhead = model_size_gb * 3  # Model + Gradients + Optimizer
        remaining_memory = available_memory - model_overhead
        
        if remaining_memory <= 0:
            return 1
        
        # Schätze Memory pro Sample (sehr konservativ)
        memory_per_sample = 0.1  # 100MB pro Sample (konservativ)
        
        estimated_batch_size = int(remaining_memory / memory_per_sample)
        
        # Sicherheitsgrenzen
        return max(1, min(estimated_batch_size, 32))
    
    def monitor_gpu_usage(self) -> Dict:
        """Überwacht aktuelle GPU-Nutzung."""
        if not torch.cuda.is_available():
            return {'available': False}
        
        usage_info = []
        
        for i in range(torch.cuda.device_count()):
            # Memory Info
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            
            utilization = (allocated / total) * 100
            
            device_usage = {
                'device_id': i,
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'total_gb': total,
                'utilization_percent': utilization,
                'free_gb': total - reserved
            }
            
            usage_info.append(device_usage)
        
        return {
            'available': True,
            'devices': usage_info,
            'timestamp': torch.cuda.Event(enable_timing=True)
        }
    
    def cleanup_gpu_memory(self):
        """Bereinigt GPU Memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("🧹 GPU Memory bereinigt")
    
    def get_gpu_recommendations(self) -> List[str]:
        """Gibt GPU-spezifische Empfehlungen."""
        recommendations = []
        
        if not self.gpu_info['available']:
            recommendations.append("Installiere CUDA-kompatible PyTorch Version")
            return recommendations
        
        for device in self.gpu_info['devices']:
            memory_gb = device['memory_gb']
            compute_cap = device['compute_capability']
            
            # Memory-basierte Empfehlungen
            if memory_gb < 8:
                recommendations.append(f"GPU {device['id']}: Verwende kleine Batch Sizes (1-4)")
                recommendations.append(f"GPU {device['id']}: Aktiviere Gradient Checkpointing")
            elif memory_gb < 16:
                recommendations.append(f"GPU {device['id']}: Moderate Batch Sizes (4-8) empfohlen")
            else:
                recommendations.append(f"GPU {device['id']}: Große Batch Sizes (8-16) möglich")
            
            # Compute Capability Empfehlungen
            major, minor = map(int, compute_cap.split('.'))
            if major >= 8:  # RTX 30xx/40xx
                recommendations.append(f"GPU {device['id']}: TF32 und Flash Attention verfügbar")
            elif major >= 7:  # RTX 20xx
                recommendations.append(f"GPU {device['id']}: Mixed Precision empfohlen")
            else:
                recommendations.append(f"GPU {device['id']}: Ältere GPU - reduzierte Performance erwartet")
        
        return recommendations


def check_gpu_setup() -> bool:
    """Convenience function für GPU Setup Check."""
    gpu_utils = GPUUtils()
    return gpu_utils.check_gpu_setup()


def optimize_gpu_for_training():
    """Convenience function für GPU Optimierung."""
    gpu_utils = GPUUtils()
    gpu_utils.optimize_gpu_settings()


def get_gpu_info() -> Dict:
    """Convenience function für GPU Info."""
    gpu_utils = GPUUtils()
    return gpu_utils.gpu_info


def estimate_optimal_batch_size(model_size_gb: float) -> int:
    """Convenience function für Batch Size Schätzung."""
    gpu_utils = GPUUtils()
    return gpu_utils.get_optimal_batch_size(model_size_gb)
