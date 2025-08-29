"""
Memory Monitor Module

Contains the MemoryMonitor class for tracking GPU and CPU memory usage.
Provides memory statistics, cleanup functions, and memory optimization.
"""

import torch
import gc
import psutil
from typing import Dict


class MemoryMonitor:
    """Memory-Monitor für GPU und CPU."""

    def __init__(self):
        self.peak_gpu_memory = 0
        self.peak_cpu_memory = 0
        self.cleanup_counter = 0

    def get_memory_stats(self) -> Dict:
        """Aktuelle Memory-Statistiken."""
        # GPU Memory
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / 1e9
            gpu_reserved = torch.cuda.memory_reserved() / 1e9
            gpu_max = torch.cuda.max_memory_allocated() / 1e9
            self.peak_gpu_memory = max(self.peak_gpu_memory, gpu_allocated)
        else:
            gpu_allocated = gpu_reserved = gpu_max = 0

        # CPU Memory
        cpu_memory = psutil.virtual_memory()
        cpu_used = cpu_memory.used / 1e9
        cpu_percent = cpu_memory.percent
        self.peak_cpu_memory = max(self.peak_cpu_memory, cpu_used)

        return {
            'gpu_allocated': gpu_allocated,
            'gpu_reserved': gpu_reserved,
            'gpu_max': gpu_max,
            'gpu_peak': self.peak_gpu_memory,
            'cpu_used': cpu_used,
            'cpu_percent': cpu_percent,
            'cpu_peak': self.peak_cpu_memory
        }

    def cleanup_memory(self):
        """Aggressive Memory Cleanup."""
        self.cleanup_counter += 1
        
        # GPU Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # CPU Cleanup
        gc.collect()

    def print_memory_stats(self, prefix: str = ""):
        """Drucke Memory-Statistiken."""
        stats = self.get_memory_stats()
        print(f"{prefix}GPU: {stats['gpu_allocated']:.1f}GB allocated, {stats['gpu_peak']:.1f}GB peak")
        print(f"{prefix}CPU: {stats['cpu_used']:.1f}GB used ({stats['cpu_percent']:.1f}%)")

    def get_gpu_utilization(self) -> float:
        """Berechnet GPU Memory Utilization als Prozentsatz."""
        if not torch.cuda.is_available():
            return 0.0
        
        allocated = torch.cuda.memory_allocated()
        total = torch.cuda.get_device_properties(0).total_memory
        
        return (allocated / total) * 100

    def get_memory_efficiency_report(self) -> Dict:
        """Erstellt detaillierten Memory-Effizienz-Report."""
        stats = self.get_memory_stats()
        
        # GPU Effizienz
        gpu_utilization = self.get_gpu_utilization()
        gpu_fragmentation = stats['gpu_reserved'] - stats['gpu_allocated']
        
        # CPU Effizienz
        cpu_available = psutil.virtual_memory().available / 1e9
        
        return {
            'gpu': {
                'utilization_percent': gpu_utilization,
                'allocated_gb': stats['gpu_allocated'],
                'reserved_gb': stats['gpu_reserved'],
                'fragmentation_gb': gpu_fragmentation,
                'peak_gb': stats['gpu_peak'],
                'efficiency_rating': self._calculate_gpu_efficiency(gpu_utilization, gpu_fragmentation)
            },
            'cpu': {
                'used_gb': stats['cpu_used'],
                'used_percent': stats['cpu_percent'],
                'available_gb': cpu_available,
                'peak_gb': stats['cpu_peak'],
                'efficiency_rating': self._calculate_cpu_efficiency(stats['cpu_percent'])
            },
            'cleanup_count': self.cleanup_counter
        }

    def _calculate_gpu_efficiency(self, utilization: float, fragmentation: float) -> str:
        """Berechnet GPU Memory Effizienz-Rating."""
        if utilization > 90:
            return "Excellent" if fragmentation < 1.0 else "Good"
        elif utilization > 70:
            return "Good" if fragmentation < 2.0 else "Fair"
        elif utilization > 50:
            return "Fair"
        else:
            return "Poor"

    def _calculate_cpu_efficiency(self, cpu_percent: float) -> str:
        """Berechnet CPU Memory Effizienz-Rating."""
        if cpu_percent < 60:
            return "Excellent"
        elif cpu_percent < 75:
            return "Good"
        elif cpu_percent < 85:
            return "Fair"
        else:
            return "Poor"

    def monitor_memory_trend(self, window_size: int = 10):
        """Überwacht Memory-Trend über Zeit."""
        if not hasattr(self, '_memory_history'):
            self._memory_history = []
        
        current_stats = self.get_memory_stats()
        self._memory_history.append({
            'gpu_allocated': current_stats['gpu_allocated'],
            'cpu_percent': current_stats['cpu_percent']
        })
        
        # Sliding window
        if len(self._memory_history) > window_size:
            self._memory_history.pop(0)
        
        # Berechne Trend
        if len(self._memory_history) >= 2:
            recent = self._memory_history[-3:]  # Letzte 3 Messungen
            gpu_trend = sum(m['gpu_allocated'] for m in recent) / len(recent)
            cpu_trend = sum(m['cpu_percent'] for m in recent) / len(recent)
            
            return {
                'gpu_trend_gb': gpu_trend,
                'cpu_trend_percent': cpu_trend,
                'samples': len(self._memory_history)
            }
        
        return None

    def check_memory_pressure(self) -> Dict:
        """Prüft auf Memory-Druck und gibt Empfehlungen."""
        stats = self.get_memory_stats()
        warnings = []
        recommendations = []
        
        # GPU Memory Pressure
        gpu_util = self.get_gpu_utilization()
        if gpu_util > 95:
            warnings.append("Kritischer GPU Memory Druck")
            recommendations.append("Reduziere Batch Size oder aktiviere Gradient Checkpointing")
        elif gpu_util > 85:
            warnings.append("Hoher GPU Memory Druck")
            recommendations.append("Überwache GPU Memory genau")
        
        # CPU Memory Pressure
        if stats['cpu_percent'] > 90:
            warnings.append("Kritischer CPU Memory Druck")
            recommendations.append("Schließe andere Anwendungen oder reduziere DataLoader Workers")
        elif stats['cpu_percent'] > 80:
            warnings.append("Hoher CPU Memory Druck")
            recommendations.append("Überwache CPU Memory genau")
        
        # GPU Fragmentation
        fragmentation = stats['gpu_reserved'] - stats['gpu_allocated']
        if fragmentation > 2.0:
            warnings.append("Hohe GPU Memory Fragmentation")
            recommendations.append("Führe torch.cuda.empty_cache() aus")
        
        return {
            'pressure_level': 'Critical' if any('Kritisch' in w for w in warnings) else 
                             'High' if warnings else 'Normal',
            'warnings': warnings,
            'recommendations': recommendations,
            'stats': stats
        }

    def auto_cleanup_if_needed(self, threshold_percent: float = 85):
        """Automatische Memory-Bereinigung bei Bedarf."""
        gpu_util = self.get_gpu_utilization()
        cpu_percent = self.get_memory_stats()['cpu_percent']
        
        if gpu_util > threshold_percent or cpu_percent > threshold_percent:
            print(f"🧹 Auto-Cleanup: GPU {gpu_util:.1f}%, CPU {cpu_percent:.1f}%")
            self.cleanup_memory()
            return True
        
        return False

    def reset_peak_tracking(self):
        """Setzt Peak Memory Tracking zurück."""
        self.peak_gpu_memory = 0
        self.peak_cpu_memory = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        print("✅ Peak Memory Tracking zurückgesetzt")
