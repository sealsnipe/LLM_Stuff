"""
Performance Tracker Module

Contains the PerformanceTracker class for comprehensive performance monitoring.
Tracks training speed, efficiency metrics, and performance optimization insights.
"""

import time
import torch
from typing import Dict, List, Optional
from collections import deque
import numpy as np


class PerformanceTracker:
    """Advanced Performance Tracking für Training Optimization."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        
        # Performance metrics
        self.step_times = deque(maxlen=window_size)
        self.throughput_history = deque(maxlen=window_size)
        self.gpu_utilization_history = deque(maxlen=window_size)
        self.loss_history = deque(maxlen=window_size)
        
        # Timing tracking
        self.step_start_time = None
        self.total_training_time = 0
        self.total_steps = 0
        
        # Performance baselines
        self.best_throughput = 0
        self.best_step_time = float('inf')
        
        # Flash attention tracking
        self.flash_attention_hits = 0
        self.flash_attention_total = 0
    
    def start_step_timing(self):
        """Startet Zeitmessung für einen Trainingsschritt."""
        self.step_start_time = time.time()
    
    def end_step_timing(self, batch_size: int, sequence_length: int, loss: float):
        """Beendet Zeitmessung und aktualisiert Metriken."""
        if self.step_start_time is None:
            return
        
        step_time = time.time() - self.step_start_time
        tokens_processed = batch_size * sequence_length
        throughput = tokens_processed / step_time if step_time > 0 else 0
        
        # Update histories
        self.step_times.append(step_time)
        self.throughput_history.append(throughput)
        self.loss_history.append(loss)
        
        # Update totals
        self.total_training_time += step_time
        self.total_steps += 1
        
        # Update bests
        if throughput > self.best_throughput:
            self.best_throughput = throughput
        if step_time < self.best_step_time:
            self.best_step_time = step_time
        
        # GPU utilization (if available)
        if torch.cuda.is_available():
            gpu_util = (torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory) * 100
            self.gpu_utilization_history.append(gpu_util)
        
        self.step_start_time = None
    
    def track_flash_attention(self, used_flash: bool):
        """Trackt Flash Attention Nutzung."""
        self.flash_attention_total += 1
        if used_flash:
            self.flash_attention_hits += 1
    
    def get_current_performance(self) -> Dict:
        """Gibt aktuelle Performance-Metriken zurück."""
        if not self.step_times:
            return {}
        
        recent_steps = list(self.step_times)[-10:]  # Letzte 10 Steps
        recent_throughput = list(self.throughput_history)[-10:]
        
        return {
            'current_step_time': recent_steps[-1] if recent_steps else 0,
            'avg_step_time_10': np.mean(recent_steps) if recent_steps else 0,
            'current_throughput': recent_throughput[-1] if recent_throughput else 0,
            'avg_throughput_10': np.mean(recent_throughput) if recent_throughput else 0,
            'best_throughput': self.best_throughput,
            'best_step_time': self.best_step_time,
            'total_steps': self.total_steps
        }
    
    def get_performance_summary(self) -> Dict:
        """Erstellt umfassende Performance-Zusammenfassung."""
        if not self.step_times:
            return {'status': 'No data available'}
        
        step_times = list(self.step_times)
        throughput = list(self.throughput_history)
        gpu_util = list(self.gpu_utilization_history)
        losses = list(self.loss_history)
        
        # Berechne Statistiken
        summary = {
            'timing': {
                'avg_step_time': np.mean(step_times),
                'median_step_time': np.median(step_times),
                'p95_step_time': np.percentile(step_times, 95),
                'min_step_time': np.min(step_times),
                'max_step_time': np.max(step_times),
                'step_time_std': np.std(step_times),
                'total_training_time': self.total_training_time
            },
            'throughput': {
                'avg_tokens_per_sec': np.mean(throughput),
                'median_tokens_per_sec': np.median(throughput),
                'p95_tokens_per_sec': np.percentile(throughput, 95),
                'max_tokens_per_sec': np.max(throughput),
                'throughput_std': np.std(throughput)
            },
            'efficiency': {
                'flash_attention_rate': (self.flash_attention_hits / max(self.flash_attention_total, 1)) * 100,
                'performance_consistency': 1.0 - (np.std(step_times) / np.mean(step_times)) if step_times else 0,
                'throughput_efficiency': (np.mean(throughput) / self.best_throughput) * 100 if self.best_throughput > 0 else 0
            }
        }
        
        # GPU Statistiken falls verfügbar
        if gpu_util:
            summary['gpu'] = {
                'avg_utilization': np.mean(gpu_util),
                'max_utilization': np.max(gpu_util),
                'min_utilization': np.min(gpu_util),
                'utilization_std': np.std(gpu_util)
            }
        
        # Loss Trend
        if len(losses) > 1:
            loss_trend = np.polyfit(range(len(losses)), losses, 1)[0]  # Linear trend
            summary['training'] = {
                'loss_trend': loss_trend,
                'current_loss': losses[-1],
                'best_loss': np.min(losses),
                'loss_std': np.std(losses)
            }
        
        return summary
    
    def get_optimization_recommendations(self) -> List[str]:
        """Gibt Performance-Optimierungs-Empfehlungen."""
        recommendations = []
        
        if not self.step_times:
            return ["Nicht genügend Daten für Empfehlungen"]
        
        perf = self.get_performance_summary()
        
        # Flash Attention Empfehlungen
        flash_rate = perf['efficiency']['flash_attention_rate']
        if flash_rate < 90:
            recommendations.append(f"Flash Attention Rate niedrig ({flash_rate:.1f}%) - prüfe Attention Masks")
        
        # Throughput Empfehlungen
        throughput_eff = perf['efficiency']['throughput_efficiency']
        if throughput_eff < 80:
            recommendations.append(f"Throughput Effizienz niedrig ({throughput_eff:.1f}%) - prüfe Batch Size")
        
        # GPU Utilization Empfehlungen
        if 'gpu' in perf:
            gpu_util = perf['gpu']['avg_utilization']
            if gpu_util < 70:
                recommendations.append(f"GPU Utilization niedrig ({gpu_util:.1f}%) - erhöhe Batch Size")
            elif gpu_util > 95:
                recommendations.append(f"GPU Utilization sehr hoch ({gpu_util:.1f}%) - reduziere Batch Size")
        
        # Consistency Empfehlungen
        consistency = perf['efficiency']['performance_consistency']
        if consistency < 0.8:
            recommendations.append(f"Performance inkonsistent ({consistency:.2f}) - prüfe System Load")
        
        # Step Time Empfehlungen
        step_times = list(self.step_times)
        if len(step_times) > 10:
            recent_avg = np.mean(step_times[-10:])
            overall_avg = np.mean(step_times)
            if recent_avg > overall_avg * 1.2:
                recommendations.append("Performance verschlechtert sich - prüfe Memory Fragmentation")
        
        if not recommendations:
            recommendations.append("Performance ist optimal! 🚀")
        
        return recommendations
    
    def detect_performance_anomalies(self) -> Dict:
        """Erkennt Performance-Anomalien."""
        if len(self.step_times) < 20:
            return {'anomalies': [], 'status': 'Insufficient data'}
        
        step_times = list(self.step_times)
        throughput = list(self.throughput_history)
        
        anomalies = []
        
        # Outlier Detection für Step Times
        step_mean = np.mean(step_times)
        step_std = np.std(step_times)
        threshold = step_mean + 3 * step_std
        
        outliers = [i for i, t in enumerate(step_times) if t > threshold]
        if outliers:
            anomalies.append(f"Step Time Outliers detected: {len(outliers)} steps")
        
        # Sudden Performance Drops
        if len(step_times) > 10:
            recent_avg = np.mean(step_times[-5:])
            baseline_avg = np.mean(step_times[-20:-5])
            
            if recent_avg > baseline_avg * 1.5:
                anomalies.append("Sudden performance drop detected")
        
        # Throughput Anomalies
        if len(throughput) > 10:
            throughput_mean = np.mean(throughput)
            throughput_std = np.std(throughput)
            recent_throughput = np.mean(throughput[-5:])
            
            if recent_throughput < throughput_mean - 2 * throughput_std:
                anomalies.append("Throughput significantly below average")
        
        return {
            'anomalies': anomalies,
            'status': 'Normal' if not anomalies else 'Anomalies detected',
            'outlier_count': len(outliers) if 'outliers' in locals() else 0
        }
    
    def export_performance_data(self) -> Dict:
        """Exportiert alle Performance-Daten für Analyse."""
        return {
            'step_times': list(self.step_times),
            'throughput_history': list(self.throughput_history),
            'gpu_utilization_history': list(self.gpu_utilization_history),
            'loss_history': list(self.loss_history),
            'metadata': {
                'total_steps': self.total_steps,
                'total_training_time': self.total_training_time,
                'best_throughput': self.best_throughput,
                'best_step_time': self.best_step_time,
                'flash_attention_rate': (self.flash_attention_hits / max(self.flash_attention_total, 1)) * 100
            }
        }
    
    def reset_tracking(self):
        """Setzt alle Tracking-Daten zurück."""
        self.step_times.clear()
        self.throughput_history.clear()
        self.gpu_utilization_history.clear()
        self.loss_history.clear()
        
        self.total_training_time = 0
        self.total_steps = 0
        self.best_throughput = 0
        self.best_step_time = float('inf')
        self.flash_attention_hits = 0
        self.flash_attention_total = 0
