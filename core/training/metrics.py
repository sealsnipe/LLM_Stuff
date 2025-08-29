"""
Training Metrics Module

Contains the AdvancedMetrics class for comprehensive training monitoring.
Tracks performance metrics, gradient statistics, and training health indicators.
"""

import numpy as np


class AdvancedMetrics:
    """Advanced Training Metrics für Production-Ready Monitoring."""

    def __init__(self, window_size=100):
        self.window_size = window_size
        self.step_times = []
        self.tokens_per_sec = []
        self.grad_norms = []
        self.clip_count = 0
        self.total_steps = 0
        self.flash_hit_count = 0
        self.nan_count = 0

    def update(self, step_time, tokens_per_sec, grad_norm, was_clipped, flash_active, has_nan):
        """Update Metriken."""
        self.step_times.append(step_time)
        self.tokens_per_sec.append(tokens_per_sec)
        self.grad_norms.append(grad_norm)

        if was_clipped:
            self.clip_count += 1
        if flash_active:
            self.flash_hit_count += 1
        if has_nan:
            self.nan_count += 1

        self.total_steps += 1

        # Sliding window
        if len(self.step_times) > self.window_size:
            self.step_times.pop(0)
            self.tokens_per_sec.pop(0)
            self.grad_norms.pop(0)

    def get_stats(self):
        """Berechne Statistiken."""
        if not self.step_times:
            return {}

        return {
            'step_time_mean': np.mean(self.step_times),
            'step_time_p90': np.percentile(self.step_times, 90),
            'tokens_per_sec_mean': np.mean(self.tokens_per_sec),
            'tokens_per_sec_p90': np.percentile(self.tokens_per_sec, 90),
            'grad_norm_mean': np.mean(self.grad_norms),
            'clip_rate': self.clip_count / max(self.total_steps, 1),
            'flash_hit_rate': self.flash_hit_count / max(self.total_steps, 1),
            'nan_rate': self.nan_count / max(self.total_steps, 1)
        }

    def reset(self):
        """Reset alle Metriken."""
        self.step_times.clear()
        self.tokens_per_sec.clear()
        self.grad_norms.clear()
        self.clip_count = 0
        self.total_steps = 0
        self.flash_hit_count = 0
        self.nan_count = 0
