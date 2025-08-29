"""
Training Infrastructure Module

Contains all training-related components:
- Main trainer (trainer.py)
- Optimizers (optimizers.py)
- Learning rate schedulers (schedulers.py)
- Training metrics (metrics.py)

Provides a complete training pipeline with memory optimization and monitoring.
"""

from .trainer import Trainer
from .optimizers import CPUOffloadOptimizer, create_optimizer
from .schedulers import create_lr_scheduler
from .metrics import AdvancedMetrics

__all__ = [
    'Trainer',
    'CPUOffloadOptimizer',
    'create_optimizer',
    'create_lr_scheduler',
    'AdvancedMetrics',
]
