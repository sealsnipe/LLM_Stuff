"""
Core LLM Training Framework

A modular, clean architecture for LLM training that follows clean code principles.
This package provides a complete separation of concerns for:
- Model architecture (models/)
- Training infrastructure (training/)
- Data management (data/)
- Checkpoint management (checkpoints/)
- Monitoring & logging (monitoring/)
- Utilities (utils/)
- High-level interfaces (interfaces/)

Usage:
    from core.interfaces import TrainingInterface, ModelInterface
    
    # High-level training
    trainer = TrainingInterface()
    trainer.start_training()
    
    # Model management
    model_manager = ModelInterface()
    model = model_manager.create_model()
"""

__version__ = "1.0.0"
__author__ = "LLM Training Framework"

# Import high-level interfaces for easy access
from .interfaces.training_interface import TrainingInterface
from .interfaces.model_interface import ModelInterface

# Import core components
from .models import MemoryOptimizedLLM
from .training import Trainer
from .data import DatasetFactory
from .checkpoints import CheckpointManager
from .monitoring import ProgressDisplay, MemoryMonitor
from .utils import GPUUtils, SystemUtils

__all__ = [
    # High-level interfaces
    'TrainingInterface',
    'ModelInterface',
    
    # Core components
    'MemoryOptimizedLLM',
    'Trainer',
    'DatasetFactory',
    'CheckpointManager',
    'ProgressDisplay',
    'MemoryMonitor',
    'GPUUtils',
    'SystemUtils',
]
