"""
High-Level Interfaces Module

Contains clean, high-level APIs for the training framework:
- Training interface (training_interface.py)
- Model interface (model_interface.py)

These interfaces provide simple, clean APIs that hide the complexity of the underlying components.
They follow the Facade pattern to provide a unified interface to the subsystems.
"""

from .training_interface import TrainingInterface
from .model_interface import ModelInterface

__all__ = [
    'TrainingInterface',
    'ModelInterface',
]
