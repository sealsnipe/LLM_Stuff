"""
Data Management Module

Contains all data-related components:
- Dataset factory (dataset_factory.py)
- Data utilities (data_utils.py)
- Cache management (cache_manager.py)

Provides efficient data loading, preprocessing, and caching for training.
"""

from .dataset_factory import DatasetFactory, create_gpu_optimized_dataset
from .data_utils import create_packed_sequences, create_packed_attention_masks
from .cache_manager import CacheManager

__all__ = [
    'DatasetFactory',
    'create_gpu_optimized_dataset',
    'create_packed_sequences',
    'create_packed_attention_masks',
    'CacheManager',
]
