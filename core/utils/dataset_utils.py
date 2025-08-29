"""
Dataset Utilities for Epoch-based Training

Provides utilities for calculating dataset sizes and epoch-based training parameters.
"""

import os
import json
import glob
from typing import Optional, Tuple
from config import training_config


class DatasetSizeCalculator:
    """Calculate dataset sizes for epoch-based training."""

    def __init__(self, cache_info=None):
        self.cached_dataset_size = None
        self.cached_dataset_tokens = None
        self.cache_info = cache_info
    
    def get_packed_cache_size(self, cache_path: str = None) -> Optional[Tuple[int, int]]:
        """
        Get size of packed cache dataset.

        Args:
            cache_path: Specific cache path, or None for auto-detection

        Returns:
            Tuple of (total_samples, total_tokens) or None if not available
        """
        if cache_path is None:
            # Auto-detect cache from registry
            from .cache_registry import get_default_cache
            cache_info = get_default_cache()
            if cache_info:
                cache_dir = cache_info['path']
            else:
                cache_dir = "cache/packed_sequences"
        else:
            cache_dir = cache_path

        metadata_file = os.path.join(cache_dir, "cache_metadata.json")
        
        if not os.path.exists(metadata_file):
            return None
        
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # FIXED: Calculate from actual chunk files if metadata is incomplete
            total_sequences = metadata.get('total_sequences', 0)
            cache_max_length = metadata.get('max_length', 512)

            if total_sequences == 0:
                # Metadata incomplete - calculate from chunk files
                total_sequences = self._count_sequences_from_chunks(cache_dir)
                print(f"Warning: Incomplete metadata, counted {total_sequences:,} sequences from chunks")

            # Calculate total tokens using the ACTUAL cache sequence length
            utilization = metadata.get('utilization_stats', {}).get('avg_utilization', 0.99)
            total_tokens = int(total_sequences * cache_max_length * utilization)

            self.cached_dataset_size = total_sequences
            self.cached_dataset_tokens = total_tokens

            return total_sequences, total_tokens

        except Exception as e:
            print(f"Warning: Could not read packed cache metadata: {e}")
            return None

    def _count_sequences_from_chunks(self, cache_dir: str) -> int:
        """Count total sequences by loading chunk files (skips LZ4 compressed)."""
        import torch

        chunk_files = glob.glob(os.path.join(cache_dir, "packed_chunk_*.pt"))
        total_sequences = 0
        lz4_chunks = 0

        for chunk_file in chunk_files:
            try:
                # Try to load chunk (will fail for LZ4 compressed)
                chunk_data = torch.load(chunk_file, map_location='cpu', weights_only=False)
                if 'input_ids' in chunk_data:
                    total_sequences += chunk_data['input_ids'].shape[0]
                elif 'metadata' in chunk_data and 'num_sequences' in chunk_data['metadata']:
                    total_sequences += chunk_data['metadata']['num_sequences']
            except Exception as e:
                # Check if it's LZ4 compression error
                if "invalid load key" in str(e) and "\\x04" in str(e):
                    lz4_chunks += 1
                    continue  # Skip LZ4 compressed chunks silently
                else:
                    print(f"Warning: Could not read chunk {chunk_file}: {e}")
                    continue

        if lz4_chunks > 0:
            print(f"Info: Skipped {lz4_chunks} LZ4-compressed chunks (use metadata instead)")

        return total_sequences

    def get_fineweb_size(self) -> Optional[Tuple[int, int]]:
        """
        Estimate FineWeb dataset size.
        
        Returns:
            Tuple of (total_samples, total_tokens) or None if not available
        """
        # Check for FineWeb cache
        fineweb_dir = "cache/fineweb"
        if not os.path.exists(fineweb_dir):
            return None
        
        # Count parquet files and estimate
        parquet_files = glob.glob(os.path.join(fineweb_dir, "*.parquet"))
        if not parquet_files:
            return None
        
        # Conservative estimates based on FineWeb-Edu
        estimated_samples_per_file = 8000  # Conservative estimate
        estimated_tokens_per_sample = 512  # Average tokens per sample
        
        total_samples = len(parquet_files) * estimated_samples_per_file
        total_tokens = total_samples * estimated_tokens_per_sample
        
        return total_samples, total_tokens
    
    def get_dataset_size(self) -> Tuple[int, int]:
        """
        Get current dataset size (samples and tokens).

        Returns:
            Tuple of (total_samples, total_tokens)
        """
        # FIXED: Use cache_info if available (for accurate current values)
        if self.cache_info and 'total_sequences' in self.cache_info:
            total_sequences = self.cache_info['total_sequences']
            sequence_length = self.cache_info.get('sequence_length', training_config.sequence_length)
            total_tokens = total_sequences * sequence_length
            return total_sequences, total_tokens

        # Try packed cache first
        packed_size = self.get_packed_cache_size()
        if packed_size:
            return packed_size

        # Try FineWeb
        fineweb_size = self.get_fineweb_size()
        if fineweb_size:
            return fineweb_size

        # Fallback to synthetic dataset estimate
        return 10000, 10000 * training_config.sequence_length
    
    def calculate_epoch_based_steps(self, target_epochs: int) -> Tuple[int, int]:
        """
        Calculate max_steps for epoch-based training.

        Args:
            target_epochs: Number of epochs to train

        Returns:
            Tuple of (max_steps, total_tokens)
        """
        total_samples, dataset_tokens = self.get_dataset_size()

        # FIXED: Use epoch_dataset_fraction for partial dataset per epoch
        epoch_fraction = training_config.epoch_dataset_fraction
        tokens_per_epoch = int(dataset_tokens * epoch_fraction)
        total_training_tokens = tokens_per_epoch * target_epochs

        # Calculate steps needed - FIXED: Use actual sequence length from cache
        effective_batch_size = training_config.batch_size * training_config.gradient_accumulation_steps

        # Use sequence length from cache info if available, otherwise config
        if self.cache_info and 'sequence_length' in self.cache_info:
            actual_sequence_length = self.cache_info['sequence_length']
        else:
            actual_sequence_length = training_config.sequence_length

        tokens_per_step = effective_batch_size * actual_sequence_length
        max_steps = total_training_tokens // tokens_per_step

        return max_steps, total_training_tokens
    
    def calculate_dynamic_warmup_steps(self, max_steps: int, warmup_ratio: float = 0.025) -> int:
        """
        Calculate dynamic warmup steps based on FULL dataset size, not current cache.

        This ensures consistent warmup regardless of current cache size and prepares
        for the complete dataset that will eventually be used.

        Args:
            max_steps: Current training steps (based on current cache)
            warmup_ratio: Ratio of warmup steps to total steps (default: 2.5%)

        Returns:
            Number of warmup steps based on full dataset estimate
        """
        # Estimate full FineWeb dataset size
        full_dataset_sequences = self._estimate_full_fineweb_size()

        # Calculate steps for full dataset
        effective_batch_size = training_config.batch_size * training_config.gradient_accumulation_steps
        sequence_length = self.cache_info.get('sequence_length', 512) if self.cache_info else 512
        tokens_per_step = effective_batch_size * sequence_length

        # Full dataset training calculation
        epoch_fraction = training_config.epoch_dataset_fraction
        tokens_per_epoch = full_dataset_sequences * sequence_length * epoch_fraction
        total_training_tokens = tokens_per_epoch * training_config.target_epochs
        full_dataset_steps = total_training_tokens // tokens_per_step

        # Calculate warmup based on full dataset
        warmup_steps = int(full_dataset_steps * warmup_ratio)

        # Ensure reasonable bounds
        warmup_steps = max(100, warmup_steps)  # Minimum 100 steps
        warmup_steps = min(10000, warmup_steps)  # Maximum 10000 steps (increased for large datasets)

        # Warning if warmup exceeds current training steps
        if warmup_steps > max_steps:
            print(f"⚠️  WARNING: Warmup steps ({warmup_steps:,}) exceed current training steps ({max_steps:,})")
            print(f"   This happens when training with partial cache. Consider expanding cache first.")
            print(f"   Full dataset estimate: {full_dataset_sequences:,} sequences")
            print(f"   Current dataset: {self.cache_info.get('total_sequences', 'unknown') if self.cache_info else 'unknown'} sequences")

            # For very small datasets (like FineWeb-Edu), cap warmup at reasonable fraction
            current_sequences = self.cache_info.get('total_sequences', 0) if self.cache_info else 0
            if current_sequences < 10_000:  # Very small dataset
                print(f"   Using reduced warmup for small dataset")
                warmup_steps = max(10, int(max_steps * 0.1))  # 10% warmup for small datasets

        return warmup_steps

    def _estimate_full_fineweb_size(self) -> int:
        """
        Estimate the full FineWeb dataset size based on known information.

        FineWeb-Edu contains approximately 1.3 trillion tokens.
        With 512 sequence length, this equals roughly 2.5M sequences.

        Returns:
            Estimated number of sequences in full FineWeb dataset
        """
        # Conservative estimate based on FineWeb-Edu size
        # 1.3T tokens / 512 tokens per sequence ≈ 2.5M sequences
        estimated_full_sequences = 2_500_000

        # If we have current cache info, we can refine the estimate
        if self.cache_info and 'total_sequences' in self.cache_info:
            current_sequences = self.cache_info['total_sequences']

            # If current cache is substantial, use it to refine estimate
            if current_sequences > 100_000:  # If we have significant data
                # Assume current cache represents a reasonable fraction
                # Scale up conservatively
                scaling_factor = max(5.0, 2_500_000 / current_sequences)
                estimated_full_sequences = int(current_sequences * scaling_factor)
                estimated_full_sequences = min(estimated_full_sequences, 3_000_000)  # Cap at 3M

        return estimated_full_sequences
    
    def get_training_info(self) -> dict:
        """
        Get comprehensive training information.
        
        Returns:
            Dictionary with training parameters and estimates
        """
        total_samples, dataset_tokens = self.get_dataset_size()
        
        if training_config.use_epoch_based_training:
            max_steps, total_training_tokens = self.calculate_epoch_based_steps(training_config.target_epochs)
            warmup_steps = self.calculate_dynamic_warmup_steps(max_steps)
            
            return {
                "mode": "epoch_based",
                "target_epochs": training_config.target_epochs,
                "dataset_samples": total_samples,
                "dataset_tokens": dataset_tokens,
                "total_training_tokens": total_training_tokens,
                "max_steps": max_steps,
                "warmup_steps": warmup_steps,
                "tokens_per_epoch": dataset_tokens,
                "steps_per_epoch": max_steps // training_config.target_epochs
            }
        else:
            max_steps = training_config.max_steps
            warmup_steps = training_config.warmup_steps
            epochs = training_config.target_tokens / dataset_tokens
            
            return {
                "mode": "token_based",
                "target_tokens": training_config.target_tokens,
                "dataset_samples": total_samples,
                "dataset_tokens": dataset_tokens,
                "total_training_tokens": training_config.target_tokens,
                "max_steps": max_steps,
                "warmup_steps": warmup_steps,
                "estimated_epochs": epochs,
                "tokens_per_epoch": dataset_tokens
            }


# Global instance
_dataset_calculator = None


def get_dataset_calculator(cache_info=None) -> DatasetSizeCalculator:
    """Get global dataset calculator instance with cache info."""
    global _dataset_calculator
    # FIXED: Always create new instance when cache_info is provided (for updated caches)
    if _dataset_calculator is None or cache_info is not None:
        _dataset_calculator = DatasetSizeCalculator(cache_info)
    return _dataset_calculator


def get_dynamic_max_steps() -> int:
    """Get dynamically calculated max_steps based on current config."""
    calculator = get_dataset_calculator()
    
    if training_config.use_epoch_based_training:
        max_steps, _ = calculator.calculate_epoch_based_steps(training_config.target_epochs)
        return max_steps
    else:
        return training_config.max_steps


def get_dynamic_warmup_steps() -> int:
    """Get dynamically calculated warmup_steps based on max_steps."""
    calculator = get_dataset_calculator()
    max_steps = get_dynamic_max_steps()
    return calculator.calculate_dynamic_warmup_steps(max_steps)
