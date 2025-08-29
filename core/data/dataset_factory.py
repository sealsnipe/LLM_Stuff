"""
Dataset Factory Module

Contains the DatasetFactory class and create_gpu_optimized_dataset function.
Handles creation of optimized datasets for training with various data sources.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import os
from typing import Union, Dict, Optional

from config import training_config, dataset_config
from .fast_dataset_loader import load_samples_fast


class DatasetFactory:
    """Factory for creating optimized datasets."""
    
    def __init__(self):
        self.cache_dir = dataset_config.fineweb_cache_dir
        
    def create_synthetic_dataset(self, num_samples: int):
        """Create synthetic dataset for testing with optional sequence packing."""
        # Quiet operation - no print statements for professional output

        # Generate random token sequences
        vocab_size = 49152  # SmolLM tokenizer vocab size
        seq_length = training_config.sequence_length

        input_ids = torch.randint(
            0, vocab_size,
            (num_samples, seq_length),
            dtype=torch.long
        )

        # Apply sequence packing if enabled
        if training_config.use_sequence_packing:
            input_ids = self._apply_synthetic_packing(input_ids)

        # For causal LM, labels = input_ids
        labels = input_ids.clone()

        dataset = TensorDataset(input_ids, labels)

        dataloader = DataLoader(
            dataset,
            batch_size=training_config.batch_size,
            shuffle=True,
            num_workers=0,  # Windows compatibility
            pin_memory=True,
            drop_last=True
        )

        return dataloader

    def _apply_synthetic_packing(self, input_ids):
        """Apply sequence packing to synthetic data."""
        try:
            from .data_utils import create_packed_sequences

            # Convert to list of sequences
            sequences = [input_ids[i] for i in range(input_ids.shape[0])]

            # Create packed sequences
            packed_data = create_packed_sequences(
                sequences=sequences,
                max_length=training_config.sequence_length,
                eos_token_id=2  # SmolLM EOS token
            )

            if packed_data is not None:
                return packed_data['input_ids']
            else:
                return input_ids

        except Exception as e:
            return input_ids
    
    def create_fineweb_dataset(self, num_samples: int = None):
        """Create FineWeb-Edu dataset with sequence packing."""
        try:
            print(f"📚 Loading FineWeb-Edu dataset...")

            # Use fast_dataset_loader (FIXED: remove unsupported arguments)
            fast_dataset = load_samples_fast(
                num_samples=num_samples or dataset_config.fineweb_num_samples,
                verbose=False  # Silent mode for training
            )

            if fast_dataset is None:
                raise Exception("Failed to load FineWeb dataset")

            # Apply sequence packing if enabled
            if training_config.use_sequence_packing:
                print(f"📦 Applying sequence packing...")
                fast_dataset = self._apply_sequence_packing(fast_dataset)

            # Create DataLoader
            dataloader = DataLoader(
                fast_dataset,
                batch_size=training_config.batch_size,
                shuffle=True,
                num_workers=0,  # Windows compatibility
                pin_memory=True,
                drop_last=True
            )

            print(f"✅ FineWeb dataset loaded: {len(fast_dataset):,} samples")
            return dataloader

        except Exception as e:
            print(f"⚠️ FineWeb loading failed: {e}")
            return None

    def _apply_sequence_packing(self, dataset):
        """Apply sequence packing to dataset with tokenization."""
        try:
            from .data_utils import create_packed_sequences
            from transformers import AutoTokenizer

            # Initialize tokenizer
            tokenizer = AutoTokenizer.from_pretrained(dataset_config.default_tokenizer)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # FIXED: Tokenize raw text data first
            sequences = []
            max_samples = min(len(dataset), 1000)  # Limit for packing

            for i in range(max_samples):
                sample = dataset[i]

                # Handle different sample formats
                if isinstance(sample, dict):
                    if 'input_ids' in sample:
                        # Already tokenized
                        sequences.append(sample['input_ids'])
                    elif 'text' in sample:
                        # Raw text - tokenize it
                        text = sample['text']
                        tokens = tokenizer.encode(
                            text,
                            add_special_tokens=True,
                            max_length=training_config.sequence_length,
                            truncation=True,
                            return_tensors='pt'
                        ).squeeze()
                        sequences.append(tokens)
                elif isinstance(sample, torch.Tensor):
                    sequences.append(sample)

            if not sequences:
                print("⚠️ No valid sequences found for packing")
                return dataset

            # Create packed sequences
            packed_data = create_packed_sequences(
                sequences=sequences,
                max_length=training_config.sequence_length,
                eos_token_id=tokenizer.eos_token_id
            )

            if packed_data is None:
                print("⚠️ Sequence packing failed, using original dataset")
                return dataset

            # Create new dataset from packed sequences with proper format
            class PackedDataset(torch.utils.data.Dataset):
                def __init__(self, input_ids):
                    self.input_ids = input_ids

                def __len__(self):
                    return len(self.input_ids)

                def __getitem__(self, idx):
                    return {'input_ids': self.input_ids[idx]}

            packed_dataset = PackedDataset(packed_data['input_ids'])
            print(f"✅ Sequence packing applied: {len(sequences)} → {len(packed_dataset)} sequences")

            return packed_dataset

        except Exception as e:
            print(f"⚠️ Sequence packing failed: {e}")
            return dataset
    
    def create_packed_cache_dataset(self, cache_path: str = None):
        """Create dataset from packed cache using the original system."""
        try:
            if cache_path:
                cache_dir = cache_path
            else:
                # FIXED: Look for actual cache directories with chunks
                base_cache_dir = training_config.packed_cache_dir

                # Try common cache locations
                possible_dirs = [
                    os.path.join(base_cache_dir, "512", "FineWeb"),
                    os.path.join(base_cache_dir, "1024", "FineWeb"),
                    os.path.join(base_cache_dir, "2048", "FineWeb"),
                    base_cache_dir
                ]

                cache_dir = None
                for dir_path in possible_dirs:
                    if os.path.exists(dir_path):
                        # Check for cache files
                        cache_files = [f for f in os.listdir(dir_path) if f.startswith('packed_chunk_') and f.endswith('.pt')]
                        if cache_files:
                            cache_dir = dir_path
                            print(f"📦 Found packed cache: {cache_dir} ({len(cache_files)} chunks)")
                            break

                if not cache_dir:
                    print(f"⚠️ No packed cache found in {base_cache_dir}")
                    return None

            # Use the original sequence_packing_cache system
            import sys
            import os
            scripts_path = os.path.join(os.path.dirname(__file__), '..', '..', 'scripts')
            if scripts_path not in sys.path:
                sys.path.append(scripts_path)

            # Import with fallback
            try:
                from sequence_packing_cache import create_packed_dataloader
            except ImportError as e:
                print(f"⚠️ Could not import sequence_packing_cache: {e}")
                return None

            dataloader = create_packed_dataloader(
                cache_dir=cache_dir,
                batch_size=training_config.batch_size,
                device='cuda',
                num_workers=0  # Windows compatibility
            )

            print(f"✅ Using packed cache dataset: {cache_dir}")
            return dataloader

        except Exception as e:
            print(f"⚠️ Error creating packed cache dataset: {e}")
            import traceback
            traceback.print_exc()
            return None


def create_gpu_optimized_dataset(num_samples: int = None, use_real_data: bool = True, dataset_size: str = "auto", return_splits: bool = False):
    """
    Erstellt GPU-optimierten Datensatz mit automatischer Token-basierter Größe.
    
    Args:
        num_samples: Anzahl Samples (None = auto-calculate from target_tokens)
        use_real_data: Verwende echte FineWeb-Edu Daten
        dataset_size: "auto" = calculate from training_config.target_tokens
        return_splits: Ob Train/Val Splits zurückgegeben werden sollen
    """
    
    factory = DatasetFactory()
    
    # Auto-calculate samples if not provided
    if num_samples is None or dataset_size == "auto":
        num_samples = dataset_config.get_samples_for_tokens(training_config.target_tokens)
    
    # Try different data sources in order of preference
    dataloader = None
    
    if use_real_data:
        # 1. Try packed cache first (fastest)
        if training_config.use_packed_cache:
            dataloader = factory.create_packed_cache_dataset()
            if dataloader:
                print("✅ Using packed cache dataset")
                return dataloader
        
        # 2. Try FineWeb-Edu
        dataloader = factory.create_fineweb_dataset(num_samples)
        if dataloader:
            print("✅ Using FineWeb-Edu dataset")
            return dataloader
    
    # 3. Fallback to synthetic data
    print("⚠️ Falling back to synthetic dataset")
    dataloader = factory.create_synthetic_dataset(num_samples)
    
    return dataloader
