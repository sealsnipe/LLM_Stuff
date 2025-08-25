#!/usr/bin/env python3
"""
🎯 FineWeb-Edu Dataset Loader für Small Language Model Training
Lädt und preprocessed FineWeb-Edu für unser 346M Parameter Modell
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset
from transformers import AutoTokenizer
import os
import glob
from typing import Optional, Dict, List
from tqdm import tqdm
import json
from config import model_config, training_config, dataset_config, system_config

def find_cached_fineweb_data():
    """Findet bereits gecachte FineWeb-Edu Parquet-Dateien."""
    cache_base = os.path.expanduser("~/.cache/huggingface/hub")

    # Suche nach FineWeb-Edu Ordnern
    pattern = os.path.join(cache_base, "datasets--HuggingFaceFW--fineweb-edu", "snapshots", "*", "data", "*", "*.parquet")
    parquet_files = glob.glob(pattern)

    if parquet_files:
        print(f"🎯 Found {len(parquet_files)} cached parquet files")
        # Gruppiere nach Ordner
        folders = set(os.path.dirname(f) for f in parquet_files)
        print(f"   Folders: {list(folders)}")

        # Schätze Dataset-Größe
        total_size_gb = sum(os.path.getsize(f) for f in parquet_files) / (1024**3)
        print(f"   Total size: {total_size_gb:.1f} GB")

        return parquet_files

    return []

def estimate_dataset_size(parquet_files: List[str]) -> int:
    """Schätzt die Anzahl der Samples im Dataset schnell."""
    if not parquet_files:
        return 0

    try:
        # Lade nur die erste Datei zum Schätzen
        sample_dataset = HFDataset.from_parquet([parquet_files[0]])
        samples_per_file = len(sample_dataset)
        total_estimated = samples_per_file * len(parquet_files)

        print(f"📊 Estimated dataset size: {total_estimated:,} samples")
        print(f"   ({samples_per_file:,} samples per file × {len(parquet_files)} files)")

        return total_estimated
    except:
        return 0

def load_from_cached_parquet(parquet_files: List[str], num_samples: int = None):
    """Lädt Dataset direkt aus gecachten Parquet-Dateien - OPTIMIERT."""
    try:
        print(f"📂 Loading from {len(parquet_files)} parquet files...")

        # OPTIMIERUNG 1: Nur benötigte Dateien laden
        if num_samples and num_samples <= 1000000:  # Für < 1M samples
            # Schätze wie viele Dateien wir brauchen (ca. 500k samples pro Datei)
            estimated_files_needed = min(max(1, num_samples // 500000), len(parquet_files))
            parquet_files = parquet_files[:estimated_files_needed]
            print(f"🎯 Using only {len(parquet_files)} files for {num_samples:,} samples")

        # OPTIMIERUNG 2: Streaming für große Datasets
        if num_samples and num_samples > 100000:
            print("⚡ Using streaming mode for faster loading...")
            dataset = HFDataset.from_parquet(parquet_files, streaming=True)
            # Nimm nur die ersten N samples
            dataset = dataset.take(num_samples)
            # Konvertiere zu normalem Dataset
            dataset = HFDataset.from_dict({
                key: [item[key] for item in dataset]
                for key in next(iter(dataset)).keys()
            })
        else:
            # Normale Ladung für kleine Datasets
            dataset = HFDataset.from_parquet(parquet_files)
            if num_samples and num_samples < len(dataset):
                dataset = dataset.select(range(num_samples))

        print(f"✅ Loaded {len(dataset):,} samples from cached data")
        return dataset

    except Exception as e:
        print(f"❌ Error loading from parquet: {e}")
        return None

class FineWebEduDataset(Dataset):
    """FineWeb-Edu Dataset für LLM Training."""
    
    def __init__(
        self,
        tokenizer_name: str = None,
        max_length: int = None,
        num_samples: Optional[int] = None,
        cache_dir: str = None,
        streaming: bool = None
    ):
        # Use config defaults if not specified
        self.tokenizer_name = tokenizer_name or dataset_config.default_tokenizer
        self.max_length = max_length or training_config.sequence_length
        self.cache_dir = cache_dir or dataset_config.fineweb_cache_dir
        streaming = streaming if streaming is not None else dataset_config.fineweb_streaming
        
        # Setup tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        print(f"🔧 Loading FineWeb-Edu Dataset...")
        print(f"   Tokenizer: {self.tokenizer_name}")
        print(f"   Max Length: {self.max_length}")
        print(f"   Samples: {num_samples or 'All'}")
        print(f"   Streaming: {streaming}")
        print(f"   Cache Dir: {self.cache_dir}")
        
        # Load dataset
        if num_samples:
            split = f"train[:{num_samples}]"
        else:
            split = "train"
            
        try:
            # For tiny/small samples, use WikiText for fast testing
            if num_samples and num_samples <= 10000:
                print("🎯 Using WikiText for fast testing...")
                self.dataset = load_dataset(
                    "wikitext",
                    "wikitext-2-raw-v1",
                    split=f"train[:{num_samples}]",
                    streaming=False,
                    cache_dir=self.cache_dir
                )
                streaming = False
            elif num_samples and num_samples <= 100000:
                print("🎯 Using cached FineWeb-Edu data...")

                # FIRST: Try to load from cached parquet files directly
                cached_files = find_cached_fineweb_data()
                if cached_files:
                    self.dataset = load_from_cached_parquet(cached_files, num_samples)
                    if self.dataset:
                        streaming = False
                    else:
                        raise Exception("Failed to load from cached parquet")
                else:
                    raise Exception("No cached parquet files found")

                # If that fails, try the normal HuggingFace approach
                if not hasattr(self, 'dataset') or self.dataset is None:
                    print("🔄 Fallback to HuggingFace datasets...")
                    try:
                        # First try: Use any cached FineWeb-Edu data
                        self.dataset = load_dataset(
                            "HuggingFaceFW/fineweb-edu",
                            split=f"train[:{num_samples}]",
                            streaming=streaming,
                            cache_dir=self.cache_dir
                        )
                        print(f"✅ Using HuggingFace cached data ({num_samples:,} samples)")
                    except:
                        # Fallback: Try sample-10BT specifically
                        print("🔄 Fallback to sample-10BT...")
                        self.dataset = load_dataset(
                            "HuggingFaceFW/fineweb-edu",
                            name="sample-10BT",
                            split=f"train[:{num_samples}]",
                            streaming=streaming,
                            cache_dir=self.cache_dir
                        )
            elif num_samples and num_samples <= 1000000:
                print("🎯 Using FineWeb-Edu sample-100BT...")
                # Use 100BT sample for larger training (~277GB)
                self.dataset = load_dataset(
                    "HuggingFaceFW/fineweb-edu",
                    name="sample-100BT",
                    split=f"train[:{num_samples}]",
                    streaming=streaming,
                    cache_dir=self.cache_dir
                )
            else:
                print("🎯 Using FineWeb-Edu sample-350BT...")
                # Use 350BT sample for production training (~388GB)
                self.dataset = load_dataset(
                    "HuggingFaceFW/fineweb-edu",
                    name="sample-350BT",
                    split=split,
                    streaming=streaming,
                    cache_dir=self.cache_dir
                )
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            print("💡 Trying fallback with WikiText...")
            # Fallback to WikiText (much smaller)
            self.dataset = load_dataset(
                "wikitext",
                "wikitext-2-raw-v1",
                split="train[:1000]",
                streaming=False,
                cache_dir=self.cache_dir
            )
            streaming = False
        
        # Convert to list if not streaming (for indexing)
        if not streaming:
            print("📦 Converting dataset to list...")
            self.texts = []
            for item in tqdm(self.dataset, desc="Loading texts"):
                self.texts.append(item['text'])
            print(f"✅ Loaded {len(self.texts)} texts")
        else:
            self.texts = None
            print("✅ Streaming dataset ready")
    
    def __len__(self):
        if self.texts:
            return len(self.texts)
        else:
            # Für streaming datasets - grobe Schätzung
            return 1000000  # Wird durch DataLoader begrenzt
    
    def __getitem__(self, idx):
        if self.texts:
            text = self.texts[idx]
        else:
            # Für streaming - iteriere durch dataset
            for i, item in enumerate(self.dataset):
                if i == idx:
                    text = item['text']
                    break
            else:
                raise IndexError("Index out of range")
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # Labels für Causal LM (shifted input_ids)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Ignore padding tokens
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def create_fineweb_dataloader(
    dataset_size: str = "small",
    num_samples: int = None,
    batch_size: Optional[int] = None,
    streaming: bool = None,
    tokenizer_name: str = None
) -> DataLoader:
    """Erstellt DataLoader für FineWeb-Edu."""
    
    # Use config defaults
    batch_size = batch_size or training_config.batch_size
    tokenizer_name = tokenizer_name or dataset_config.default_tokenizer
    
    # Get num_samples from dataset_size if not specified
    if num_samples is None:
        if dataset_size in dataset_config.dataset_sizes:
            num_samples = dataset_config.dataset_sizes[dataset_size]["num_samples"]
        else:
            num_samples = dataset_config.fineweb_num_samples
    
    # Auto-determine streaming based on dataset size
    if streaming is None:
        streaming = dataset_size in ["large", "full"] if dataset_size in dataset_config.dataset_sizes else dataset_config.fineweb_streaming
    
    print(f"🚀 Creating FineWeb-Edu DataLoader...")
    print(f"   Dataset Size: {dataset_size}")
    print(f"   Samples: {num_samples:,}" if num_samples else "   Samples: All")
    print(f"   Batch Size: {batch_size}")
    print(f"   Sequence Length: {training_config.sequence_length}")
    print(f"   Streaming: {streaming}")
    
    dataset = FineWebEduDataset(
        tokenizer_name=tokenizer_name,
        max_length=training_config.sequence_length,
        num_samples=num_samples,
        streaming=streaming
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not streaming,  # Streaming datasets können nicht geshuffled werden
        num_workers=training_config.dataloader_num_workers,
        pin_memory=training_config.dataloader_pin_memory,
        persistent_workers=training_config.dataloader_persistent_workers,
        prefetch_factor=training_config.dataloader_prefetch_factor,
        drop_last=True
    )
    
    print(f"✅ DataLoader created successfully!")
    return dataloader

def test_dataset_loading():
    """Testet das Dataset Loading."""
    print("🧪 Testing FineWeb-Edu Dataset Loading...")
    
    # Test mit kleinem Sample
    try:
        dataloader = create_fineweb_dataloader(
            dataset_size="tiny",
            batch_size=2,
            streaming=False
        )
        
        print("\n📊 Dataset Test Results:")
        
        # Teste ersten Batch
        batch = next(iter(dataloader))
        
        print(f"   Batch Keys: {list(batch.keys())}")
        print(f"   Input IDs Shape: {batch['input_ids'].shape}")
        print(f"   Attention Mask Shape: {batch['attention_mask'].shape}")
        print(f"   Labels Shape: {batch['labels'].shape}")
        
        # Dekodiere ersten Text
        tokenizer = AutoTokenizer.from_pretrained(dataset_config.default_tokenizer)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        first_text = tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True)
        print(f"\n📝 Sample Text (first 200 chars):")
        print(f"   {first_text[:200]}...")
        
        # Statistiken
        total_tokens = (batch['input_ids'] != tokenizer.pad_token_id).sum().item()
        total_possible = batch['input_ids'].numel()
        
        print(f"\n📈 Token Statistics:")
        print(f"   Total Tokens: {total_tokens:,}")
        print(f"   Total Possible: {total_possible:,}")
        print(f"   Utilization: {total_tokens/total_possible:.1%}")
        
        print("\n✅ Dataset loading test successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Dataset loading test failed: {e}")
        return False

def show_dataset_configs():
    """Zeigt verfügbare Dataset-Konfigurationen."""
    
    print("📋 Available Dataset Configurations:")
    print("=" * 60)
    
    for name, config in dataset_config.dataset_sizes.items():
        print(f"🎯 {name.upper()}:")
        print(f"   Samples: {config['num_samples'] or 'All'}")
        print(f"   Description: {config['description']}")
        print(f"   Estimated Tokens: {config['estimated_tokens']}")
        print(f"   Training Time: {config['training_time']}")
        print()
    
    return dataset_config.dataset_sizes

if __name__ == "__main__":
    print("🎯 FineWeb-Edu Dataset Loader")
    print("=" * 50)
    
    # Zeige verfügbare Konfigurationen
    configs = show_dataset_configs()
    
    # Teste Dataset Loading
    success = test_dataset_loading()
    
    if success:
        print("\n🚀 Ready for Training!")
        print("\nUsage Examples:")
        print("# Tiny test:")
        print("dataloader = create_fineweb_dataloader('tiny')")
        print("\n# Production training:")
        print("dataloader = create_fineweb_dataloader('medium')")
    else:
        print("\n⚠️  Dataset loading failed. Check your internet connection and try again.")
