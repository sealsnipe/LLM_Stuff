#!/usr/bin/env python3
"""
 Fast Dataset Loader mit Progress-Anzeige
Schnelles, transparentes Laden von FineWeb-Edu mit Fortschrittsanzeige
"""

import os
import glob
import time
from typing import List, Optional
from tqdm import tqdm
from datasets import Dataset as HFDataset, concatenate_datasets
from config import dataset_config

def find_cached_files_fast(verbose=True):
    """Findet gecachte Parquet-Dateien SCHNELL."""
    if verbose:
        print("Searching for cached parquet files...")
    
    # Priorität 1: cache/fineweb
    cache_dir = "cache/fineweb"
    if os.path.exists(cache_dir):
        files = glob.glob(os.path.join(cache_dir, "*.parquet"))
        if files:
            if verbose:
                print(f"Found {len(files)} files in {cache_dir}")
            return sorted(files)
    
    # Priorität 2: HuggingFace Cache
    hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
    pattern = os.path.join(hf_cache, "datasets--HuggingFaceFW--fineweb-edu", "**", "*.parquet")
    files = glob.glob(pattern, recursive=True)
    if files:
        if verbose:
            print(f"Found {len(files)} files in HuggingFace cache")
        return sorted(files)

    if verbose:
        print("No cached files found")
    return []

def load_samples_fast(num_samples: int, verbose: bool = True):
    """
    Lädt Samples SCHNELL mit detaillierter Progress-Anzeige.
    Verwendet die bewährte Sample-basierte Logik.
    """
    if verbose:
        print(f"Fast Loading: {num_samples:,} samples")
    start_time = time.time()

    # Finde gecachte Dateien
    cached_files = find_cached_files_fast(verbose=verbose)
    if not cached_files:
        if verbose:
            print("No cached files found - cannot load")
        return None

    if verbose:
        # Zeige Dateien
        print(f" Available files ({len(cached_files)}):")
        total_size_gb = 0
        for i, file in enumerate(cached_files[:5]):  # Zeige erste 5
            basename = os.path.basename(file)
            size_mb = os.path.getsize(file) / (1024**2)
            total_size_gb += size_mb / 1024
            print(f"   {i+1}. {basename} ({size_mb:.1f}MB)")

        if len(cached_files) > 5:
            print(f"   ... and {len(cached_files)-5} more files")
        print(f"   Total size: {total_size_gb:.1f}GB")
    
    # Intelligente Datei-Auswahl basierend auf Samples
    if num_samples <= 1000:
        files_to_load = cached_files[:1]  # 1 Datei für tiny
        if verbose:
            print(f" Tiny dataset: Using 1 file")
    elif num_samples <= 10000:
        files_to_load = cached_files[:1]  # 1 Datei für small
        if verbose:
            print(f" Small dataset: Using 1 file")
    elif num_samples <= 100000:
        files_to_load = cached_files[:2]  # 2 Dateien für medium-small
        if verbose:
            print(f" Medium-small dataset: Using 2 files")
    elif num_samples <= 300000:
        files_to_load = cached_files[:3]  # 3 Dateien für medium
        if verbose:
            print(f" Medium dataset: Using 3 files")
    elif num_samples <= 1000000:
        files_to_load = cached_files[:6]  # 6 Dateien für large
        if verbose:
            print(f" Large dataset: Using 6 files")
    else:
        files_to_load = cached_files  # Alle Dateien für production
        if verbose:
            print(f" Production dataset: Using all {len(cached_files)} files")

    if verbose:
        print(f"Loading {len(files_to_load)} files...")
    else:
        print(f"Loading {len(files_to_load)} files for {num_samples:,} samples...")
    
    # Lade Dateien mit Progress Bar
    datasets = []
    total_samples_loaded = 0

    if verbose:
        pbar_desc = "Loading files"
    else:
        pbar_desc = "Loading"

    with tqdm(total=len(files_to_load), desc=pbar_desc, unit="file", disable=not verbose) as pbar:
        for i, file in enumerate(files_to_load):
            basename = os.path.basename(file)
            pbar.set_description(f"Loading {basename}")
            
            # Lade einzelne Datei
            file_start = time.time()
            file_dataset = HFDataset.from_parquet(file)
            file_time = time.time() - file_start
            
            datasets.append(file_dataset)
            total_samples_loaded += len(file_dataset)
            
            # Update Progress
            pbar.update(1)
            pbar.set_postfix({
                'samples': f"{total_samples_loaded:,}",
                'speed': f"{len(file_dataset)/file_time:.0f}/s",
                'file_time': f"{file_time:.1f}s"
            })
            
            # Early exit wenn genug Samples
            if total_samples_loaded >= num_samples:
                if verbose:
                    print(f"  Early exit: Got {total_samples_loaded:,} samples (target: {num_samples:,})")
                break

    # Kombiniere Datasets
    if verbose:
        print("🔗 Combining datasets...")
    if len(datasets) == 1:
        combined_dataset = datasets[0]
    else:
        combined_dataset = concatenate_datasets(datasets)

    # Limitiere auf gewünschte Anzahl
    if num_samples < len(combined_dataset):
        if verbose:
            print(f"  Selecting first {num_samples:,} samples from {len(combined_dataset):,}")
        combined_dataset = combined_dataset.select(range(num_samples))

    # Final Stats
    elapsed = time.time() - start_time
    samples_per_sec = len(combined_dataset) / elapsed if elapsed > 0 else 0

    if verbose:
        print(f" SUCCESS!")
        print(f"   Samples: {len(combined_dataset):,}")
        print(f"   Time: {elapsed:.1f}s")
        print(f"   Speed: {samples_per_sec:,.0f} samples/sec")
        print(f"   Files used: {len(datasets)}/{len(cached_files)}")

        # Zeige nur ein kompaktes Beispiel
        if len(combined_dataset) > 0:
            sample = combined_dataset[0]
            text = sample.get('text', '')
            text_preview = text[:80] + "..." if len(text) > 80 else text
            print(f" Sample: {text_preview} (score: {sample.get('score', 'N/A')})")
    else:
        # Kompakte Ausgabe für Training
        print(f"Loaded {len(combined_dataset):,} samples in {elapsed:.1f}s ({samples_per_sec:,.0f} samples/sec)")

    return combined_dataset

def create_train_val_splits(dataset, train_ratio: float = 0.9):
    """Erstellt Train/Validation Splits."""
    print(f"  Creating train/val splits ({train_ratio:.1%} train, {1-train_ratio:.1%} val)...")

    total_samples = len(dataset)
    train_size = int(total_samples * train_ratio)
    val_size = total_samples - train_size

    # Erstelle Splits
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, total_samples))

    print(f" Split Results:")
    print(f"   Train: {len(train_dataset):,} samples ({len(train_dataset)/total_samples:.1%})")
    print(f"   Val: {len(val_dataset):,} samples ({len(val_dataset)/total_samples:.1%})")
    print(f"   Total: {total_samples:,} samples")

    # Zeige Beispiele aus beiden Splits
    print(f"\n Train Split Sample:")
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        for key, value in sample.items():
            if isinstance(value, str):
                preview = value[:80] + "..." if len(value) > 80 else value
                print(f"   {key}: {preview}")
            else:
                print(f"   {key}: {value}")

    print(f"\n Val Split Sample:")
    if len(val_dataset) > 0:
        sample = val_dataset[0]
        for key, value in sample.items():
            if isinstance(value, str):
                preview = value[:80] + "..." if len(value) > 80 else value
                print(f"   {key}: {preview}")
            else:
                print(f"   {key}: {value}")

    return train_dataset, val_dataset

def analyze_dataset_structure(dataset, name="Dataset"):
    """Analysiert die Struktur des Datasets."""
    print(f"\n🔍 {name} Structure Analysis:")
    print("-" * 50)

    if len(dataset) == 0:
        print(" Empty dataset")
        return

    # Erste Sample analysieren
    sample = dataset[0]
    print(f" Dataset Info:")
    print(f"   Total samples: {len(dataset):,}")
    print(f"   Keys: {list(sample.keys())}")

    # Analysiere jeden Key
    for key, value in sample.items():
        print(f"\n Key: '{key}'")
        print(f"   Type: {type(value).__name__}")

        if isinstance(value, str):
            # Text-Statistiken
            lengths = [len(dataset[i][key]) for i in range(min(100, len(dataset)))]
            avg_len = sum(lengths) / len(lengths)
            print(f"   Avg length: {avg_len:.0f} chars")
            print(f"   Min/Max: {min(lengths)}/{max(lengths)} chars")
            print(f"   Sample: {value[:100]}...")
        elif isinstance(value, (int, float)):
            # Numerische Statistiken
            values = [dataset[i][key] for i in range(min(100, len(dataset)))]
            print(f"   Sample values: {values[:5]}...")
        elif isinstance(value, list):
            # Listen-Statistiken
            print(f"   Length: {len(value)}")
            print(f"   Sample: {value[:10]}...")
        else:
            print(f"   Value: {value}")

def create_fast_dataloader(num_samples: int = None, batch_size: int = 32, with_splits: bool = True):
    """Erstellt DataLoader mit dem schnellen Loading und optionalen Splits."""
    from torch.utils.data import DataLoader

    # Verwende Config wenn nicht angegeben
    if num_samples is None:
        profile = dataset_config.default_dataset_size
        num_samples = dataset_config.dataset_sizes[profile]['num_samples']
        print(f" Using profile '{profile}': {num_samples:,} samples")

    # Lade Dataset
    dataset = load_samples_fast(num_samples)
    if dataset is None:
        print(" Failed to load dataset")
        return None

    # Analysiere Dataset-Struktur
    analyze_dataset_structure(dataset, "Loaded Dataset")

    # Erstelle Splits wenn gewünscht
    if with_splits:
        train_dataset, val_dataset = create_train_val_splits(dataset, train_ratio=0.9)

        # Analysiere Splits
        analyze_dataset_structure(train_dataset, "Train Split")
        analyze_dataset_structure(val_dataset, "Validation Split")

        return {
            'full_dataset': dataset,
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'total_samples': len(dataset)
        }
    else:
        return {
            'full_dataset': dataset,
            'total_samples': len(dataset)
        }

if __name__ == "__main__":
    print(" Testing Fast Dataset Loader with Splits")
    print("=" * 60)

    # Test verschiedene Größen mit Splits
    test_sizes = [1000, 10000, 100000]

    for size in test_sizes:
        print(f"\n Testing {size:,} samples with splits:")
        print("=" * 50)

        # Test mit Splits
        result = create_fast_dataloader(num_samples=size, with_splits=True)

        if result:
            print(f"\n SUCCESS: Complete dataset with splits created!")
            print(f"   Total: {result['total_samples']:,} samples")
            print(f"   Train: {result['train_samples']:,} samples")
            print(f"   Val: {result['val_samples']:,} samples")
        else:
            print(f"    Failed")

        print("-" * 60)
