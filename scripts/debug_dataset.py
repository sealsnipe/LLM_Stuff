#!/usr/bin/env python3
"""
Minimal Dataset Loading Test - Find the bottleneck!
"""

import time
import os
import pandas as pd
import glob
from datasets import Dataset
from transformers import AutoTokenizer

def test_dataset_loading():
    """Test dataset loading step by step."""
    
    print("🔍 DATASET LOADING DEBUG TEST")
    print("="*50)
    
    # Step 1: Test Local Parquet Discovery
    print("Step 1: Testing local parquet discovery...")
    start_time = time.time()

    try:
        # Finde lokale Parquet-Dateien
        parquet_files = glob.glob("cache/fineweb/*.parquet")
        discovery_time = time.time() - start_time
        print(f"✅ Local parquet discovery: {discovery_time:.4f}s")
        print(f"   Found {len(parquet_files)} parquet files")
        if parquet_files:
            for i, file in enumerate(parquet_files[:3]):
                size_mb = os.path.getsize(file) / (1024**2)
                print(f"   {i+1}. {os.path.basename(file)} ({size_mb:.1f}MB)")
        else:
            print("❌ No parquet files found in cache/fineweb/")
            return
    except Exception as e:
        print(f"❌ Parquet discovery failed: {e}")
        return
    
    # Step 2: Test Small Parquet Loading
    print("\nStep 2: Testing small parquet loading...")
    start_time = time.time()

    try:
        # Lade nur erste Parquet-Datei
        first_file = parquet_files[0]
        print(f"   Loading: {os.path.basename(first_file)}")

        df = pd.read_parquet(first_file)
        small_load_time = time.time() - start_time
        print(f"✅ Small parquet load: {small_load_time:.2f}s")
        print(f"   Rows: {len(df):,}")
        print(f"   Columns: {list(df.columns)}")
        if 'text' in df.columns and len(df) > 0:
            print(f"   First sample length: {len(df['text'].iloc[0])} chars")

        # Convert to HuggingFace Dataset für weitere Tests
        small_dataset = Dataset.from_pandas(df.head(100))  # Nur erste 100 rows
        print(f"   Converted to HF Dataset: {len(small_dataset)} samples")

    except Exception as e:
        print(f"❌ Small parquet loading failed: {e}")
        return
    
    # Step 3: Test Tokenizer Loading
    print("\nStep 3: Testing tokenizer loading...")
    start_time = time.time()
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
        tokenizer_time = time.time() - start_time
        print(f"✅ Tokenizer loading: {tokenizer_time:.2f}s")
    except Exception as e:
        print(f"❌ Tokenizer loading failed: {e}")
        return
    
    # Step 4: Test Single Text Tokenization
    print("\nStep 4: Testing single text tokenization...")
    start_time = time.time()
    
    try:
        sample_text = small_dataset[0]['text'][:1000]  # Nur erste 1000 chars
        tokens = tokenizer.encode(sample_text, add_special_tokens=False)
        tokenize_time = time.time() - start_time
        print(f"✅ Single tokenization: {tokenize_time:.4f}s")
        print(f"   Text length: {len(sample_text)} chars")
        print(f"   Token count: {len(tokens)}")
    except Exception as e:
        print(f"❌ Tokenization failed: {e}")
        return
    
    # Step 5: Test Batch Tokenization
    print("\nStep 5: Testing batch tokenization (10 texts)...")
    start_time = time.time()
    
    try:
        batch_texts = [small_dataset[i]['text'][:1000] for i in range(10)]
        for text in batch_texts:
            tokens = tokenizer.encode(text, add_special_tokens=False)
        batch_time = time.time() - start_time
        print(f"✅ Batch tokenization (10 texts): {batch_time:.4f}s")
        print(f"   Average per text: {batch_time/10:.4f}s")
    except Exception as e:
        print(f"❌ Batch tokenization failed: {e}")
        return
    
    # Step 6: Test Large Dataset Info
    print("\nStep 6: Testing large dataset info...")
    start_time = time.time()
    
    try:
        # Versuche größeres Dataset zu laden (aber nicht alles!)
        medium_dataset = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train[:1000]")
        medium_time = time.time() - start_time
        print(f"✅ Medium dataset (1000 samples): {medium_time:.2f}s")
        print(f"   Sample count: {len(medium_dataset)}")
    except Exception as e:
        print(f"❌ Medium dataset loading failed: {e}")
        return
    
    # Step 7: Check Cache Directory
    print("\nStep 7: Checking cache directory...")
    cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
    if os.path.exists(cache_dir):
        cache_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                        for dirpath, dirnames, filenames in os.walk(cache_dir)
                        for filename in filenames) / (1024**3)  # GB
        print(f"✅ Cache directory exists: {cache_dir}")
        print(f"   Cache size: {cache_size:.2f} GB")
    else:
        print(f"❌ Cache directory not found: {cache_dir}")
    
    print("\n" + "="*50)
    print("🎯 PERFORMANCE SUMMARY:")
    print(f"   Dataset discovery: {discovery_time:.2f}s")
    print(f"   Small load (100):  {small_load_time:.2f}s")
    print(f"   Tokenizer load:    {tokenizer_time:.2f}s")
    print(f"   Single tokenize:   {tokenize_time:.4f}s")
    print(f"   Batch tokenize:    {batch_time:.4f}s")
    print(f"   Medium load (1k):  {medium_time:.2f}s")

def test_problematic_sizes():
    """Test different parquet loading sizes to find the breaking point."""

    print("\n🔍 PARQUET SIZE SCALING TEST")
    print("="*50)

    # Finde Parquet-Dateien
    parquet_files = glob.glob("cache/fineweb/*.parquet")
    if not parquet_files:
        print("❌ No parquet files found")
        return

    sizes = [100, 500, 1000, 5000, 10000, 50000]

    for size in sizes:
        print(f"\nTesting {size:,} samples...")
        start_time = time.time()

        try:
            # Lade genug Parquet-Dateien für die gewünschte Sample-Anzahl
            all_dfs = []
            total_samples = 0

            for file in parquet_files:
                if total_samples >= size:
                    break

                df = pd.read_parquet(file)
                needed = min(len(df), size - total_samples)
                all_dfs.append(df.head(needed))
                total_samples += needed
                print(f"   Loaded {os.path.basename(file)}: +{needed} samples (total: {total_samples})")

            # Combine DataFrames
            combined_df = pd.concat(all_dfs, ignore_index=True)
            dataset = Dataset.from_pandas(combined_df)

            load_time = time.time() - start_time
            print(f"✅ {len(dataset):,} samples: {load_time:.2f}s ({load_time/len(dataset)*1000:.2f}ms per sample)")

            # Memory check
            import psutil
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            print(f"   Memory usage: {memory_mb:.0f} MB")

            del dataset, combined_df, all_dfs  # Cleanup

        except Exception as e:
            print(f"❌ {size:,} samples failed: {e}")
            break

        # Stop if it takes too long
        if load_time > 30:
            print(f"⚠️  Stopping test - {size:,} samples took {load_time:.2f}s")
            break

if __name__ == "__main__":
    test_dataset_loading()
    test_problematic_sizes()
