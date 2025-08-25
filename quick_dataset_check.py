#!/usr/bin/env python3
"""
🔍 Quick Dataset Size Check
Verifiziert die Größe Ihres FineWeb-Edu Datasets
"""

from dataset_loader import find_cached_fineweb_data
from datasets import Dataset as HFDataset
import time

def quick_size_check():
    print("🔍 Quick Dataset Size Check")
    print("=" * 40)
    
    # Finde gecachte Dateien
    parquet_files = find_cached_fineweb_data()
    
    if not parquet_files:
        print("❌ No cached files found")
        return
    
    print(f"\n📊 Analyzing {len(parquet_files)} parquet files...")
    
    # Lade nur die erste Datei für Statistiken
    print("📂 Loading first file for analysis...")
    start_time = time.time()
    
    sample_dataset = HFDataset.from_parquet([parquet_files[0]])
    load_time = time.time() - start_time
    
    samples_in_first_file = len(sample_dataset)
    estimated_total = samples_in_first_file * len(parquet_files)
    
    print(f"\n📈 Results:")
    print(f"   First file: {samples_in_first_file:,} samples")
    print(f"   Load time: {load_time:.1f} seconds")
    print(f"   Estimated total: {estimated_total:,} samples")
    print(f"   Loading speed: {samples_in_first_file/load_time:,.0f} samples/sec")
    
    # Schätze Tokens
    if len(sample_dataset) > 0:
        sample_text = sample_dataset[0]['text']
        sample_tokens = len(sample_text.split())  # Grobe Schätzung
        estimated_tokens_per_sample = sample_tokens
        total_estimated_tokens = estimated_total * estimated_tokens_per_sample
        
        print(f"\n🎯 Token Analysis:")
        print(f"   Sample text length: {len(sample_text)} chars")
        print(f"   Estimated tokens per sample: {estimated_tokens_per_sample}")
        print(f"   Total estimated tokens: {total_estimated_tokens:,}")
        print(f"   In billions: {total_estimated_tokens/1e9:.1f}B tokens")
        
        # Vergleich mit sample-10BT
        if 8e9 <= total_estimated_tokens <= 12e9:
            print(f"   ✅ This matches sample-10BT (10B tokens)!")
        elif 80e9 <= total_estimated_tokens <= 120e9:
            print(f"   ✅ This matches sample-100BT (100B tokens)!")
        else:
            print(f"   ❓ Unknown dataset size")
    
    print(f"\n🚀 Performance Recommendations:")
    if estimated_total > 1000000:
        print("   - Use streaming=True for large datasets")
        print("   - Limit num_samples to 100k-1M for training")
        print("   - Consider using only subset of files")
    else:
        print("   - Current size is manageable")
        print("   - Can load full dataset into memory")

if __name__ == "__main__":
    quick_size_check()
