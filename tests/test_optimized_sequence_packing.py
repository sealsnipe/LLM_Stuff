"""
🧪 TEST: Optimized Sequence Packing Performance
Benchmarks the new optimized implementation vs original
"""

import sys
import os
import time
import torch
import glob
from transformers import AutoTokenizer

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimized_sequence_packing import (
    pack_sequences_heap_optimized,
    batch_tokenize_optimized,
    streaming_pack_sequences,
    load_parquet_fast_pandas,
    streaming_pack_sequences_pandas,
    reset_global_carry
)

def create_test_data(num_samples: int = 1000):
    """Erstelle Test-Daten verschiedener Längen"""
    import random
    
    # Verschiedene Text-Längen simulieren
    test_texts = []
    
    # Kurze Texte (10-50 tokens)
    short_texts = [
        "This is a short text example.",
        "Machine learning is fascinating.",
        "Python programming rocks!",
        "AI will change the world.",
        "Deep learning models are powerful."
    ]
    
    # Mittlere Texte (100-300 tokens)
    medium_base = "This is a medium length text that contains multiple sentences and covers various topics. " * 5
    
    # Lange Texte (400-500 tokens)
    long_base = "This is a very long text that simulates real-world documents with substantial content. " * 20
    
    for i in range(num_samples):
        if i % 10 == 0:  # 10% lange Texte
            text = long_base + f" Document {i}."
        elif i % 3 == 0:  # 30% mittlere Texte
            text = medium_base + f" Article {i}."
        else:  # 60% kurze Texte
            text = random.choice(short_texts) + f" Sample {i}."
        
        test_texts.append(text)
    
    return test_texts

def benchmark_tokenization():
    """Benchmark batched vs serial tokenization"""
    print("\n🧪 BENCHMARK: Tokenization Performance")
    print("=" * 50)
    
    # Setup
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    test_sizes = [100, 500, 1000, 2000]
    
    for size in test_sizes:
        print(f"\n📊 Testing {size:,} samples:")
        test_texts = create_test_data(size)
        
        # Test batched tokenization
        start_time = time.time()
        tokenized_batch, fragments_batch = batch_tokenize_optimized(
            test_texts, tokenizer, max_length=512, batch_size=64
        )
        batch_time = time.time() - start_time
        
        # Test serial tokenization (original method)
        start_time = time.time()
        tokenized_serial = []
        fragments_serial = []
        
        for text in test_texts:
            if len(text) > 100000:
                continue
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) < 32:
                fragments_serial.extend(tokens)
            else:
                tokens_with_eos = tokens + [tokenizer.eos_token_id]
                tokenized_serial.append(tokens_with_eos)
        
        serial_time = time.time() - start_time
        
        # Results
        speedup = serial_time / batch_time if batch_time > 0 else float('inf')
        print(f"   Batched: {batch_time:.2f}s ({len(tokenized_batch):,} sequences)")
        print(f"   Serial:  {serial_time:.2f}s ({len(tokenized_serial):,} sequences)")
        print(f"   Speedup: {speedup:.1f}x faster! 🚀")

def benchmark_packing():
    """Benchmark heap-based vs original packing"""
    print("\n🧪 BENCHMARK: Packing Performance")
    print("=" * 50)
    
    # Setup
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    test_sizes = [100, 500, 1000]
    
    for size in test_sizes:
        print(f"\n📊 Testing {size:,} samples:")
        test_texts = create_test_data(size)
        
        # Reset carry-over state
        reset_global_carry()
        
        # Test optimized packing
        start_time = time.time()
        result_optimized = pack_sequences_heap_optimized(test_texts, tokenizer, max_length=512)
        optimized_time = time.time() - start_time
        
        # Calculate utilization
        if result_optimized['input_ids'].size(0) > 0:
            total_tokens = (result_optimized['input_ids'] != tokenizer.pad_token_id).sum().item()
            total_positions = result_optimized['input_ids'].numel()
            utilization = (total_tokens / total_positions) * 100 if total_positions > 0 else 0
        else:
            utilization = 0
        
        print(f"   Optimized: {optimized_time:.2f}s")
        print(f"   Sequences: {result_optimized['input_ids'].size(0):,}")
        print(f"   Utilization: {utilization:.1f}%")

def test_streaming():
    """Test streaming functionality"""
    print("\n🧪 TEST: Streaming Functionality")
    print("=" * 50)
    
    # Setup
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create test dataset
    test_texts = create_test_data(1000)
    test_dataset = [{'text': text} for text in test_texts]
    
    # Reset carry-over state
    reset_global_carry()
    
    # Test streaming
    start_time = time.time()
    packed_chunks = streaming_pack_sequences(
        test_dataset, tokenizer, max_length=512, chunk_size=200
    )
    streaming_time = time.time() - start_time
    
    # Calculate total sequences
    total_sequences = sum(chunk['input_ids'].size(0) for chunk in packed_chunks)
    
    print(f"   Time: {streaming_time:.2f}s")
    print(f"   Chunks: {len(packed_chunks):,}")
    print(f"   Total sequences: {total_sequences:,}")
    print(f"   Avg sequences/chunk: {total_sequences/len(packed_chunks):.1f}")

def test_memory_usage():
    """Test memory efficiency"""
    print("\n🧪 TEST: Memory Usage")
    print("=" * 50)
    
    import psutil
    import gc
    
    # Setup
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Measure baseline memory
    gc.collect()
    baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    # Test with larger dataset
    test_texts = create_test_data(2000)
    
    # Reset carry-over state
    reset_global_carry()
    
    # Measure memory during processing
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    result = pack_sequences_heap_optimized(test_texts, tokenizer, max_length=512)
    
    peak_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    # Cleanup
    del result
    del test_texts
    gc.collect()
    
    final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    print(f"   Baseline: {baseline_memory:.1f} MB")
    print(f"   Start: {start_memory:.1f} MB")
    print(f"   Peak: {peak_memory:.1f} MB")
    print(f"   Final: {final_memory:.1f} MB")
    print(f"   Memory increase: {peak_memory - baseline_memory:.1f} MB")

def test_pandas_loading():
    """Test Pandas-based parquet loading"""
    print("\n🧪 TEST: Pandas Parquet Loading")
    print("=" * 50)

    try:
        # Setup tokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Test Pandas loading
        print("📥 Testing Pandas parquet loading...")

        # Look for cache directory - check multiple possible locations
        cache_dirs = [
            "../cache",
            "../../cache",
            "cache",
            "../cache/fineweb",
            "../../cache/fineweb",
            "cache/fineweb",
            "C:/Users/AI-Matze/AI/llm-coding/LLM_Stuff/cache/fineweb",
            "/Users/AI-Matze/AI/llm-coding/LLM_Stuff/cache/fineweb"
        ]
        cache_dir = None

        for dir_path in cache_dirs:
            if os.path.exists(dir_path):
                parquet_files = glob.glob(os.path.join(dir_path, "*.parquet"))
                if parquet_files:
                    cache_dir = dir_path
                    print(f"   📁 Found {len(parquet_files)} parquet files in {dir_path}")
                    break

        if not cache_dir:
            print("   ⚠️  No cached parquet files found - skipping pandas test")
            return

        print(f"   📁 Found cache directory: {cache_dir}")

        # Test fast loading
        start_time = time.time()
        texts = load_parquet_fast_pandas(cache_dir, num_samples=1000)
        load_time = time.time() - start_time

        if not texts:
            print("   ❌ No texts loaded")
            return

        print(f"   ⚡ Pandas loading: {load_time:.2f}s ({len(texts):,} texts)")
        print(f"   📊 Speed: {len(texts)/load_time:,.0f} texts/sec")

        # Test streaming packing with Pandas
        print("\n🌊 Testing streaming packing with Pandas...")
        reset_global_carry()

        start_time = time.time()
        packed_chunks = streaming_pack_sequences_pandas(
            cache_dir=cache_dir,
            tokenizer=tokenizer,
            max_length=512,
            chunk_size=500,
            max_samples=1000
        )
        streaming_time = time.time() - start_time

        # Calculate stats
        total_sequences = sum(chunk['input_ids'].size(0) for chunk in packed_chunks)

        if total_sequences > 0:
            # Calculate utilization from first chunk
            first_chunk = packed_chunks[0]
            total_tokens = (first_chunk['input_ids'] != tokenizer.pad_token_id).sum().item()
            total_positions = first_chunk['input_ids'].numel()
            utilization = (total_tokens / total_positions) * 100 if total_positions > 0 else 0

            print(f"   ✅ Streaming results:")
            print(f"      Time: {streaming_time:.2f}s")
            print(f"      Chunks: {len(packed_chunks):,}")
            print(f"      Total sequences: {total_sequences:,}")
            print(f"      Utilization: {utilization:.1f}%")
            print(f"      Speed: {1000/streaming_time:,.0f} samples/sec")

            # Estimate for 16M samples
            estimated_time_16m = (16_000_000 / (1000/streaming_time)) / 60 if streaming_time > 0 else 0
            print(f"      📈 Estimated time for 16M samples: {estimated_time_16m:.0f} minutes")

            if estimated_time_16m < 120:
                print(f"      🚀 SUCCESS: Under 2h target!")
            else:
                print(f"      ⚠️  Above 2h target")
        else:
            print("   ❌ No sequences generated")

    except Exception as e:
        print(f"   ❌ Pandas test failed: {e}")
        import traceback
        traceback.print_exc()

def test_with_real_data():
    """Test with real dataset subset (fallback method)"""
    print("\n🧪 TEST: Real Dataset Performance (HF Datasets)")
    print("=" * 50)

    try:
        # Import fast dataset loader
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from fast_dataset_loader import load_samples_fast

        # Setup tokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Test with small real dataset
        print("📥 Loading real dataset subset...")
        dataset = load_samples_fast(1000, verbose=False)  # 1k samples for quick test

        if dataset is None:
            print("   ⚠️  No cached dataset found - skipping real data test")
            return

        # Extract texts
        texts = [item['text'] for item in dataset]
        print(f"   📊 Loaded {len(texts):,} real texts")

        # Reset carry-over state
        reset_global_carry()

        # Test optimized packing
        start_time = time.time()
        result = pack_sequences_heap_optimized(texts, tokenizer, max_length=512)
        total_time = time.time() - start_time

        # Calculate stats
        if result['input_ids'].size(0) > 0:
            total_tokens = (result['input_ids'] != tokenizer.pad_token_id).sum().item()
            total_positions = result['input_ids'].numel()
            utilization = (total_tokens / total_positions) * 100 if total_positions > 0 else 0

            # Estimate for full dataset
            samples_per_sec = len(texts) / total_time if total_time > 0 else 0
            estimated_time_16m = (16_000_000 / samples_per_sec) / 60 if samples_per_sec > 0 else 0  # minutes

            print(f"   ✅ Results:")
            print(f"      Time: {total_time:.2f}s")
            print(f"      Sequences: {result['input_ids'].size(0):,}")
            print(f"      Utilization: {utilization:.1f}%")
            print(f"      Speed: {samples_per_sec:,.0f} samples/sec")
            print(f"      📈 Estimated time for 16M samples: {estimated_time_16m:.0f} minutes")

            if estimated_time_16m < 120:  # Less than 2 hours
                print(f"      🚀 SUCCESS: Under 2h target!")
            else:
                print(f"      ⚠️  Still above 2h target")
        else:
            print("   ❌ No sequences generated")

    except Exception as e:
        print(f"   ❌ Real data test failed: {e}")
        import traceback
        traceback.print_exc()

def run_all_tests():
    """Run all performance tests"""
    print("🚀 OPTIMIZED SEQUENCE PACKING TESTS")
    print("=" * 60)

    try:
        benchmark_tokenization()
        benchmark_packing()
        test_streaming()
        test_memory_usage()
        test_pandas_loading()  # Test Pandas loading first
        test_with_real_data()  # Fallback to HF datasets

        print("\n✅ ALL TESTS COMPLETED!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()
