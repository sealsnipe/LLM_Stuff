#!/usr/bin/env python3
"""
🧪 TEST: Packed Sequence Cache System
Validates the complete cache pipeline

Tests:
1. Cache creation with small dataset
2. Cache loading and validation
3. DataLoader integration
4. Performance benchmarks
5. Resume capability
"""

import os
import sys
import time
import torch
import tempfile
import shutil
from transformers import AutoTokenizer

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sequence_packing_cache import PackedCacheCreator, PackedSequenceDataset, create_packed_dataloader

def create_test_data(output_dir: str, num_samples: int = 1000):
    """Create small test dataset for validation"""
    import pandas as pd
    
    print(f"📝 Creating test dataset: {num_samples} samples")
    
    # Generate test texts of varying lengths
    test_texts = []
    
    for i in range(num_samples):
        if i % 10 == 0:  # 10% long texts
            text = "This is a very long text that simulates real documents. " * 20 + f" Document {i}."
        elif i % 3 == 0:  # 30% medium texts  
            text = "This is a medium length text with multiple sentences. " * 5 + f" Article {i}."
        else:  # 60% short texts
            text = f"Short text example {i}. Machine learning is fascinating."
        
        test_texts.append(text)
    
    # Create DataFrame and save as parquet
    df = pd.DataFrame({'text': test_texts})
    os.makedirs(output_dir, exist_ok=True)
    
    parquet_file = os.path.join(output_dir, "test_data.parquet")
    df.to_parquet(parquet_file, index=False)
    
    print(f"✅ Test data created: {parquet_file}")
    return parquet_file

def test_cache_creation():
    """Test 1: Cache creation"""
    print("\n🧪 TEST 1: Cache Creation")
    print("=" * 40)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data
        test_input_dir = os.path.join(temp_dir, "input")
        test_output_dir = os.path.join(temp_dir, "output")
        
        create_test_data(test_input_dir, num_samples=100)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create cache
        creator = PackedCacheCreator(tokenizer, max_length=512, use_compression=False)
        
        start_time = time.time()
        creator.create_cache(test_input_dir, test_output_dir, chunk_size=50)
        creation_time = time.time() - start_time
        
        # Validate cache was created
        metadata_file = os.path.join(test_output_dir, "cache_metadata.json")
        assert os.path.exists(metadata_file), "Metadata file not created"
        
        import glob
        chunk_files = glob.glob(os.path.join(test_output_dir, "packed_chunk_*.pt"))
        assert len(chunk_files) > 0, "No chunk files created"
        
        print(f"✅ Cache created successfully")
        print(f"   Creation time: {creation_time:.2f}s")
        print(f"   Chunk files: {len(chunk_files)}")
        print(f"   Metadata: {os.path.exists(metadata_file)}")
        
        return test_output_dir

def test_cache_loading():
    """Test 2: Cache loading and validation"""
    print("\n🧪 TEST 2: Cache Loading")
    print("=" * 40)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test cache
        test_input_dir = os.path.join(temp_dir, "input")
        test_output_dir = os.path.join(temp_dir, "output")
        
        create_test_data(test_input_dir, num_samples=200)
        
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        creator = PackedCacheCreator(tokenizer, max_length=512, use_compression=False)
        creator.create_cache(test_input_dir, test_output_dir, chunk_size=100)
        
        # Test loading
        start_time = time.time()
        dataset = PackedSequenceDataset(test_output_dir, device='cpu')
        loading_time = time.time() - start_time
        
        # Validate dataset
        assert len(dataset) > 0, "Dataset is empty"
        
        # Test random access
        sample = dataset[0]
        assert 'input_ids' in sample, "Missing input_ids"
        assert sample['input_ids'].shape[0] == 512, f"Wrong sequence length: {sample['input_ids'].shape}"
        
        # Test multiple samples
        for i in [0, len(dataset)//2, len(dataset)-1]:
            sample = dataset[i]
            assert sample['input_ids'].shape[0] == 512, f"Wrong length at index {i}"
        
        print(f"✅ Cache loaded successfully")
        print(f"   Loading time: {loading_time:.2f}s")
        print(f"   Dataset size: {len(dataset):,}")
        print(f"   Sequence length: {sample['input_ids'].shape[0]}")
        print(f"   Utilization: {dataset.metadata['utilization_stats']['avg_utilization']:.1f}%")

def test_dataloader_integration():
    """Test 3: DataLoader integration"""
    print("\n🧪 TEST 3: DataLoader Integration")
    print("=" * 40)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test cache
        test_input_dir = os.path.join(temp_dir, "input")
        test_output_dir = os.path.join(temp_dir, "output")
        
        create_test_data(test_input_dir, num_samples=300)
        
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        creator = PackedCacheCreator(tokenizer, max_length=512, use_compression=False)
        creator.create_cache(test_input_dir, test_output_dir, chunk_size=150)
        
        # Create DataLoader
        start_time = time.time()
        dataloader = create_packed_dataloader(test_output_dir, batch_size=8, device='cpu', num_workers=0)
        dataloader_time = time.time() - start_time
        
        # Test iteration
        batch_count = 0
        total_samples = 0
        
        iteration_start = time.time()
        for batch in dataloader:
            batch_count += 1
            total_samples += batch['input_ids'].shape[0]
            
            # Validate batch
            assert batch['input_ids'].shape[1] == 512, f"Wrong sequence length in batch"
            assert batch['input_ids'].shape[0] == 8, f"Wrong batch size: {batch['input_ids'].shape[0]}"
            
            if batch_count >= 5:  # Test first 5 batches
                break
        
        iteration_time = time.time() - iteration_start
        
        print(f"✅ DataLoader working correctly")
        print(f"   DataLoader creation: {dataloader_time:.2f}s")
        print(f"   Iteration time (5 batches): {iteration_time:.2f}s")
        print(f"   Batches tested: {batch_count}")
        print(f"   Samples processed: {total_samples}")
        print(f"   Speed: {total_samples/iteration_time:.0f} samples/sec")

def test_resume_capability():
    """Test 4: Resume capability"""
    print("\n🧪 TEST 4: Resume Capability")
    print("=" * 40)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data
        test_input_dir = os.path.join(temp_dir, "input")
        test_output_dir = os.path.join(temp_dir, "output")
        
        create_test_data(test_input_dir, num_samples=400)
        
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # First run - create partial cache
        creator1 = PackedCacheCreator(tokenizer, max_length=512, use_compression=False)
        
        # Simulate interruption by creating only first chunk manually
        import glob
        from optimized_sequence_packing import streaming_pack_sequences_pandas, reset_global_carry
        
        reset_global_carry()
        packed_chunks = streaming_pack_sequences_pandas(
            cache_dir=test_input_dir,
            tokenizer=tokenizer,
            max_length=512,
            chunk_size=200
        )
        
        # Save only first chunk
        os.makedirs(test_output_dir, exist_ok=True)
        first_chunk = packed_chunks[0]
        chunk_data = {
            'input_ids': first_chunk['input_ids'],
            'metadata': {
                'num_sequences': len(first_chunk['input_ids']),
                'chunk_id': 0
            }
        }
        torch.save(chunk_data, os.path.join(test_output_dir, "packed_chunk_000000.pt"))
        
        print(f"   Created partial cache with 1 chunk")
        
        # Second run - should resume
        creator2 = PackedCacheCreator(tokenizer, max_length=512, use_compression=False)
        creator2.create_cache(test_input_dir, test_output_dir, chunk_size=200)
        
        # Validate complete cache
        chunk_files = glob.glob(os.path.join(test_output_dir, "packed_chunk_*.pt"))
        metadata_file = os.path.join(test_output_dir, "cache_metadata.json")
        
        assert len(chunk_files) >= 2, f"Resume failed - only {len(chunk_files)} chunks"
        assert os.path.exists(metadata_file), "Metadata not created after resume"
        
        print(f"✅ Resume capability working")
        print(f"   Final chunks: {len(chunk_files)}")
        print(f"   Metadata created: {os.path.exists(metadata_file)}")

def test_performance_benchmark():
    """Test 5: Performance benchmark"""
    print("\n🧪 TEST 5: Performance Benchmark")
    print("=" * 40)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create larger test dataset
        test_input_dir = os.path.join(temp_dir, "input")
        test_output_dir = os.path.join(temp_dir, "output")
        
        create_test_data(test_input_dir, num_samples=1000)
        
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Benchmark cache creation
        creator = PackedCacheCreator(tokenizer, max_length=512, use_compression=False)
        
        creation_start = time.time()
        creator.create_cache(test_input_dir, test_output_dir, chunk_size=500)
        creation_time = time.time() - creation_start
        
        # Benchmark loading
        loading_start = time.time()
        dataset = PackedSequenceDataset(test_output_dir, device='cpu')
        loading_time = time.time() - loading_start
        
        # Benchmark iteration
        dataloader = create_packed_dataloader(test_output_dir, batch_size=16, device='cpu', num_workers=0)
        
        iteration_start = time.time()
        sample_count = 0
        for i, batch in enumerate(dataloader):
            sample_count += batch['input_ids'].shape[0]
            if i >= 10:  # Test 10 batches
                break
        iteration_time = time.time() - iteration_start
        
        # Calculate rates
        creation_rate = 1000 / creation_time
        loading_rate = len(dataset) / loading_time if loading_time > 0 else float('inf')
        iteration_rate = sample_count / iteration_time if iteration_time > 0 else float('inf')
        
        print(f"✅ Performance benchmark complete")
        print(f"   Cache creation: {creation_time:.2f}s ({creation_rate:.0f} samples/sec)")
        print(f"   Cache loading: {loading_time:.2f}s ({loading_rate:.0f} samples/sec)")
        print(f"   Iteration: {iteration_time:.2f}s ({iteration_rate:.0f} samples/sec)")
        print(f"   Utilization: {dataset.metadata['utilization_stats']['avg_utilization']:.1f}%")

def run_all_tests():
    """Run complete test suite"""
    print("🚀 PACKED CACHE SYSTEM TESTS")
    print("=" * 60)
    
    tests = [
        test_cache_creation,
        test_cache_loading,
        test_dataloader_integration,
        test_resume_capability,
        test_performance_benchmark
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
            print(f"   ✅ PASSED")
        except Exception as e:
            failed += 1
            print(f"   ❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n📊 TEST RESULTS:")
    print(f"   Passed: {passed}/{len(tests)}")
    print(f"   Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print(f"   🎉 ALL TESTS PASSED!")
        return True
    else:
        print(f"   ⚠️  Some tests failed")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
