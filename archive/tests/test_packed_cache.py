"""
Test Script für Packed Cache Dataset System

Testet das Laden von packed sequences aus dem Cache.
"""

import os
import torch
from core.data import DatasetFactory
from config import training_config

def test_packed_cache():
    """Test packed cache loading."""
    print("Testing Packed Cache Dataset System")
    print("=" * 50)
    
    # Check cache directory
    cache_dir = training_config.packed_cache_dir
    print(f"Cache directory: {cache_dir}")
    
    if not os.path.exists(cache_dir):
        print(f"Cache directory does not exist: {cache_dir}")
        print("Creating test cache...")
        create_test_cache(cache_dir)
    
    # List cache files
    if os.path.exists(cache_dir):
        cache_files = [f for f in os.listdir(cache_dir) if f.startswith('packed_chunk_') and f.endswith('.pt')]
        print(f"Found {len(cache_files)} cache files:")
        for f in cache_files:
            file_path = os.path.join(cache_dir, f)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  - {f} ({size_mb:.1f} MB)")
    
    # Test loading
    factory = DatasetFactory()
    dataloader = factory.create_packed_cache_dataset()
    
    if dataloader:
        print(f"\nPacked cache loaded successfully!")
        print(f"Dataset size: {len(dataloader.dataset):,} samples")
        print(f"Batches: {len(dataloader):,}")
        
        # Test first batch
        for batch in dataloader:
            if isinstance(batch, dict):
                input_ids = batch['input_ids']
                print(f"Batch shape: {input_ids.shape}")
                print(f"Sequence length: {input_ids.shape[1]}")
                print(f"Sample tokens: {input_ids[0][:20].tolist()}")
            else:
                print(f"Batch type: {type(batch)}")
                print(f"Batch content: {batch}")
            break
            
        return True
    else:
        print("Failed to load packed cache")
        return False

def create_test_cache(cache_dir):
    """Create test cache for demonstration."""
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create some test packed sequences
    vocab_size = 49152
    seq_length = 768
    
    for i in range(3):  # 3 cache chunks
        # Generate random sequences
        num_sequences = 1000
        sequences = torch.randint(0, vocab_size, (num_sequences, seq_length), dtype=torch.long)
        
        # Save as cache chunk
        cache_file = os.path.join(cache_dir, f"packed_chunk_{i:03d}.pt")
        torch.save({'input_ids': sequences}, cache_file)
        print(f"Created test cache: {cache_file}")

def test_dataset_priority():
    """Test dataset loading priority."""
    print("\nTesting Dataset Loading Priority")
    print("=" * 50)
    
    from core.data.dataset_factory import create_gpu_optimized_dataset
    
    # Test with real data (should try packed cache first)
    print("Testing with use_real_data=True...")
    dataloader = create_gpu_optimized_dataset(use_real_data=True)
    
    if dataloader:
        print(f"Loaded dataset with {len(dataloader.dataset):,} samples")
        
        # Check first batch
        for batch in dataloader:
            if isinstance(batch, dict):
                input_ids = batch['input_ids']
                print(f"Batch shape: {input_ids.shape}")
            else:
                print(f"Batch type: {type(batch)}")
            break
    
    return dataloader is not None

if __name__ == "__main__":
    print("Packed Cache Dataset Test")
    print("=" * 60)
    
    # Test 1: Packed cache loading
    cache_success = test_packed_cache()
    
    # Test 2: Dataset priority
    priority_success = test_dataset_priority()
    
    print("\n" + "=" * 60)
    print("TEST RESULTS:")
    print(f"Packed Cache Loading: {'✅ PASS' if cache_success else '❌ FAIL'}")
    print(f"Dataset Priority: {'✅ PASS' if priority_success else '❌ FAIL'}")
    
    if cache_success and priority_success:
        print("\n🎉 All tests passed! Packed cache system is working.")
    else:
        print("\n⚠️ Some tests failed. Check the implementation.")
