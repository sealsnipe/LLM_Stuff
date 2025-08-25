#!/usr/bin/env python3
"""
🧪 Test Cached Dataset Loading
Testet ob das gecachte FineWeb-Edu Dataset direkt geladen werden kann
"""

from dataset_loader import find_cached_fineweb_data, load_from_cached_parquet, create_fineweb_dataloader

def test_cached_files():
    print("🔍 Searching for cached FineWeb-Edu files...")
    
    cached_files = find_cached_fineweb_data()
    
    if cached_files:
        print(f"✅ Found {len(cached_files)} cached parquet files")
        for i, file in enumerate(cached_files[:5]):  # Show first 5
            print(f"   {i+1}. {file}")
        if len(cached_files) > 5:
            print(f"   ... and {len(cached_files) - 5} more")
        
        # Test loading
        print("\n📂 Testing direct parquet loading...")
        dataset = load_from_cached_parquet(cached_files, num_samples=1000)
        
        if dataset:
            print(f"✅ Successfully loaded {len(dataset)} samples")
            print(f"   Columns: {dataset.column_names}")
            if len(dataset) > 0:
                print(f"   Sample text: {dataset[0]['text'][:100]}...")
            return True
        else:
            print("❌ Failed to load from parquet")
            return False
    else:
        print("❌ No cached files found")
        return False

def test_dataloader():
    print("\n🚀 Testing complete dataloader...")
    
    try:
        dataloader = create_fineweb_dataloader(
            dataset_size='medium',
            num_samples=1000,
            batch_size=2
        )
        
        print("✅ Dataloader created successfully")
        
        # Test first batch
        batch = next(iter(dataloader))
        print(f"   Batch shape: {batch['input_ids'].shape}")
        print(f"   Batch keys: {list(batch.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataloader test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Cached FineWeb-Edu Dataset")
    print("=" * 50)
    
    # Test 1: Find cached files
    cached_success = test_cached_files()
    
    # Test 2: Full dataloader
    dataloader_success = test_dataloader()
    
    print("\n" + "=" * 50)
    if cached_success and dataloader_success:
        print("✅ All tests passed!")
        print("🚀 Ready to run: python training.py")
    else:
        print("❌ Some tests failed")
        if not cached_success:
            print("   - Cached file loading failed")
        if not dataloader_success:
            print("   - Dataloader creation failed")
