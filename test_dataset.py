#!/usr/bin/env python3
"""
🧪 Quick Dataset Test
Testet ob das 27GB FineWeb-Edu Dataset korrekt geladen wird
"""

from dataset_loader import create_fineweb_dataloader

def test_dataset_loading():
    print("🧪 Testing FineWeb-Edu Dataset Loading...")
    print("   Dataset: sample-10BT (27GB)")
    print("   Samples: 100,000")
    print()
    
    try:
        # Test dataset loading
        dataloader = create_fineweb_dataloader(
            dataset_size='medium',
            num_samples=100000,
            batch_size=2
        )
        
        print("Dataset loading works!")
        print(f"   Dataloader created successfully")
        
        # Test first batch
        batch = next(iter(dataloader))
        print(f"   First batch shape: {batch['input_ids'].shape}")
        print(f"   Batch keys: {list(batch.keys())}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = test_dataset_loading()
    if success:
        print("\n✅ Dataset test successful!")
        print("🚀 Ready to run: python training.py")
    else:
        print("\n❌ Dataset test failed!")
