#!/usr/bin/env python3
"""
🧪 Test Fast Dataset Loader
Testet den neuen schnellen Dataset Loader
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fast_dataset_loader import load_samples_fast, find_cached_files_fast, create_fast_dataloader

def test_find_cached_files():
    """Test das Finden von gecachten Dateien."""
    print("🧪 Testing cached file discovery...")
    
    cached_files = find_cached_files_fast()
    
    if cached_files:
        print(f"   ✅ Found {len(cached_files)} cached files")
        print(f"   📁 First file: {os.path.basename(cached_files[0])}")
        return True
    else:
        print("   ⚠️  No cached files found")
        print("   💡 This is expected if no cache/fineweb directory exists")
        return False

def test_load_small_sample():
    """Test das Laden einer kleinen Sample-Anzahl."""
    print("🧪 Testing small sample loading...")
    
    try:
        dataset = load_samples_fast(1000)
        
        if dataset and len(dataset) > 0:
            print(f"   ✅ Loaded {len(dataset):,} samples")
            
            # Test Datenstruktur
            sample = dataset[0]
            required_keys = ['text', 'id', 'language']
            
            for key in required_keys:
                if key in sample:
                    print(f"   ✅ Key '{key}' present")
                else:
                    print(f"   ⚠️  Key '{key}' missing")
            
            # Test Text-Inhalt
            text = sample.get('text', '')
            if len(text) > 0:
                print(f"   ✅ Text content: {len(text)} characters")
                print(f"   📝 Preview: {text[:100]}...")
            else:
                print(f"   ❌ Empty text content")
                return False
            
            return True
        else:
            print("   ❌ Failed to load dataset or empty dataset")
            return False
            
    except Exception as e:
        print(f"   ❌ Error loading dataset: {e}")
        return False

def test_dataset_structure():
    """Test die Struktur des geladenen Datasets."""
    print("🧪 Testing dataset structure...")
    
    try:
        dataset = load_samples_fast(100)  # Kleine Anzahl für schnellen Test
        
        if not dataset:
            print("   ⚠️  No dataset loaded - skipping structure test")
            return False
        
        # Test verschiedene Samples
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            
            # Test Datentypen
            text = sample.get('text', '')
            if not isinstance(text, str):
                print(f"   ❌ Sample {i}: text is not string")
                return False
            
            language = sample.get('language', '')
            if language != 'en':
                print(f"   ⚠️  Sample {i}: language is '{language}', expected 'en'")
            
            score = sample.get('score', 0)
            if not isinstance(score, (int, float)):
                print(f"   ❌ Sample {i}: score is not numeric")
                return False
        
        print(f"   ✅ Dataset structure is valid")
        print(f"   📊 Tested {min(3, len(dataset))} samples")
        return True
        
    except Exception as e:
        print(f"   ❌ Error testing structure: {e}")
        return False

def test_performance():
    """Test die Performance des Loaders."""
    print("🧪 Testing loader performance...")
    
    import time
    
    try:
        # Test verschiedene Größen
        test_sizes = [1000, 5000, 10000]
        
        for size in test_sizes:
            start_time = time.time()
            dataset = load_samples_fast(size)
            elapsed = time.time() - start_time
            
            if dataset:
                speed = len(dataset) / elapsed if elapsed > 0 else 0
                print(f"   📊 {size:,} samples: {elapsed:.2f}s ({speed:,.0f} samples/sec)")
            else:
                print(f"   ❌ Failed to load {size:,} samples")
                return False
        
        print(f"   ✅ Performance test completed")
        return True
        
    except Exception as e:
        print(f"   ❌ Performance test failed: {e}")
        return False

def test_dataloader_creation():
    """Test die DataLoader-Erstellung."""
    print("🧪 Testing DataLoader creation...")
    
    try:
        result = create_fast_dataloader(num_samples=1000, with_splits=False)
        
        if result and 'full_dataset' in result:
            dataset = result['full_dataset']
            print(f"   ✅ DataLoader created with {len(dataset):,} samples")
            return True
        else:
            print("   ❌ Failed to create DataLoader")
            return False
            
    except Exception as e:
        print(f"   ❌ DataLoader creation failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Fast Dataset Loader")
    print("=" * 60)
    
    tests = [
        test_find_cached_files,
        test_load_small_sample,
        test_dataset_structure,
        test_performance,
        test_dataloader_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print()
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"   ❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        print("🚀 Fast dataset loader is working correctly")
    else:
        print("⚠️  Some tests failed")
        print("💡 Check if cache/fineweb directory exists with parquet files")
