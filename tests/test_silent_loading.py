#!/usr/bin/env python3
"""
🧪 Test Silent Loading für Training
Zeigt wie das Dataset Loading im Training aussieht
"""

from fast_dataset_loader import load_samples_fast

def test_silent_vs_verbose():
    print("🧪 Testing Silent vs Verbose Loading")
    print("=" * 60)
    
    print("\n📢 VERBOSE MODE (für Tests):")
    print("-" * 40)
    dataset_verbose = load_samples_fast(10000, verbose=True)
    
    print("\n🔇 SILENT MODE (für Training):")
    print("-" * 40)
    dataset_silent = load_samples_fast(10000, verbose=False)
    
    print("\n✅ Both modes loaded successfully!")
    print(f"   Verbose: {len(dataset_verbose):,} samples")
    print(f"   Silent: {len(dataset_silent):,} samples")

if __name__ == "__main__":
    test_silent_vs_verbose()
