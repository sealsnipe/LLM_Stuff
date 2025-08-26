#!/usr/bin/env python3
"""
🧪 Test Script für FineWeb-Edu Dataset
Testet Dataset Loading und zeigt Beispiele
"""

from dataset_loader import create_fineweb_dataloader, test_dataset_loading, show_dataset_configs
from transformers import AutoTokenizer
import torch
from config import dataset_config, training_config

def analyze_dataset_quality():
    """Analysiert die Qualität des FineWeb-Edu Datasets."""
    print("🔍 FineWeb-Edu Dataset Quality Analysis")
    print("=" * 50)
    
    try:
        # Lade kleinen Sample
        dataloader = create_fineweb_dataloader(
            dataset_size="tiny",
            batch_size=1,
            streaming=False
        )
        
        tokenizer = AutoTokenizer.from_pretrained(dataset_config.default_tokenizer)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("📝 Sample Texts from FineWeb-Edu:")
        print("-" * 50)
        
        # Analysiere erste 5 Samples
        for i, batch in enumerate(dataloader):
            if i >= 5:
                break
                
            input_ids = batch['input_ids'].squeeze()
            text = tokenizer.decode(input_ids, skip_special_tokens=True)
            
            # Entferne Padding
            text = text.replace(tokenizer.pad_token, "").strip()
            
            print(f"\n📄 Sample {i+1}:")
            print(f"   Length: {len(text)} chars")
            print(f"   Tokens: {(input_ids != tokenizer.pad_token_id).sum().item()}")
            print(f"   Preview: {text[:200]}...")
            
            # Einfache Qualitäts-Checks
            has_punctuation = any(p in text for p in '.!?')
            has_uppercase = any(c.isupper() for c in text)
            words = text.split()
            avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
            
            print(f"   Quality Indicators:")
            print(f"     - Has punctuation: {has_punctuation}")
            print(f"     - Has uppercase: {has_uppercase}")
            print(f"     - Avg word length: {avg_word_length:.1f}")
            print(f"     - Word count: {len(words)}")
            
    except Exception as e:
        print(f"❌ Quality analysis failed: {e}")
        print("💡 This might be due to internet connection or dataset access issues")

def compare_datasets():
    """Vergleicht verschiedene Dataset-Größen."""
    print("\n📊 Dataset Size Comparison")
    print("=" * 50)
    
    sizes = ["tiny", "small"]  # Nur kleine Sizes für Test
    
    for size in sizes:
        print(f"\n🎯 Testing {size.upper()} dataset...")
        
        try:
            dataloader = create_fineweb_dataloader(
                dataset_size=size,
                batch_size=4,
                streaming=False
            )
            
            # Teste ersten Batch
            batch = next(iter(dataloader))
            
            print(f"   ✅ Loaded successfully")
            if size in dataset_config.dataset_sizes:
                config_info = dataset_config.dataset_sizes[size]
                print(f"   Samples: {config_info['num_samples']}")
                print(f"   Description: {config_info['description']}")
            print(f"   Batch shape: {batch['input_ids'].shape}")
            print(f"   Memory usage: ~{batch['input_ids'].numel() * 4 / 1e6:.1f} MB")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")

def test_tokenization():
    """Testet die Tokenization."""
    print("\n🔤 Tokenization Test")
    print("=" * 50)
    
    tokenizer = AutoTokenizer.from_pretrained(dataset_config.default_tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test Texte
    test_texts = [
        "Hello world! This is a test.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating.",
        "Python is a great programming language.",
        "FineWeb-Edu contains high-quality educational content."
    ]
    
    print(f"Tokenizer: {tokenizer.name_or_path}")
    print(f"Vocab Size: {tokenizer.vocab_size}")
    print(f"Pad Token: {tokenizer.pad_token}")
    print(f"EOS Token: {tokenizer.eos_token}")
    print(f"Sequence Length: {training_config.sequence_length}")
    print()
    
    for i, text in enumerate(test_texts):
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        
        print(f"Text {i+1}: {text}")
        print(f"  Tokens: {len(tokens)}")
        print(f"  Token IDs: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
        print(f"  Decoded: {decoded}")
        print()

def estimate_training_time():
    """Schätzt Trainingszeiten für verschiedene Dataset-Größen."""
    print("\n⏱️  Training Time Estimation")
    print("=" * 50)
    
    # Basis-Annahmen (RTX 4070 Ti / RTX 3090)
    steps_per_second = 1.2  # Geschätzt für unser Modell
    tokens_per_sample = 200  # Durchschnitt
    
    print(f"Assumptions (RTX 4070 Ti / RTX 3090):")
    print(f"  Steps/second: {steps_per_second}")
    print(f"  Tokens/sample: {tokens_per_sample}")
    print(f"  Batch size: {training_config.batch_size}")
    print(f"  Gradient accumulation: {training_config.gradient_accumulation_steps}")
    print(f"  Effective batch size: {training_config.batch_size * training_config.gradient_accumulation_steps}")
    print()
    
    for name, config in dataset_config.dataset_sizes.items():
        samples = config["num_samples"]
        if samples is None:
            print(f"{name.upper()}: Full dataset - too large to estimate")
            continue
            
        # Berechne Steps (samples / effective_batch_size)
        effective_batch_size = training_config.batch_size * training_config.gradient_accumulation_steps
        steps = samples // effective_batch_size
        
        # Berechne Zeit
        seconds = steps / steps_per_second
        minutes = seconds / 60
        hours = minutes / 60
        
        # Berechne Tokens
        total_tokens = samples * tokens_per_sample
        
        print(f"{name.upper()}:")
        print(f"  Samples: {samples:,}")
        print(f"  Steps: {steps:,}")
        print(f"  Tokens: {total_tokens/1e6:.1f}M")
        if hours < 1:
            print(f"  Time: {minutes:.0f} minutes")
        else:
            print(f"  Time: {hours:.1f} hours")
        print()

def test_gpu_compatibility():
    """Testet GPU-Kompatibilität für Dataset Loading."""
    print("\n🖥️  GPU Compatibility Test")
    print("=" * 50)
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.version.cuda}")
        print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Test GPU memory with small batch
        try:
            device = torch.device("cuda")
            test_tensor = torch.randn(training_config.batch_size, training_config.sequence_length, device=device)
            print(f"✅ GPU memory test passed")
            print(f"📊 Test tensor shape: {test_tensor.shape}")
            print(f"💾 Memory used: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"❌ GPU memory test failed: {e}")
    else:
        print("❌ CUDA not available - will use CPU (slower)")
    
    # Check dependencies
    try:
        from datasets import load_dataset
        print("✅ datasets library available")
    except ImportError:
        print("❌ datasets library not found - install with: pip install datasets")
    
    try:
        from transformers import AutoTokenizer
        print("✅ transformers library available")
    except ImportError:
        print("❌ transformers library not found - install with: pip install transformers")

def main():
    """Hauptfunktion für alle Tests."""
    print("🧪 FineWeb-Edu Dataset Test Suite")
    print("=" * 60)
    
    # 1. GPU Compatibility Check
    print("\n1️⃣  GPU Compatibility Check")
    test_gpu_compatibility()
    
    # 2. Basis Dataset Test
    print("\n2️⃣  Basic Dataset Loading Test")
    success = test_dataset_loading()
    
    if not success:
        print("\n❌ Basic dataset loading failed. Skipping advanced tests.")
        print("💡 Check your internet connection and try again.")
        return
    
    # 3. Tokenization Test
    print("\n3️⃣  Tokenization Test")
    test_tokenization()
    
    # 4. Dataset Size Comparison
    print("\n4️⃣  Dataset Size Comparison")
    compare_datasets()
    
    # 5. Dataset Qualitäts-Analyse
    print("\n5️⃣  Dataset Quality Analysis")
    analyze_dataset_quality()
    
    # 6. Training Time Estimation
    print("\n6️⃣  Training Time Estimation")
    estimate_training_time()
    
    # 7. Verfügbare Konfigurationen
    print("\n7️⃣  Available Configurations")
    show_dataset_configs()
    
    print("\n✅ All tests completed!")
    print("\n🚀 Ready to start training with:")
    print("   python gpu_training_optimized.py")
    print("\n💡 Or test with different dataset sizes:")
    print("   # Edit gpu_training_optimized.py and change dataset_size parameter")

if __name__ == "__main__":
    main()
