#!/usr/bin/env python3
"""
🚀 SEQUENCE PACKING CACHE CREATOR
Pre-process sequences once, use forever!

Usage:
    python create_packed_cache.py --input_dir cache/fineweb --output_dir cache/packed_sequences

Features:
- Resume capability (crash-safe)
- LZ4 compression (optional)
- Progress tracking
- Integrity checks
- GPU optimization
"""

import argparse
import os
import sys
import time
import glob
from transformers import AutoTokenizer

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

from sequence_packing_cache import PackedCacheCreator
from cache_coordination import CacheCoordinator, create_coordination_info
from unified_metadata_manager import UnifiedMetadataManager

def analyze_parquet_files(input_dir: str) -> dict:
    """
    Analyze all parquet files in the input directory.

    Returns:
        Dict with file analysis information
    """
    parquet_pattern = os.path.join(input_dir, "*.parquet")
    parquet_files = sorted(glob.glob(parquet_pattern))

    if not parquet_files:
        return {"total_files": 0, "files": [], "first_file": None, "last_file": None}

    analysis = {
        "total_files": len(parquet_files),
        "files": parquet_files,
        "first_file": os.path.basename(parquet_files[0]),
        "last_file": os.path.basename(parquet_files[-1]),
        "file_range": f"{os.path.basename(parquet_files[0])} → {os.path.basename(parquet_files[-1])}"
    }

    print(f"📊 PARQUET FILE ANALYSIS:")
    print(f"   Total files: {analysis['total_files']:,}")
    print(f"   First file: {analysis['first_file']}")
    print(f"   Last file: {analysis['last_file']}")
    print(f"   Range: {analysis['file_range']}")

    return analysis

def parse_args():
    parser = argparse.ArgumentParser(description='Create packed sequence cache')
    
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory with parquet files (e.g., cache/fineweb)')
    
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for packed cache (e.g., cache/packed_sequences)')
    
    parser.add_argument('--tokenizer', type=str, default='HuggingFaceTB/SmolLM-135M',
                       help='Tokenizer to use (default: HuggingFaceTB/SmolLM-135M)')
    
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length (default: 512)')
    
    parser.add_argument('--chunk_size', type=int, default=10000,
                       help='Chunk size for processing (default: 10000)')
    
    parser.add_argument('--no_compression', action='store_true',
                       help='Disable LZ4 compression')
    
    parser.add_argument('--force', action='store_true',
                       help='Force recreation (delete existing cache)')
    
    parser.add_argument('--dry_run', action='store_true',
                       help='Show what would be done without actually doing it')
    
    return parser.parse_args()

def validate_inputs(args):
    """Validate input arguments"""
    
    # Check input directory
    if not os.path.exists(args.input_dir):
        print(f"❌ Input directory not found: {args.input_dir}")
        return False
    
    # Check for parquet files
    import glob
    parquet_files = glob.glob(os.path.join(args.input_dir, "*.parquet"))
    if not parquet_files:
        print(f"❌ No parquet files found in: {args.input_dir}")
        return False
    
    print(f"✅ Found {len(parquet_files)} parquet files in input directory")
    
    # Check output directory
    if os.path.exists(args.output_dir):
        if args.force:
            print(f"🗑️  Force mode: Will recreate cache in {args.output_dir}")
            import shutil
            if not args.dry_run:
                shutil.rmtree(args.output_dir)
        else:
            # 🎯 FIXED: Check in the correct subdirectory structure
            final_output_dir = os.path.join(args.output_dir, "512", "FineWeb")
            existing_chunks = glob.glob(os.path.join(final_output_dir, "packed_chunk_*.pt"))
            if existing_chunks:
                print(f"📦 Found {len(existing_chunks)} existing chunks - will resume")
            else:
                print(f"📁 Output directory structure ready - will create cache")
    
    # Check compression availability
    if not args.no_compression:
        try:
            import lz4.frame
            print("✅ LZ4 compression available")
        except ImportError:
            print("⚠️  LZ4 not available - will use uncompressed storage")
            print("   Install with: pip install lz4")
    
    return True

def estimate_requirements(args):
    """Estimate time and space requirements"""
    
    # Sample a few files to estimate
    import glob
    import pandas as pd
    
    parquet_files = glob.glob(os.path.join(args.input_dir, "*.parquet"))
    
    print("\n📊 Estimating requirements...")
    
    # Sample first file
    try:
        sample_df = pd.read_parquet(parquet_files[0])
        if 'text' in sample_df.columns:
            sample_texts = sample_df['text'].head(1000).tolist()
            
            # Estimate total samples
            total_samples = len(sample_df) * len(parquet_files)
            
            # Estimate processing time (based on our benchmarks)
            samples_per_hour = 500_000  # Conservative estimate
            estimated_hours = total_samples / samples_per_hour
            
            # Estimate storage (conservative)
            avg_tokens_per_sample = args.max_length * 0.99  # 99% utilization
            bytes_per_token = 4  # int32
            total_bytes = total_samples * avg_tokens_per_sample * bytes_per_token
            
            if not args.no_compression:
                total_bytes *= 0.7  # LZ4 compression ratio
            
            total_gb = total_bytes / (1024**3)
            
            print(f"   📈 Estimated samples: {total_samples:,}")
            print(f"   ⏱️  Estimated time: {estimated_hours:.1f} hours")
            print(f"   💾 Estimated storage: {total_gb:.1f} GB")
            
            if estimated_hours > 48:
                print(f"   ⚠️  Very long processing time - consider smaller chunks")
            
            if total_gb > 100:
                print(f"   ⚠️  Large storage requirement - ensure sufficient disk space")
                
        else:
            print("   ⚠️  Cannot estimate - no 'text' column in sample file")
            
    except Exception as e:
        print(f"   ⚠️  Cannot estimate requirements: {e}")

def main():
    print("🚀 SEQUENCE PACKING CACHE CREATOR")
    print("=" * 50)
    
    args = parse_args()
    
    # Validate inputs
    if not validate_inputs(args):
        sys.exit(1)

    # Setup coordination and analyze files
    coordination_info = create_coordination_info(args.input_dir, args.output_dir)
    if coordination_info["total_files"] == 0:
        print("❌ No parquet files found!")
        sys.exit(1)

    # Setup coordinator
    coordinator = CacheCoordinator(args.output_dir)

    # Estimate requirements
    estimate_requirements(args)
    
    if args.dry_run:
        print("\n🔍 DRY RUN - No actual processing will be done")
        print(f"   Input: {args.input_dir}")
        print(f"   Output: {args.output_dir}")
        print(f"   Tokenizer: {args.tokenizer}")
        print(f"   Max length: {args.max_length}")
        print(f"   Chunk size: {args.chunk_size}")
        print(f"   Compression: {not args.no_compression}")
        return
    
    # Confirm before starting
    print(f"\n🎯 Ready to create packed cache:")
    print(f"   Input: {args.input_dir}")
    print(f"   Output: {args.output_dir}")
    print(f"   Tokenizer: {args.tokenizer}")
    print(f"   Max length: {args.max_length}")
    print(f"   Compression: {not args.no_compression}")
    
    response = input("\nProceed? [y/N]: ").strip().lower()
    if response != 'y':
        print("❌ Cancelled by user")
        sys.exit(0)
    
    # Load tokenizer
    print(f"\n📥 Loading tokenizer: {args.tokenizer}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"✅ Tokenizer loaded: {len(tokenizer)} tokens")
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        sys.exit(1)
    
    # Create cache creator
    creator = PackedCacheCreator(
        tokenizer=tokenizer,
        max_length=args.max_length,
        use_compression=not args.no_compression
    )
    
    # Start processing
    start_time = time.time()
    
    try:
        # Initialize metadata manager
        final_cache_dir = os.path.join(args.output_dir, "512", "FineWeb")
        metadata_manager = UnifiedMetadataManager(final_cache_dir)

        # Create forward process metadata
        chunk_range = (coordination_info['forward_range'][0], coordination_info['forward_range'][1])
        metadata_manager.create_process_metadata('forward', chunk_range, args.max_length, tokenizer.name_or_path)

        creator.create_cache(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            chunk_size=args.chunk_size,
            coordination_info=coordination_info
        )

        # Finalize forward process metadata
        metadata_manager.finalize_process_metadata('forward')

        # Create unified metadata
        metadata_manager.create_unified_metadata()

        total_time = time.time() - start_time

        print(f"\n🎉 SUCCESS!")
        print(f"   Total time: {total_time/3600:.1f} hours")
        print(f"   Cache location: {args.output_dir}")
        print(f"   Ready for training!")
        
        # Show usage example
        print(f"\n💡 Usage in training:")
        print(f"   from sequence_packing_cache import create_packed_dataloader")
        print(f"   dataloader = create_packed_dataloader('{args.output_dir}', batch_size=32)")
        
    except KeyboardInterrupt:
        print(f"\n⏸️  Interrupted by user")
        print(f"   Progress saved - you can resume by running the same command")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        print(f"   You can resume by running the same command")
        sys.exit(1)

if __name__ == "__main__":
    main()
