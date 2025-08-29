#!/usr/bin/env python3
"""
🔄 REVERSE SEQUENCE PACKING CACHE CREATOR
Starts from the LAST file and works backwards - perfect for helper machines!

Usage:
    python create_packed_cache_reverse.py --input_dir cache/fineweb --output_dir cache/packed_sequences

Features:
- Starts from last parquet file (014_xxxxx.parquet)
- Works backwards to avoid conflicts with main machine
- Same resume capability and compression
- Automatic chunk numbering to avoid conflicts
"""

import argparse
import os
import sys
import time
import glob
from typing import List
from transformers import AutoTokenizer

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

from sequence_packing_cache import PackedCacheCreator
from cache_coordination import CacheCoordinator, create_coordination_info
from core.data.sequence_packing import reset_global_carry

def analyze_parquet_files_reverse(input_dir: str) -> dict:
    """
    Analyze all parquet files for reverse processing.

    Returns:
        Dict with file analysis information for reverse processing
    """
    parquet_pattern = os.path.join(input_dir, "*.parquet")
    parquet_files = sorted(glob.glob(parquet_pattern))

    if not parquet_files:
        return {"total_files": 0, "files": [], "first_file": None, "last_file": None}

    # For reverse processing, we reverse the order
    reverse_files = list(reversed(parquet_files))

    analysis = {
        "total_files": len(parquet_files),
        "files": parquet_files,
        "reverse_files": reverse_files,
        "first_file": os.path.basename(parquet_files[0]),
        "last_file": os.path.basename(parquet_files[-1]),
        "reverse_first": os.path.basename(reverse_files[0]),  # Last file in normal order
        "reverse_last": os.path.basename(reverse_files[-1]),  # First file in normal order
        "file_range": f"{os.path.basename(reverse_files[0])} → {os.path.basename(reverse_files[-1])}"
    }

    print(f"📊 PARQUET FILE ANALYSIS (REVERSE MODE):")
    print(f"   Total files: {analysis['total_files']:,}")
    print(f"   🔄 Reverse processing order:")
    print(f"      Starting with: {analysis['reverse_first']} (normally last)")
    print(f"      Ending with: {analysis['reverse_last']} (normally first)")
    print(f"   🔄 Reverse range: {analysis['file_range']}")

    return analysis

class ReverseCacheCreator(PackedCacheCreator):
    """
    🔄 REVERSE: Cache creator that starts from the last file
    """

    def _find_existing_reverse_chunks(self, output_dir: str) -> List[str]:
        """Find existing REVERSE chunk files (starting from 010000+)"""
        pattern = os.path.join(output_dir, "packed_chunk_01*.pt")
        existing = sorted(glob.glob(pattern))
        return existing

    def _find_existing_reverse_chunks_in_range(self, output_dir: str, start_num: int, end_num: int) -> List[str]:
        """Find existing reverse chunk files within a specific coordinated range"""
        pattern = os.path.join(output_dir, "packed_chunk_*.pt")
        all_chunks = sorted(glob.glob(pattern))

        # Filter chunks within the specified reverse range
        range_chunks = []
        for chunk_file in all_chunks:
            # Extract chunk number from filename
            basename = os.path.basename(chunk_file)
            chunk_num_str = basename.split('_')[2].split('.')[0]
            try:
                chunk_num = int(chunk_num_str)
                if start_num <= chunk_num <= end_num:
                    range_chunks.append(chunk_file)
            except ValueError:
                continue

        return range_chunks

    def _calculate_reverse_resume_position(self, parquet_files: List[str], chunk_size: int, existing_chunks: int) -> tuple:
        """Calculate which file and batch to resume from based on existing REVERSE chunks"""
        if existing_chunks == 0:
            return 0, 1  # Start from beginning

        # Calculate total batches processed so far
        total_batches_processed = existing_chunks
        current_batch_count = 0

        # Go through files in REVERSE order (same as processing order)
        for file_idx, file_path in enumerate(parquet_files):
            try:
                import pandas as pd
                df = pd.read_parquet(file_path)
                if 'text' not in df.columns:
                    continue

                texts = df['text'].tolist()
                file_batches = (len(texts) + chunk_size - 1) // chunk_size

                if current_batch_count + file_batches >= total_batches_processed:
                    # Resume position is in this file
                    batches_into_file = total_batches_processed - current_batch_count
                    resume_batch = batches_into_file + 1  # Next batch to process
                    return file_idx, resume_batch

                current_batch_count += file_batches

            except Exception as e:
                print(f"⚠️  Error calculating resume position for {file_path}: {e}")
                continue

        # If we get here, all files are processed
        return len(parquet_files), 1
    
    def _streaming_pack_and_save_reverse(self, input_dir: str, output_dir: str, chunk_size: int, start_chunk: int, coordination_info: dict = None) -> int:
        """Stream processing BACKWARDS from last file"""
        from core.data.sequence_packing import pack_sequences_heap_optimized, reset_global_carry
        import pandas as pd
        
        # 🎯 FIXED: Create proper directory structure like normal script
        final_output_dir = os.path.join(output_dir, "512", "FineWeb")
        os.makedirs(final_output_dir, exist_ok=True)
        print(f"📁 Using output directory: {final_output_dir}")

        # Find all parquet files and REVERSE the order
        parquet_pattern = os.path.join(input_dir, "*.parquet")
        parquet_files = sorted(glob.glob(parquet_pattern), reverse=True)  # REVERSE!

        if not parquet_files:
            print(f"❌ No parquet files found in {input_dir}")
            return 0
        
        print(f"🔄 REVERSE packing from {len(parquet_files)} parquet files (last to first)")
        print(f"   Starting with: {os.path.basename(parquet_files[0])}")
        print(f"   Ending with: {os.path.basename(parquet_files[-1])}")
        
        # 🎯 Use coordination info for proper reverse chunk numbering
        if coordination_info:
            reverse_range = coordination_info["chunk_strategy"]["reverse_range"]
            reverse_start_base = reverse_range[0]  # e.g., 1400
            reverse_end = reverse_range[1]         # e.g., 2100
            print(f"🎯 Using coordinated reverse range: {reverse_start_base}-{reverse_end}")
        else:
            # Fallback to old logic
            total_files = len(parquet_files)
            estimated_chunks_per_file = 50
            estimated_total_chunks = total_files * estimated_chunks_per_file
            reverse_start_base = max(10000, estimated_total_chunks * 2)
            reverse_end = reverse_start_base + 1000
            print(f"⚠️  No coordination info - using fallback range: {reverse_start_base}-{reverse_end}")

        # Store for later use in metadata updates
        self._reverse_start_base = reverse_start_base

        print(f"🔢 Coordinated chunk numbering strategy:")
        print(f"   Reverse range: {reverse_start_base}-{reverse_end}")
        print(f"   This uses coordinated ranges to avoid conflicts")
        total_processed = 0

        # 🔄 RESUME LOGIC: Find existing chunks in the coordinated reverse range
        existing_reverse_chunks_in_range = self._find_existing_reverse_chunks_in_range(
            final_output_dir, reverse_start_base, reverse_end
        )
        existing_count = len(existing_reverse_chunks_in_range)
        chunk_counter = reverse_start_base + existing_count

        print(f"🔄 Found {existing_count} existing reverse chunks in range {reverse_start_base}-{reverse_end}")
        print(f"🎯 Next chunk number: {chunk_counter}")
        print(f"🔄 Calculating resume position for {existing_count} existing chunks...")
        resume_file_idx, resume_batch_num = self._calculate_reverse_resume_position(
            parquet_files, chunk_size, existing_count
        )
        print(f"🎯 Resume position: file {resume_file_idx+1}, batch {resume_batch_num}")

        print(f"🚀 Starting file processing loop...")
        for file_idx, file_path in enumerate(parquet_files):
            print(f"🔄 Checking file {file_idx+1}: {os.path.basename(file_path)}")
            # Skip files that are already processed
            if file_idx < resume_file_idx:
                print(f"📁 Skipping file {file_idx+1}/{len(parquet_files)}: {os.path.basename(file_path)} (already processed)")
                continue

            print(f"📁 Processing file {file_idx+1}/{len(parquet_files)}: {os.path.basename(file_path)} (REVERSE)")
            try:
                
                # Load file with Pandas
                df = pd.read_parquet(file_path)
                
                if 'text' not in df.columns:
                    print(f"\r📁 Processing file {file_idx+1}/{len(parquet_files)}: {os.path.basename(file_path)} - ⚠️ No 'text' column")
                    continue
                
                texts = df['text'].tolist()
                total_chunks = (len(texts) + chunk_size - 1) // chunk_size

                # Determine starting batch for this file
                start_batch = resume_batch_num if file_idx == resume_file_idx else 1

                # Process in chunks with immediate saving
                for chunk_start in range((start_batch-1) * chunk_size, len(texts), chunk_size):
                    chunk_end = min(chunk_start + chunk_size, len(texts))
                    chunk_texts = texts[chunk_start:chunk_end]
                    chunk_num = chunk_start // chunk_size + 1
                    
                    # Show batch progress (only create line once, then update inline)
                    if chunk_num == start_batch:
                        print(f" - Batch {chunk_num}/{total_chunks} (resuming)", end="", flush=True)
                    else:
                        print(f"\r - Batch {chunk_num}/{total_chunks}", end="", flush=True)
                    
                    # Pack this chunk (completely silently)
                    packed = pack_sequences_heap_optimized(chunk_texts, self.tokenizer, self.max_length, verbose=False)
                    
                    if packed['input_ids'].size(0) > 0:
                        # Save immediately with HIGH chunk numbers to avoid conflicts
                        chunk_file = os.path.join(final_output_dir, f"packed_chunk_{chunk_counter:06d}.pt")

                        chunk_data = {
                            'input_ids': packed['input_ids'],
                            'attention_mask': packed.get('attention_mask'),
                            'position_ids': packed.get('position_ids'),
                            'metadata': {
                                'utilization': self._calculate_utilization(packed['input_ids']),
                                'num_sequences': len(packed['input_ids']),
                                'chunk_id': chunk_counter,
                                'creation_time': time.time(),
                                'max_length': self.max_length,
                                'source_file': os.path.basename(file_path),
                                'batch_in_file': chunk_num,
                                'reverse_processed': True  # Mark as reverse processed
                            }
                        }

                        # DEBUG: Show what we're saving
                        print(f"\n💾 Saving chunk {chunk_counter:06d}: {packed['input_ids'].shape[0]} sequences")

                        # Save with compression and integrity check
                        file_hash = self._save_chunk(chunk_data, chunk_file)
                        chunk_counter += 1

                        # 🔄 UPDATE METADATA AFTER EACH CHUNK (REVERSE) - Fixed parameter
                        chunks_created = chunk_counter - max(start_chunk, self._reverse_start_base)
                        self._update_metadata_incremental(final_output_dir, chunks_created)
                    
                    total_processed += len(chunk_texts)
                
                # Clear batch line and move to next file
                print(f"\r{' ' * 50}")  # Clear the batch line
                
            except Exception as e:
                print(f"\r📁 Processing file {file_idx+1}/{len(parquet_files)}: {os.path.basename(file_path)} - ❌ Failed: {e}")
                continue
        
        return chunk_counter - max(start_chunk, 10000)
    
    def create_cache_reverse(self, input_dir: str, output_dir: str, chunk_size: int = 10000, coordination_info: dict = None):
        """
        🔄 Main method: Create packed cache BACKWARDS using coordinated ranges
        """
        print(f"🔄 Starting REVERSE sequence packing cache creation...")
        print(f"   Input: {input_dir}")
        print(f"   Output: {output_dir}")
        print(f"   Max length: {self.max_length}")
        print(f"   Compression: {self.use_compression}")
        print(f"   🚨 REVERSE MODE: Starting from LAST file!")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 🔄 FIXED: Check for existing REVERSE chunks and resume properly
        final_output_dir = os.path.join(output_dir, "512", "FineWeb")
        existing_reverse_chunks = self._find_existing_reverse_chunks(final_output_dir)
        existing_chunk_count = len(existing_reverse_chunks)
        start_chunk = existing_chunk_count

        if start_chunk > 0:
            print(f"📦 Found {start_chunk} existing REVERSE chunks - resuming from chunk {start_chunk}")
            # Find the highest chunk number to continue from
            highest_chunk = max([int(os.path.basename(f).split('_')[2].split('.')[0]) for f in existing_reverse_chunks])
            print(f"   Highest existing chunk: {highest_chunk}")
            print(f"   Will continue from: {highest_chunk + 1}")
            # existing_chunk_count already set above
            # Set the starting chunk counter to continue from the highest existing chunk
            start_chunk = highest_chunk + 1
        else:
            print(f"📦 No existing REVERSE chunks found - starting fresh")
            # existing_chunk_count already set to 0 above
            start_chunk = 0
        
        # Reset global carry for clean start
        reset_global_carry()
        
        # Create packed sequences BACKWARDS
        start_time = time.time()
        
        try:
            total_saved = self._streaming_pack_and_save_reverse(
                input_dir=input_dir,
                output_dir=output_dir,
                chunk_size=chunk_size,
                start_chunk=start_chunk,
                coordination_info=coordination_info
            )
            
            # Create metadata in the correct directory
            final_output_dir = os.path.join(output_dir, "512", "FineWeb")
            self._save_metadata(final_output_dir, [], start_time, total_saved)
            
            total_time = time.time() - start_time
            print(f"✅ REVERSE cache creation complete!")
            print(f"   Total chunks: {total_saved}")
            print(f"   Total time: {total_time/3600:.1f}h")
            print(f"   Cache directory: {output_dir}")
            
        except Exception as e:
            print(f"❌ REVERSE cache creation failed: {e}")
            print(f"💡 You can restart this helper script anytime")
            raise

def parse_args():
    parser = argparse.ArgumentParser(description='Create packed sequence cache (REVERSE mode)')
    
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
    
    return parser.parse_args()

def main():
    print("🔄 REVERSE SEQUENCE PACKING CACHE CREATOR")
    print("=" * 60)
    print("🚨 HELPER MODE: Processes files from LAST to FIRST")
    print("=" * 60)
    
    args = parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input_dir):
        print(f"❌ Input directory not found: {args.input_dir}")
        sys.exit(1)

    # Setup coordination and analyze files for reverse processing
    coordination_info = create_coordination_info(args.input_dir, args.output_dir)
    if coordination_info["total_files"] == 0:
        print("❌ No parquet files found!")
        sys.exit(1)

    # Setup coordinator for reverse processing
    coordinator = CacheCoordinator(args.output_dir)

    # Show detailed reverse analysis
    file_analysis = analyze_parquet_files_reverse(args.input_dir)
    
    # Confirm before starting
    print(f"\n🎯 Ready to create REVERSE packed cache:")
    print(f"   Input: {args.input_dir}")
    print(f"   Output: {args.output_dir}")
    print(f"   Tokenizer: {args.tokenizer}")
    print(f"   Max length: {args.max_length}")
    print(f"   🔄 REVERSE MODE: Last file first!")
    
    response = input("\nProceed with REVERSE processing? [y/N]: ").strip().lower()
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
    
    # Create REVERSE cache creator
    creator = ReverseCacheCreator(
        tokenizer=tokenizer,
        max_length=args.max_length,
        use_compression=not args.no_compression
    )
    
    # Start REVERSE processing
    start_time = time.time()
    
    try:
        creator.create_cache_reverse(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            chunk_size=args.chunk_size,
            coordination_info=coordination_info
        )
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 REVERSE SUCCESS!")
        print(f"   Total time: {total_time/3600:.1f} hours")
        print(f"   Cache location: {args.output_dir}")
        print(f"   🔄 Helper processing complete!")
        
    except KeyboardInterrupt:
        print(f"\n⏸️  REVERSE processing interrupted")
        print(f"   You can restart this helper script anytime")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ REVERSE processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
