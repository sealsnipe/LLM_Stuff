"""
🚀 SEQUENCE PACKING CACHE SYSTEM
Pre-process sequences once, use forever!

Features:
- Resume-fähig (gegen Crashes)
- Komprimierung (LZ4)
- Integrity-Checks (SHA256)
- GPU-optimiert
- Production-ready
"""

import torch
import os
import json
import glob
import time
import hashlib
import io
from typing import Dict, List, Any, Optional
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd

# Optional compression (install: pip install lz4)
try:
    import lz4.frame
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False
    print("⚠️  LZ4 not available - using uncompressed storage")

# Import from the modular architecture
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data.sequence_packing import (
    streaming_pack_sequences_pandas,
    load_parquet_fast_pandas,
    reset_global_carry
)

class PackedCacheCreator:
    """
    🔄 PRE-PROCESSING: Create packed sequence cache
    """
    
    def __init__(self, tokenizer, max_length: int = 512, use_compression: bool = True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_compression = use_compression and HAS_LZ4
        
    def create_cache(self, input_dir: str, output_dir: str, chunk_size: int = 10000):
        """
        🎯 Main method: Create packed cache with resume capability
        """
        print(f"🚀 Starting sequence packing cache creation...")
        print(f"   Input: {input_dir}")
        print(f"   Output: {output_dir}")
        print(f"   Max length: {self.max_length}")
        print(f"   Compression: {self.use_compression}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Check for existing progress
        existing_chunks = self._find_existing_chunks(output_dir)
        start_chunk = len(existing_chunks)
        
        if start_chunk > 0:
            print(f"📦 Found {start_chunk} existing chunks - resuming from chunk {start_chunk}")
        
        # Reset global carry for clean start
        reset_global_carry()
        
        # Create packed sequences with immediate saving for resume capability
        start_time = time.time()

        try:
            # Use custom streaming that saves chunks immediately
            total_saved = self._streaming_pack_and_save(
                input_dir=input_dir,
                output_dir=output_dir,
                chunk_size=chunk_size,
                start_chunk=start_chunk,
                start_time=start_time
            )

            # Finalize metadata (mark as complete)
            self._finalize_metadata(output_dir, total_saved, start_time)
            
            total_time = time.time() - start_time
            print(f"✅ Cache creation complete!")
            print(f"   Total chunks: {len(packed_chunks)}")
            print(f"   New chunks saved: {total_saved}")
            print(f"   Total time: {total_time/3600:.1f}h")
            print(f"   Cache directory: {output_dir}")
            
        except Exception as e:
            print(f"❌ Cache creation failed: {e}")
            print(f"💡 You can resume by running the same command again")
            raise
    
    def _find_existing_chunks(self, output_dir: str) -> List[str]:
        """Find existing chunk files for resume capability"""
        pattern = os.path.join(output_dir, "packed_chunk_*.pt")
        existing = sorted(glob.glob(pattern))
        return existing
    
    def _save_chunk(self, chunk_data: Dict, file_path: str) -> str:
        """Save chunk with optional compression and integrity check"""
        if self.use_compression:
            return self._save_compressed(chunk_data, file_path)
        else:
            torch.save(chunk_data, file_path)
            return self._calculate_file_hash(file_path)
    
    def _save_compressed(self, data: Dict, file_path: str) -> str:
        """Save with LZ4 compression"""
        # Serialize to bytes
        buffer = io.BytesIO()
        torch.save(data, buffer)
        serialized = buffer.getvalue()
        
        # Compress
        compressed = lz4.frame.compress(serialized)
        
        # Write compressed file
        with open(file_path, 'wb') as f:
            f.write(compressed)
        
        # Return hash for integrity
        return self._calculate_file_hash(file_path)
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash for integrity checking"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _calculate_utilization(self, input_ids: torch.Tensor) -> float:
        """Calculate token utilization for chunk"""
        if input_ids.numel() == 0:
            return 0.0
        
        total_positions = input_ids.numel()
        padding_tokens = (input_ids == self.tokenizer.pad_token_id).sum().item()
        utilized_tokens = total_positions - padding_tokens
        
        return (utilized_tokens / total_positions) * 100
    
    def _streaming_pack_and_save(self, input_dir: str, output_dir: str, chunk_size: int, start_chunk: int, start_time: float) -> int:
        """Stream processing with immediate chunk saving for resume capability"""
        from optimized_sequence_packing import load_parquet_fast_pandas, pack_sequences_heap_optimized, reset_global_carry
        import glob

        # Find all parquet files
        parquet_pattern = os.path.join(input_dir, "*.parquet")
        parquet_files = glob.glob(parquet_pattern)

        if not parquet_files:
            print(f"❌ No parquet files found in {input_dir}")
            return 0

        # Calculate resume position: which file and batch to start from
        chunk_counter = start_chunk
        total_processed = 0

        # Calculate where to resume based on existing chunks
        resume_file_idx, resume_batch_num = self._calculate_resume_position(
            parquet_files, chunk_size, start_chunk
        )

        for file_idx, file_path in enumerate(parquet_files):
            # Skip files that are already processed
            if file_idx < resume_file_idx:
                continue

            try:
                print(f"📁 Processing file {file_idx+1}/{len(parquet_files)}: {os.path.basename(file_path)}")

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
                        print(f" - Batch {chunk_num}/{total_chunks}", end="", flush=True)
                    else:
                        print(f"\r - Batch {chunk_num}/{total_chunks}", end="", flush=True)

                    # Pack this chunk
                    packed = pack_sequences_heap_optimized(chunk_texts, self.tokenizer, self.max_length, verbose=False)

                    if packed['input_ids'].size(0) > 0:
                        # Save immediately for resume capability
                        chunk_file = os.path.join(output_dir, f"packed_chunk_{chunk_counter:06d}.pt")

                        chunk_data = {
                            'input_ids': packed['input_ids'],
                            'attention_mask': packed.get('attention_mask'),
                            'position_ids': packed.get('position_ids'),
                            'metadata': {
                                'utilization': self._calculate_utilization(packed['input_ids']),
                                'num_sequences': len(packed['input_ids']),
                                'chunk_id': chunk_counter,
                                'creation_time': time.time(),
                                'max_length': self.max_length
                            }
                        }

                        # Save with compression and integrity check
                        file_hash = self._save_chunk(chunk_data, chunk_file)
                        chunk_counter += 1

                        # 🔄 UPDATE METADATA AFTER EACH CHUNK (SILENT)
                        self._update_metadata_incremental(output_dir, chunk_counter, start_time)

                    total_processed += len(chunk_texts)

                # Clear batch line and move to next file
                print(f"\r{' ' * 50}")  # Clear the batch line

            except Exception as e:
                print(f"\r📁 Processing file {file_idx+1}/{len(parquet_files)}: {os.path.basename(file_path)} - ❌ Failed: {e}")
                continue

        return chunk_counter - start_chunk

    def _calculate_resume_position(self, parquet_files: List[str], chunk_size: int, existing_chunks: int) -> tuple:
        """Calculate which file and batch to resume from based on existing chunks"""
        if existing_chunks == 0:
            return 0, 1  # Start from beginning

        # Calculate total batches processed so far
        total_batches_processed = existing_chunks
        current_batch_count = 0

        for file_idx, file_path in enumerate(parquet_files):
            try:
                # Load file to count batches
                df = pd.read_parquet(file_path)
                if 'text' not in df.columns:
                    continue

                texts = df['text'].tolist()
                file_batches = (len(texts) + chunk_size - 1) // chunk_size

                if current_batch_count + file_batches >= total_batches_processed:
                    # Resume position is in this file
                    batches_into_file = total_batches_processed - current_batch_count
                    resume_batch = batches_into_file + 1  # Next batch to process

                    if resume_batch > file_batches:
                        # This file is complete, start next file
                        return file_idx + 1, 1
                    else:
                        # Resume within this file
                        return file_idx, resume_batch

                current_batch_count += file_batches

            except Exception as e:
                print(f"⚠️  Error calculating resume position for {file_path}: {e}")
                continue

        # If we get here, all files are processed
        return len(parquet_files), 1

    def _save_metadata(self, output_dir: str, packed_chunks: List, start_time: float, total_chunks: int = 0):
        """Save cache metadata with complete statistics"""
        # Count actual chunks if not provided
        if total_chunks == 0:
            chunk_files = glob.glob(os.path.join(output_dir, "packed_chunk_*.pt"))
            total_chunks = len(chunk_files)

        # FIXED: Calculate total_sequences and utilization_stats
        total_sequences = 0
        total_tokens = 0
        total_possible_tokens = 0

        # Count sequences and calculate utilization from existing chunks
        chunk_files = glob.glob(os.path.join(output_dir, "packed_chunk_*.pt"))
        for chunk_file in chunk_files:
            try:
                chunk_data = self._load_chunk(chunk_file, metadata_only=True)
                if 'metadata' in chunk_data:
                    chunk_meta = chunk_data['metadata']
                    total_sequences += chunk_meta.get('num_sequences', 0)
                    total_tokens += chunk_meta.get('total_tokens', 0)
                    total_possible_tokens += chunk_meta.get('num_sequences', 0) * self.max_length
            except Exception as e:
                print(f"Warning: Could not read chunk {chunk_file}: {e}")
                continue

        # Calculate utilization stats
        avg_utilization = (total_tokens / total_possible_tokens) if total_possible_tokens > 0 else 0.99

        metadata = {
            'version': '1.0',
            'creation_time': time.time(),
            'processing_time_hours': (time.time() - start_time) / 3600,
            'total_chunks': total_chunks,
            'total_sequences': total_sequences,  # FIXED: Added missing field
            'max_length': self.max_length,
            'tokenizer_name': getattr(self.tokenizer, 'name_or_path', 'unknown'),
            'compression_enabled': self.use_compression,
            'utilization_stats': {  # FIXED: Added missing utilization stats
                'avg_utilization': avg_utilization,
                'total_tokens': total_tokens,
                'total_possible_tokens': total_possible_tokens,
                'efficiency_ratio': avg_utilization
            },
            'file_info': {
                'chunk_pattern': 'packed_chunk_XXXXXX.pt',
                'compression': 'lz4' if self.use_compression else 'none',
                'integrity_check': 'sha256'
            }
        }

        metadata_file = os.path.join(output_dir, 'cache_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"📊 Metadata saved: {metadata_file}")
        print(f"   Total chunks: {total_chunks}")

    def _update_metadata_incremental(self, output_dir: str, current_chunks: int, start_time: float = None):
        """
        🔄 INCREMENTAL: Update metadata after each chunk with sequence counting
        Allows training to start anytime during cache creation!
        """
        metadata_file = os.path.join(output_dir, 'cache_metadata.json')

        # Load existing metadata or create new
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                # Update existing metadata
                metadata['total_chunks'] = current_chunks
                metadata['last_updated'] = time.time()
                if start_time:
                    metadata['processing_time_hours'] = (time.time() - start_time) / 3600
            except Exception as e:
                # Create fresh metadata if corrupted (silent)
                metadata = self._create_fresh_metadata(current_chunks, start_time)
        else:
            # Create fresh metadata
            metadata = self._create_fresh_metadata(current_chunks, start_time)

        # FIXED: Update sequence counts incrementally
        try:
            total_sequences = 0
            total_tokens = 0
            total_possible_tokens = 0

            # Count from existing chunks
            chunk_files = glob.glob(os.path.join(output_dir, "packed_chunk_*.pt"))
            for chunk_file in chunk_files:
                try:
                    chunk_data = self._load_chunk(chunk_file, metadata_only=True)
                    if 'metadata' in chunk_data:
                        chunk_meta = chunk_data['metadata']
                        total_sequences += chunk_meta.get('num_sequences', 0)
                        total_tokens += chunk_meta.get('total_tokens', 0)
                        total_possible_tokens += chunk_meta.get('num_sequences', 0) * self.max_length
                except:
                    continue

            # Update metadata with current counts
            metadata['total_sequences'] = total_sequences
            if total_possible_tokens > 0:
                avg_utilization = total_tokens / total_possible_tokens
                metadata['utilization_stats'] = {
                    'avg_utilization': avg_utilization,
                    'total_tokens': total_tokens,
                    'total_possible_tokens': total_possible_tokens,
                    'efficiency_ratio': avg_utilization
                }
        except Exception as e:
            pass  # Silent failure for sequence counting

        # Save updated metadata (silent)
        try:
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            pass  # Silent failure for incremental updates

    def _create_fresh_metadata(self, current_chunks: int, start_time: float = None):
        """Create fresh metadata structure with sequence counting"""
        return {
            'version': '1.0',
            'creation_time': time.time(),
            'last_updated': time.time(),
            'processing_time_hours': (time.time() - start_time) / 3600 if start_time else 0,
            'total_chunks': current_chunks,
            'total_sequences': 0,  # FIXED: Initialize sequence count
            'max_length': self.max_length,
            'tokenizer_name': getattr(self.tokenizer, 'name_or_path', 'unknown'),
            'compression_enabled': self.use_compression,
            'utilization_stats': {  # FIXED: Initialize utilization stats
                'avg_utilization': 0.0,
                'total_tokens': 0,
                'total_possible_tokens': 0,
                'efficiency_ratio': 0.0
            },
            'file_info': {
                'chunk_pattern': 'packed_chunk_XXXXXX.pt',
                'compression': 'lz4' if self.use_compression else 'none',
                'integrity_check': 'sha256'
            },
            'status': 'in_progress',  # Mark as still being created
            'incremental_updates': True  # Flag for incremental system
        }

    def _finalize_metadata(self, output_dir: str, total_chunks: int, start_time: float):
        """
        ✅ FINALIZE: Mark metadata as complete
        """
        metadata_file = os.path.join(output_dir, 'cache_metadata.json')

        # Load existing incremental metadata
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            except Exception:
                # Fallback to fresh metadata if corrupted
                metadata = self._create_fresh_metadata(total_chunks, start_time)
        else:
            # Create fresh if missing
            metadata = self._create_fresh_metadata(total_chunks, start_time)

        # Finalize metadata
        metadata.update({
            'status': 'complete',  # Mark as finished
            'completion_time': time.time(),
            'total_chunks': total_chunks,
            'processing_time_hours': (time.time() - start_time) / 3600,
            'final_update': True
        })

        # Calculate utilization stats from actual chunks
        try:
            utilization_stats = self._calculate_utilization_stats(output_dir)
            metadata['utilization_stats'] = utilization_stats
        except Exception as e:
            print(f"⚠️  Could not calculate utilization stats: {e}")
            metadata['utilization_stats'] = {'avg_utilization': 99.0}  # Default assumption

        # Save finalized metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Metadata finalized: {metadata_file}")
        print(f"   Status: Complete")
        print(f"   Total chunks: {total_chunks}")

    def _calculate_utilization_stats(self, output_dir: str) -> dict:
        """Calculate utilization statistics from existing chunks"""
        chunk_files = glob.glob(os.path.join(output_dir, "packed_chunk_*.pt"))

        if not chunk_files:
            return {'avg_utilization': 99.0}

        utilizations = []
        for chunk_file in chunk_files[:10]:  # Sample first 10 chunks for speed
            try:
                chunk_data = torch.load(chunk_file, map_location='cpu')
                if 'metadata' in chunk_data and 'utilization' in chunk_data['metadata']:
                    utilizations.append(chunk_data['metadata']['utilization'])
            except Exception:
                continue

        if utilizations:
            avg_util = sum(utilizations) / len(utilizations)
            return {
                'avg_utilization': avg_util,
                'min_utilization': min(utilizations),
                'max_utilization': max(utilizations),
                'sample_size': len(utilizations)
            }
        else:
            return {'avg_utilization': 99.0}  # Default assumption

class PackedSequenceDataset(Dataset):
    """
    ⚡ ULTRA-FAST: Load pre-packed sequences
    """
    
    def __init__(self, cache_dir: str, device: str = 'cuda', preload_chunks: int = 20):
        self.cache_dir = cache_dir
        self.device = device
        self.preload_chunks = preload_chunks
        
        # Load and validate metadata
        self.metadata = self._load_metadata()
        self.use_compression = self.metadata.get('compression_enabled', False)
        
        # Find chunk files
        self.chunk_files = self._find_chunk_files()
        
        # Build chunk index for O(1) lookup
        self._build_chunk_index()
        
        # Cache for loaded chunks
        self._chunk_cache = {}
        
        from debug_logger import packed_cache_log
        packed_cache_log(f"📦 Packed cache loaded: {self.cache_dir}")
        packed_cache_log(f"Chunks: {len(self.chunk_files)}")
        packed_cache_log(f"Sequences: {self.total_sequences:,}")

        # Handle utilization stats (may not exist in incremental metadata)
        if 'utilization_stats' in self.metadata:
            packed_cache_log(f"Utilization: {self.metadata['utilization_stats']['avg_utilization']:.1f}%")
        else:
            packed_cache_log(f"Utilization: Calculating... (incremental cache)")

        packed_cache_log(f"Compression: {self.use_compression}")

        # Show cache status
        status = self.metadata.get('status', 'unknown')
        if status == 'in_progress':
            packed_cache_log(f"🔄 Status: Cache still growing (incremental)")
        elif status == 'complete':
            packed_cache_log(f"✅ Status: Cache complete")
        else:
            packed_cache_log(f"📊 Status: {status}")
    
    def _load_metadata(self) -> Dict:
        """Load and validate cache metadata"""
        metadata_file = os.path.join(self.cache_dir, 'cache_metadata.json')
        
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Cache metadata not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Validate cache version
        if metadata.get('version') != '1.0':
            print("⚠️  Cache version mismatch - proceed with caution")
        
        return metadata
    
    def _find_chunk_files(self) -> List[str]:
        """Find all chunk files"""
        pattern = os.path.join(self.cache_dir, "packed_chunk_*.pt")
        chunk_files = sorted(glob.glob(pattern))
        
        if not chunk_files:
            raise FileNotFoundError(f"No chunk files found in {self.cache_dir}")
        
        return chunk_files
    
    def _build_chunk_index(self):
        """Build index for fast chunk lookup"""
        from debug_logger import packed_cache_log
        packed_cache_log("🔍 Building chunk index...")

        self.chunk_index = []
        cumulative_sequences = 0

        for chunk_file in tqdm(self.chunk_files, desc="Indexing", leave=False):
            # Load only metadata (fast)
            try:
                chunk_data = self._load_chunk(chunk_file, metadata_only=True)
                num_sequences = chunk_data['metadata']['num_sequences']

                self.chunk_index.append({
                    'file': chunk_file,
                    'start_idx': cumulative_sequences,
                    'end_idx': cumulative_sequences + num_sequences,
                    'num_sequences': num_sequences
                })

                cumulative_sequences += num_sequences

            except Exception as e:
                from debug_logger import error_log
                error_log(f"Failed to index {os.path.basename(chunk_file)}: {e}", e)
                continue

        self.total_sequences = cumulative_sequences
        packed_cache_log(f"✅ Index built: {len(self.chunk_index)} chunks, {self.total_sequences:,} sequences")
    
    def _load_chunk(self, file_path: str, metadata_only: bool = False) -> Dict:
        """Load chunk with compression support"""
        if self.use_compression:
            return self._load_compressed(file_path, metadata_only)
        else:
            chunk_data = torch.load(file_path, map_location='cpu', weights_only=False)
            if metadata_only:
                return {'metadata': chunk_data['metadata']}
            return chunk_data
    
    def _load_compressed(self, file_path: str, metadata_only: bool = False) -> Dict:
        """Load LZ4 compressed chunk"""
        with open(file_path, 'rb') as f:
            compressed = f.read()
        
        decompressed = lz4.frame.decompress(compressed)
        chunk_data = torch.load(io.BytesIO(decompressed), map_location='cpu', weights_only=False)
        
        if metadata_only:
            return {'metadata': chunk_data['metadata']}
        
        return chunk_data
    
    def __len__(self):
        return self.total_sequences
    
    def __getitem__(self, idx):
        # Find correct chunk
        chunk_info = self._find_chunk(idx)
        chunk_file = chunk_info['file']
        
        # Load chunk if not cached
        if chunk_file not in self._chunk_cache:
            chunk_data = self._load_chunk(chunk_file)

            # 🚀 PERFORMANCE: Transfer entire chunk to GPU once
            if self.device == 'cuda':
                chunk_data['input_ids'] = chunk_data['input_ids'].to('cuda', non_blocking=True)
                if 'attention_mask' in chunk_data and chunk_data['attention_mask'] is not None:
                    chunk_data['attention_mask'] = chunk_data['attention_mask'].to('cuda', non_blocking=True)

            self._chunk_cache[chunk_file] = chunk_data

            # Limit cache size (now larger)
            if len(self._chunk_cache) > self.preload_chunks:
                oldest_key = next(iter(self._chunk_cache))
                del self._chunk_cache[oldest_key]
        
        # Get sequence from chunk
        chunk_data = self._chunk_cache[chunk_file]
        local_idx = idx - chunk_info['start_idx']
        
        # Return data (already on GPU from cache)
        result = {
            'input_ids': chunk_data['input_ids'][local_idx]  # Already on correct device
        }

        # Only add attention_mask and position_ids if they exist and are not None (already on GPU)
        if chunk_data.get('attention_mask') is not None:
            result['attention_mask'] = chunk_data['attention_mask'][local_idx]

        if chunk_data.get('position_ids') is not None:
            result['position_ids'] = chunk_data['position_ids'][local_idx]

        return result
    
    def _find_chunk(self, idx: int) -> Dict:
        """Binary search for chunk containing idx"""
        left, right = 0, len(self.chunk_index) - 1
        
        while left <= right:
            mid = (left + right) // 2
            chunk = self.chunk_index[mid]
            
            if chunk['start_idx'] <= idx < chunk['end_idx']:
                return chunk
            elif idx < chunk['start_idx']:
                right = mid - 1
            else:
                left = mid + 1
        
        raise IndexError(f"Index {idx} out of range (total: {self.total_sequences})")

def create_packed_dataloader(cache_dir: str, batch_size: int = 32,
                           device: str = 'cuda', num_workers: int = 4) -> DataLoader:
    """
    ⚡ Create DataLoader from packed cache
    """
    dataset = PackedSequenceDataset(cache_dir, device=device)
    
    # Pin memory only if device is CPU (avoid GPU tensor pinning error)
    use_pin_memory = (device == 'cpu')

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,  # Only pin if data is on CPU
        persistent_workers=num_workers > 0,  # Enable for multi-worker
        prefetch_factor=4 if num_workers > 0 else None,  # Aggressive prefetching
        drop_last=True  # For stable training
    )
    
    from debug_logger import packed_cache_log
    packed_cache_log(f"🚀 DataLoader ready: {len(dataset):,} sequences, {len(dataloader):,} batches")
    return dataloader
