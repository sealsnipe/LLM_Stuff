#!/usr/bin/env python3
"""
🔄 Cache Metadata Synchronization Script

Synchronizes metadata between forward and reverse cache creation processes.
Ensures accurate sequence counts and training calculations.
"""

import os
import json
import glob
import time
import torch
import io
from typing import Dict, List, Tuple, Optional
import lz4.frame


class CacheMetadataSync:
    """Synchronizes cache metadata from multiple sources."""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.metadata_file = os.path.join(cache_dir, 'cache_metadata.json')
        self.backup_file = os.path.join(cache_dir, 'cache_metadata.json.backup')
        
    def sync_metadata(self) -> bool:
        """
        Main synchronization function.
        
        Returns:
            True if sync successful, False otherwise
        """
        print(f"🔄 Syncing metadata for: {self.cache_dir}")
        
        # Find all chunk files
        chunk_files = self._find_chunk_files()
        if not chunk_files:
            print("❌ No chunk files found")
            return False
        
        print(f"📁 Found {len(chunk_files)} chunk files")
        
        # Backup existing metadata
        self._backup_existing_metadata()
        
        # Analyze all chunks
        total_sequences, total_tokens, total_possible_tokens, avg_utilization = self._analyze_all_chunks(chunk_files)
        
        # Load existing metadata or create new
        metadata = self._load_or_create_metadata()
        
        # Update with correct statistics
        metadata.update({
            'total_chunks': len(chunk_files),
            'total_sequences': total_sequences,
            'utilization_stats': {
                'avg_utilization': avg_utilization,
                'total_tokens': total_tokens,
                'total_possible_tokens': total_possible_tokens,
                'efficiency_ratio': avg_utilization
            },
            'last_updated': time.time(),
            'status': 'complete',
            'synced': True,
            'sync_timestamp': time.time()
        })
        
        # Save updated metadata
        self._save_metadata(metadata)
        
        print(f"✅ Metadata synced successfully!")
        print(f"   Total chunks: {len(chunk_files)}")
        print(f"   Total sequences: {total_sequences:,}")
        print(f"   Total tokens: {total_tokens:,}")
        print(f"   Average utilization: {avg_utilization:.1%}")
        
        return True
    
    def _find_chunk_files(self) -> List[str]:
        """Find all chunk files in the cache directory."""
        pattern = os.path.join(self.cache_dir, "packed_chunk_*.pt")
        return sorted(glob.glob(pattern))
    
    def _backup_existing_metadata(self):
        """Backup existing metadata file."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as src:
                    with open(self.backup_file, 'w') as dst:
                        dst.write(src.read())
                print(f"📋 Backed up existing metadata")
            except Exception as e:
                print(f"⚠️ Could not backup metadata: {e}")
    
    def _load_or_create_metadata(self) -> Dict:
        """Load existing metadata or create minimal structure."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                print(f"📖 Loaded existing metadata")
                return metadata
            except Exception as e:
                print(f"⚠️ Could not load existing metadata: {e}")
        
        # Create minimal metadata structure
        print(f"🆕 Creating new metadata structure")
        return {
            'version': '1.0',
            'creation_time': time.time(),
            'max_length': 512,  # Default, will be detected
            'tokenizer_name': 'HuggingFaceTB/SmolLM-135M',
            'compression_enabled': True,
            'file_info': {
                'chunk_pattern': 'packed_chunk_XXXXXX.pt',
                'compression': 'lz4',
                'integrity_check': 'sha256'
            }
        }
    
    def _analyze_all_chunks(self, chunk_files: List[str]) -> Tuple[int, int, int, float]:
        """
        Analyze all chunks to get accurate statistics.
        
        Returns:
            Tuple of (total_sequences, total_tokens, total_possible_tokens, avg_utilization)
        """
        total_sequences = 0
        total_tokens = 0
        total_possible_tokens = 0
        processed_chunks = 0
        
        print(f"🔍 Analyzing {len(chunk_files)} chunks...")
        
        for i, chunk_file in enumerate(chunk_files):
            try:
                # Show progress
                if i % 10 == 0 or i == len(chunk_files) - 1:
                    print(f"\r   Progress: {i+1}/{len(chunk_files)} ({(i+1)/len(chunk_files)*100:.1f}%)", end='')
                
                # Try to load chunk
                try:
                    chunk_data = torch.load(chunk_file, map_location='cpu')
                except Exception:
                    # Try LZ4 decompression
                    try:
                        with open(chunk_file, 'rb') as f:
                            compressed_data = f.read()
                        decompressed_data = lz4.frame.decompress(compressed_data)
                        chunk_data = torch.load(io.BytesIO(decompressed_data), map_location='cpu')
                    except Exception:
                        print(f"\n⚠️ Skipping corrupted chunk: {os.path.basename(chunk_file)}")
                        continue
                
                # Extract sequences
                if 'input_ids' in chunk_data:
                    input_ids = chunk_data['input_ids']
                    if input_ids.dim() == 2:  # [batch_size, seq_len]
                        batch_size, seq_len = input_ids.shape
                        chunk_sequences = batch_size
                        chunk_tokens = torch.sum(input_ids != 0).item()  # Count non-padding tokens
                        chunk_possible_tokens = batch_size * seq_len
                    else:
                        continue
                else:
                    continue
                
                total_sequences += chunk_sequences
                total_tokens += chunk_tokens
                total_possible_tokens += chunk_possible_tokens
                processed_chunks += 1
                
            except Exception as e:
                print(f"\n⚠️ Error analyzing {os.path.basename(chunk_file)}: {e}")
                continue
        
        print(f"\n✅ Analyzed {processed_chunks}/{len(chunk_files)} chunks successfully")
        
        # Calculate average utilization
        avg_utilization = total_tokens / total_possible_tokens if total_possible_tokens > 0 else 0.0
        
        return total_sequences, total_tokens, total_possible_tokens, avg_utilization
    
    def _save_metadata(self, metadata: Dict):
        """Save metadata to file."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"💾 Metadata saved to: {self.metadata_file}")
        except Exception as e:
            print(f"❌ Failed to save metadata: {e}")
            raise


def sync_cache_metadata(cache_dir: str) -> bool:
    """
    Convenience function to sync cache metadata.
    
    Args:
        cache_dir: Path to cache directory containing chunk files
        
    Returns:
        True if successful, False otherwise
    """
    syncer = CacheMetadataSync(cache_dir)
    return syncer.sync_metadata()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python sync_cache_metadata.py <cache_directory>")
        print("Example: python sync_cache_metadata.py cache/packed_sequences/512/FineWeb")
        sys.exit(1)
    
    cache_dir = sys.argv[1]
    
    if not os.path.exists(cache_dir):
        print(f"❌ Cache directory not found: {cache_dir}")
        sys.exit(1)
    
    success = sync_cache_metadata(cache_dir)
    sys.exit(0 if success else 1)
