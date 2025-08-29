#!/usr/bin/env python3
"""
🔄 Unified Cache Metadata Manager

Manages metadata for both forward and reverse cache creation processes.
Ensures proper metadata synchronization and merging.
"""

import os
import json
import time
import glob
import torch
import lz4.frame
import io
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class UnifiedMetadataManager:
    """Manages metadata for packed sequence caches with forward/reverse coordination."""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.main_metadata_file = self.cache_dir / 'cache_metadata.json'
        self.forward_metadata_file = self.cache_dir / 'cache_metadata_forward.json'
        self.reverse_metadata_file = self.cache_dir / 'cache_metadata_reverse.json'
        self.unified_metadata_file = self.cache_dir / 'cache_metadata_unified.json'
        
    def create_process_metadata(self, process_type: str, chunk_range: Tuple[int, int], 
                              max_length: int = 512, tokenizer_name: str = 'HuggingFaceTB/SmolLM-135M') -> Dict:
        """
        Create metadata for a specific process (forward/reverse).
        
        Args:
            process_type: 'forward' or 'reverse'
            chunk_range: (start_chunk, end_chunk)
            max_length: Sequence length
            tokenizer_name: Tokenizer identifier
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            'version': '1.0',
            'process_type': process_type,
            'creation_time': time.time(),
            'last_updated': time.time(),
            'chunk_range': {
                'start': chunk_range[0],
                'end': chunk_range[1],
                'assigned_chunks': chunk_range[1] - chunk_range[0] + 1
            },
            'max_length': max_length,
            'tokenizer_name': tokenizer_name,
            'compression_enabled': True,
            'file_info': {
                'chunk_pattern': 'packed_chunk_XXXXXX.pt',
                'compression': 'lz4',
                'integrity_check': 'sha256'
            },
            'status': 'in_progress',
            'total_chunks': 0,
            'total_sequences': 0,
            'utilization_stats': {
                'avg_utilization': 0.0,
                'total_tokens': 0,
                'total_possible_tokens': 0,
                'efficiency_ratio': 0.0
            },
            'processing_stats': {
                'start_time': time.time(),
                'processing_time_hours': 0.0,
                'chunks_per_hour': 0.0,
                'estimated_completion': None
            }
        }
        
        # Save process-specific metadata
        metadata_file = self.forward_metadata_file if process_type == 'forward' else self.reverse_metadata_file
        self._save_metadata(metadata, metadata_file)
        
        print(f"📋 Created {process_type} metadata: chunks {chunk_range[0]}-{chunk_range[1]}")
        return metadata
    
    def update_process_metadata(self, process_type: str, chunks_created: int, 
                              sequences_added: int = 0, tokens_added: int = 0):
        """Update metadata for a specific process."""
        metadata_file = self.forward_metadata_file if process_type == 'forward' else self.reverse_metadata_file
        
        if not metadata_file.exists():
            print(f"⚠️ No {process_type} metadata found to update")
            return
        
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Update counts
            metadata['total_chunks'] = chunks_created
            metadata['total_sequences'] += sequences_added
            metadata['last_updated'] = time.time()
            
            # Update utilization stats
            if tokens_added > 0:
                current_tokens = metadata['utilization_stats']['total_tokens']
                current_possible = metadata['utilization_stats']['total_possible_tokens']
                
                metadata['utilization_stats']['total_tokens'] = current_tokens + tokens_added
                metadata['utilization_stats']['total_possible_tokens'] = current_possible + (sequences_added * metadata['max_length'])
                
                if metadata['utilization_stats']['total_possible_tokens'] > 0:
                    metadata['utilization_stats']['avg_utilization'] = (
                        metadata['utilization_stats']['total_tokens'] / 
                        metadata['utilization_stats']['total_possible_tokens']
                    )
                    metadata['utilization_stats']['efficiency_ratio'] = metadata['utilization_stats']['avg_utilization']
            
            # Update processing stats
            start_time = metadata['processing_stats']['start_time']
            elapsed_hours = (time.time() - start_time) / 3600
            metadata['processing_stats']['processing_time_hours'] = elapsed_hours
            
            if elapsed_hours > 0:
                metadata['processing_stats']['chunks_per_hour'] = chunks_created / elapsed_hours
                
                # Estimate completion
                assigned_chunks = metadata['chunk_range']['assigned_chunks']
                if chunks_created < assigned_chunks:
                    remaining_chunks = assigned_chunks - chunks_created
                    hours_remaining = remaining_chunks / metadata['processing_stats']['chunks_per_hour']
                    metadata['processing_stats']['estimated_completion'] = time.time() + (hours_remaining * 3600)
            
            self._save_metadata(metadata, metadata_file)
            
        except Exception as e:
            print(f"⚠️ Error updating {process_type} metadata: {e}")
    
    def finalize_process_metadata(self, process_type: str):
        """Mark a process as complete and analyze final statistics."""
        metadata_file = self.forward_metadata_file if process_type == 'forward' else self.reverse_metadata_file
        
        if not metadata_file.exists():
            print(f"⚠️ No {process_type} metadata found to finalize")
            return
        
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Analyze actual chunks created
            chunk_files = self._find_process_chunks(process_type)
            actual_sequences, actual_tokens, actual_possible_tokens, avg_utilization = self._analyze_chunks(chunk_files)
            
            # Update with final statistics
            metadata.update({
                'status': 'complete',
                'completion_time': time.time(),
                'total_chunks': len(chunk_files),
                'total_sequences': actual_sequences,
                'utilization_stats': {
                    'avg_utilization': avg_utilization,
                    'total_tokens': actual_tokens,
                    'total_possible_tokens': actual_possible_tokens,
                    'efficiency_ratio': avg_utilization
                }
            })
            
            self._save_metadata(metadata, metadata_file)
            
            print(f"✅ {process_type.title()} process finalized:")
            print(f"   Chunks: {len(chunk_files)}")
            print(f"   Sequences: {actual_sequences:,}")
            print(f"   Tokens: {actual_tokens:,}")
            print(f"   Utilization: {avg_utilization:.1%}")
            
        except Exception as e:
            print(f"⚠️ Error finalizing {process_type} metadata: {e}")
    
    def create_unified_metadata(self) -> bool:
        """Create unified metadata from all process metadata files."""
        print("🔄 Creating unified metadata...")
        
        # Load all process metadata
        forward_metadata = self._load_metadata(self.forward_metadata_file)
        reverse_metadata = self._load_metadata(self.reverse_metadata_file)
        existing_metadata = self._load_metadata(self.main_metadata_file)

        if not forward_metadata and not reverse_metadata and not existing_metadata:
            print("❌ No metadata found")
            return False

        # If we only have existing metadata, use it as base
        if not forward_metadata and not reverse_metadata and existing_metadata:
            print("📋 Using existing metadata as base for unified metadata")
            forward_metadata = existing_metadata  # Treat existing as forward process
        
        # Analyze all chunks in directory
        all_chunk_files = sorted(glob.glob(str(self.cache_dir / "packed_chunk_*.pt")))
        total_sequences, total_tokens, total_possible_tokens, avg_utilization = self._analyze_chunks(all_chunk_files)
        
        # Create unified metadata
        unified_metadata = {
            'version': '1.0',
            'creation_time': min(
                forward_metadata.get('creation_time', time.time()) if forward_metadata else time.time(),
                reverse_metadata.get('creation_time', time.time()) if reverse_metadata else time.time()
            ),
            'last_updated': time.time(),
            'unified_creation_time': time.time(),
            'max_length': forward_metadata.get('max_length', 512) if forward_metadata else reverse_metadata.get('max_length', 512),
            'tokenizer_name': forward_metadata.get('tokenizer_name', 'HuggingFaceTB/SmolLM-135M') if forward_metadata else reverse_metadata.get('tokenizer_name', 'HuggingFaceTB/SmolLM-135M'),
            'compression_enabled': True,
            'file_info': {
                'chunk_pattern': 'packed_chunk_XXXXXX.pt',
                'compression': 'lz4',
                'integrity_check': 'sha256'
            },
            'status': 'complete',
            'total_chunks': len(all_chunk_files),
            'total_sequences': total_sequences,
            'utilization_stats': {
                'avg_utilization': avg_utilization,
                'total_tokens': total_tokens,
                'total_possible_tokens': total_possible_tokens,
                'efficiency_ratio': avg_utilization
            },
            'process_summary': {
                'forward_process': {
                    'chunks': len(self._find_process_chunks('forward')) if forward_metadata else 0,
                    'sequences': forward_metadata.get('total_sequences', 0) if forward_metadata else 0,
                    'status': forward_metadata.get('status', 'unknown') if forward_metadata else 'not_run'
                },
                'reverse_process': {
                    'chunks': len(self._find_process_chunks('reverse')) if reverse_metadata else 0,
                    'sequences': reverse_metadata.get('total_sequences', 0) if reverse_metadata else 0,
                    'status': reverse_metadata.get('status', 'unknown') if reverse_metadata else 'not_run'
                }
            },
            'dataset_name': 'FineWeb',
            'sequence_length': forward_metadata.get('max_length', 512) if forward_metadata else reverse_metadata.get('max_length', 512),
            'cache_structure_version': '3.0',
            'unified_metadata': True
        }
        
        # Save unified metadata as main metadata
        self._save_metadata(unified_metadata, self.main_metadata_file)
        self._save_metadata(unified_metadata, self.unified_metadata_file)
        
        print(f"✅ Unified metadata created:")
        print(f"   Total chunks: {len(all_chunk_files)}")
        print(f"   Total sequences: {total_sequences:,}")
        print(f"   Total tokens: {total_tokens:,}")
        print(f"   Average utilization: {avg_utilization:.1%}")
        print(f"   Forward chunks: {unified_metadata['process_summary']['forward_process']['chunks']}")
        print(f"   Reverse chunks: {unified_metadata['process_summary']['reverse_process']['chunks']}")
        
        return True
    
    def _find_process_chunks(self, process_type: str) -> List[str]:
        """Find chunks created by a specific process."""
        all_chunks = sorted(glob.glob(str(self.cache_dir / "packed_chunk_*.pt")))
        
        if process_type == 'forward':
            # Forward chunks typically start from 0
            return [f for f in all_chunks if int(f.split('_')[-1].split('.')[0]) < 1000]
        else:  # reverse
            # Reverse chunks typically start from 1000+
            return [f for f in all_chunks if int(f.split('_')[-1].split('.')[0]) >= 1000]
    
    def _analyze_chunks(self, chunk_files: List[str]) -> Tuple[int, int, int, float]:
        """Analyze chunks to get accurate statistics."""
        total_sequences = 0
        total_tokens = 0
        total_possible_tokens = 0
        
        for chunk_file in chunk_files:
            try:
                # Load with LZ4 decompression
                with open(chunk_file, 'rb') as f:
                    compressed_data = f.read()
                decompressed_data = lz4.frame.decompress(compressed_data)
                chunk_data = torch.load(io.BytesIO(decompressed_data), map_location='cpu', weights_only=False)
                
                if 'input_ids' in chunk_data:
                    sequences = chunk_data['input_ids'].shape[0]
                    tokens = torch.sum(chunk_data['input_ids'] != 0).item()
                    possible_tokens = chunk_data['input_ids'].numel()
                    
                    total_sequences += sequences
                    total_tokens += tokens
                    total_possible_tokens += possible_tokens
                    
            except Exception as e:
                print(f"⚠️ Error analyzing {os.path.basename(chunk_file)}: {e}")
                continue
        
        avg_utilization = total_tokens / total_possible_tokens if total_possible_tokens > 0 else 0.0
        return total_sequences, total_tokens, total_possible_tokens, avg_utilization
    
    def _load_metadata(self, metadata_file: Path) -> Optional[Dict]:
        """Load metadata from file."""
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Error loading {metadata_file}: {e}")
            return None
    
    def _save_metadata(self, metadata: Dict, metadata_file: Path):
        """Save metadata to file."""
        try:
            metadata_file.parent.mkdir(parents=True, exist_ok=True)
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"⚠️ Error saving {metadata_file}: {e}")


def create_unified_metadata(cache_dir: str) -> bool:
    """Convenience function to create unified metadata."""
    manager = UnifiedMetadataManager(cache_dir)
    return manager.create_unified_metadata()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python unified_metadata_manager.py <cache_directory>")
        print("Example: python unified_metadata_manager.py cache/packed_sequences/512/FineWeb")
        sys.exit(1)
    
    cache_dir = sys.argv[1]
    
    if not os.path.exists(cache_dir):
        print(f"❌ Cache directory not found: {cache_dir}")
        sys.exit(1)
    
    success = create_unified_metadata(cache_dir)
    sys.exit(0 if success else 1)
