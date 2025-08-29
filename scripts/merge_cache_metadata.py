#!/usr/bin/env python3
"""
🔄 Cache Metadata Merge Script

Merges metadata from reverse script with current cache metadata.
Fixes sequence counting and training calculations.
"""

import os
import json
import time
from typing import Dict, Optional


class CacheMetadataMerger:
    """Merges cache metadata from multiple sources."""
    
    def __init__(self, current_cache_dir: str, reverse_metadata_dir: str):
        self.current_cache_dir = current_cache_dir
        self.reverse_metadata_dir = reverse_metadata_dir
        
        self.current_metadata_file = os.path.join(current_cache_dir, 'cache_metadata.json')
        self.reverse_metadata_file = os.path.join(reverse_metadata_dir, 'cache_metadata.json')
        self.backup_file = os.path.join(current_cache_dir, 'cache_metadata_pre_merge.json.backup')
        
    def merge_metadata(self) -> bool:
        """
        Main merge function.
        
        Returns:
            True if merge successful, False otherwise
        """
        print(f"🔄 Merging cache metadata...")
        print(f"   Current cache: {self.current_cache_dir}")
        print(f"   Reverse metadata: {self.reverse_metadata_dir}")
        
        # Load current metadata
        current_metadata = self._load_current_metadata()
        if not current_metadata:
            print("❌ Could not load current metadata")
            return False
        
        # Load reverse metadata
        reverse_metadata = self._load_reverse_metadata()
        if not reverse_metadata:
            print("❌ Could not load reverse metadata")
            return False
        
        # Backup current metadata
        self._backup_current_metadata(current_metadata)
        
        # Merge metadata
        merged_metadata = self._merge_metadata_objects(current_metadata, reverse_metadata)
        
        # Save merged metadata
        self._save_merged_metadata(merged_metadata)
        
        print(f"✅ Metadata merged successfully!")
        self._print_merge_summary(current_metadata, reverse_metadata, merged_metadata)
        
        return True
    
    def _load_current_metadata(self) -> Optional[Dict]:
        """Load current cache metadata."""
        if not os.path.exists(self.current_metadata_file):
            print(f"❌ Current metadata file not found: {self.current_metadata_file}")
            return None
        
        try:
            with open(self.current_metadata_file, 'r') as f:
                metadata = json.load(f)
            print(f"📖 Loaded current metadata: {metadata.get('total_chunks', 0)} chunks, {metadata.get('total_sequences', 0)} sequences")
            return metadata
        except Exception as e:
            print(f"❌ Error loading current metadata: {e}")
            return None
    
    def _load_reverse_metadata(self) -> Optional[Dict]:
        """Load reverse script metadata."""
        if not os.path.exists(self.reverse_metadata_file):
            print(f"❌ Reverse metadata file not found: {self.reverse_metadata_file}")
            return None
        
        try:
            with open(self.reverse_metadata_file, 'r') as f:
                metadata = json.load(f)
            print(f"📖 Loaded reverse metadata: {metadata.get('total_chunks', 0)} chunks, {metadata.get('total_sequences', 0)} sequences")
            return metadata
        except Exception as e:
            print(f"❌ Error loading reverse metadata: {e}")
            return None
    
    def _backup_current_metadata(self, current_metadata: Dict):
        """Backup current metadata before merge."""
        try:
            with open(self.backup_file, 'w') as f:
                json.dump(current_metadata, f, indent=2)
            print(f"📋 Backed up current metadata to: {self.backup_file}")
        except Exception as e:
            print(f"⚠️ Could not backup current metadata: {e}")
    
    def _merge_metadata_objects(self, current: Dict, reverse: Dict) -> Dict:
        """
        Merge two metadata objects intelligently.
        
        Args:
            current: Current cache metadata
            reverse: Reverse script metadata
            
        Returns:
            Merged metadata
        """
        # Start with current metadata as base
        merged = current.copy()
        
        # Update with reverse metadata where it has better information
        
        # 1. Use reverse sequence count if it's higher and seems valid
        current_sequences = current.get('total_sequences', 0)
        reverse_sequences = reverse.get('total_sequences', 0)
        
        if reverse_sequences > current_sequences and reverse_sequences > 0:
            merged['total_sequences'] = reverse_sequences
            print(f"🔄 Using reverse sequence count: {reverse_sequences:,} (was {current_sequences:,})")
        
        # 2. Merge utilization stats - prefer reverse if it has valid data
        reverse_util = reverse.get('utilization_stats', {})
        current_util = current.get('utilization_stats', {})
        
        if reverse_util.get('total_tokens', 0) > current_util.get('total_tokens', 0):
            merged['utilization_stats'] = reverse_util.copy()
            print(f"🔄 Using reverse utilization stats")
        
        # 3. Update timestamps and processing info
        merged.update({
            'last_updated': time.time(),
            'merged': True,
            'merge_timestamp': time.time(),
            'merge_sources': {
                'current_chunks': current.get('total_chunks', 0),
                'current_sequences': current.get('total_sequences', 0),
                'reverse_chunks': reverse.get('total_chunks', 0),
                'reverse_sequences': reverse.get('total_sequences', 0)
            }
        })
        
        # 4. Keep the higher chunk count (current should be higher due to more processing)
        current_chunks = current.get('total_chunks', 0)
        reverse_chunks = reverse.get('total_chunks', 0)
        merged['total_chunks'] = max(current_chunks, reverse_chunks)
        
        return merged
    
    def _save_merged_metadata(self, merged_metadata: Dict):
        """Save merged metadata to current cache directory."""
        try:
            with open(self.current_metadata_file, 'w') as f:
                json.dump(merged_metadata, f, indent=2)
            print(f"💾 Merged metadata saved to: {self.current_metadata_file}")
        except Exception as e:
            print(f"❌ Failed to save merged metadata: {e}")
            raise
    
    def _print_merge_summary(self, current: Dict, reverse: Dict, merged: Dict):
        """Print summary of merge operation."""
        print(f"\n📊 MERGE SUMMARY:")
        print(f"   Current chunks: {current.get('total_chunks', 0)}")
        print(f"   Current sequences: {current.get('total_sequences', 0):,}")
        print(f"   Reverse chunks: {reverse.get('total_chunks', 0)}")
        print(f"   Reverse sequences: {reverse.get('total_sequences', 0):,}")
        print(f"   ─────────────────────────────────")
        print(f"   Merged chunks: {merged.get('total_chunks', 0)}")
        print(f"   Merged sequences: {merged.get('total_sequences', 0):,}")
        
        # Calculate tokens
        merged_sequences = merged.get('total_sequences', 0)
        seq_length = merged.get('max_length', 512)
        utilization = merged.get('utilization_stats', {}).get('avg_utilization', 0.99)
        estimated_tokens = int(merged_sequences * seq_length * utilization)
        
        print(f"   Estimated tokens: {estimated_tokens:,}")
        print(f"   Utilization: {utilization:.1%}")


def merge_cache_metadata(current_cache_dir: str, reverse_metadata_dir: str) -> bool:
    """
    Convenience function to merge cache metadata.
    
    Args:
        current_cache_dir: Path to current cache directory
        reverse_metadata_dir: Path to directory containing reverse metadata
        
    Returns:
        True if successful, False otherwise
    """
    merger = CacheMetadataMerger(current_cache_dir, reverse_metadata_dir)
    return merger.merge_metadata()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python merge_cache_metadata.py <current_cache_dir> <reverse_metadata_dir>")
        print("Example: python merge_cache_metadata.py cache/packed_sequences/512/FineWeb test")
        sys.exit(1)
    
    current_cache_dir = sys.argv[1]
    reverse_metadata_dir = sys.argv[2]
    
    if not os.path.exists(current_cache_dir):
        print(f"❌ Current cache directory not found: {current_cache_dir}")
        sys.exit(1)
    
    if not os.path.exists(reverse_metadata_dir):
        print(f"❌ Reverse metadata directory not found: {reverse_metadata_dir}")
        sys.exit(1)
    
    success = merge_cache_metadata(current_cache_dir, reverse_metadata_dir)
    sys.exit(0 if success else 1)
