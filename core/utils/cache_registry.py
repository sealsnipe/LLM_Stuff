"""
Cache Registry Utilities

Manages the registry of available packed caches and provides
utilities for cache selection and management.
"""

import os
import json
from typing import List, Dict, Optional


def load_cache_registry() -> List[Dict]:
    """
    Load the cache registry.
    
    Returns:
        List of available cache configurations
    """
    registry_file = "cache/packed_sequences/cache_registry.json"
    
    if not os.path.exists(registry_file):
        return []
    
    try:
        with open(registry_file, 'r') as f:
            registry = json.load(f)
        
        return registry.get('available_caches', [])
        
    except Exception as e:
        print(f"⚠️  Error loading cache registry: {e}")
        return []


def display_cache_menu(caches: List[Dict]) -> Optional[Dict]:
    """
    Display cache selection menu.
    
    Args:
        caches: List of available caches
        
    Returns:
        Selected cache info or None
    """
    if not caches:
        return None
    
    print("📋 VERFÜGBARE CACHES:")
    print("=" * 50)
    
    for i, cache in enumerate(caches, 1):
        print(f" {i}. {cache['dataset_name']} (seq_len: {cache['sequence_length']})")
        print(f"     Sequences: {cache['total_sequences']:,}")
        print(f"     Chunks: {cache['total_chunks']}")
        print(f"     Utilization: {cache['utilization']:.1%}")
        print()
    
    print("=" * 50)
    
    while True:
        try:
            choice = input(f"Auswahl [1-{len(caches)}]: ").strip()
            
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(caches):
                    return caches[idx]
            
            print(f"❌ Ungültige Auswahl. Bitte 1-{len(caches)} eingeben.")
            
        except KeyboardInterrupt:
            print("\n❌ Abgebrochen")
            return None
        except Exception as e:
            print(f"❌ Fehler: {e}")


def get_cache_path(cache_info: Dict) -> str:
    """
    Get the full path to a cache.

    Args:
        cache_info: Cache information dictionary

    Returns:
        Full path to the cache directory (normalized for OS)
    """
    if not cache_info:
        return ""

    path = cache_info.get('path', '')
    return os.path.normpath(path) if path else ""


def get_cache_sequence_length(cache_info: Dict) -> int:
    """
    Get the sequence length for a cache.
    
    Args:
        cache_info: Cache information dictionary
        
    Returns:
        Sequence length for the cache
    """
    if not cache_info:
        return 512  # Default fallback
    
    return cache_info.get('sequence_length', 512)


def validate_cache(cache_info: Dict) -> bool:
    """
    Validate that a cache exists and is usable.

    Args:
        cache_info: Cache information dictionary

    Returns:
        True if cache is valid, False otherwise
    """
    if not cache_info:
        print("   ❌ No cache_info provided")
        return False

    cache_path = get_cache_path(cache_info)

    if not cache_path:
        print("   ❌ Empty cache path")
        return False

    if not os.path.exists(cache_path):
        print(f"   ❌ Cache directory not found: {cache_path}")
        return False

    # Check for metadata file
    metadata_file = os.path.join(cache_path, "cache_metadata.json")
    if not os.path.exists(metadata_file):
        print(f"   ❌ Metadata file not found: {metadata_file}")
        return False

    # Check for chunk files
    import glob
    chunk_files = glob.glob(os.path.join(cache_path, "packed_chunk_*.pt"))
    if not chunk_files:
        print(f"   ❌ No chunk files found in: {cache_path}")
        return False

    print(f"   ✅ Cache validated: {len(chunk_files)} chunks in {cache_path}")
    return True


def update_cache_registry():
    """Update the cache registry by scanning for new caches."""
    
    registry_file = "cache/packed_sequences/cache_registry.json"
    base_dir = "cache/packed_sequences"
    
    registry = {
        'version': '1.0',
        'created': __import__('time').time(),
        'available_caches': []
    }
    
    # Scan for available caches
    if os.path.exists(base_dir):
        for seq_len_dir in os.listdir(base_dir):
            seq_len_path = os.path.join(base_dir, seq_len_dir)
            
            # Skip files (like registry.json)
            if not os.path.isdir(seq_len_path):
                continue
            
            # Skip if not a number (sequence length)
            try:
                sequence_length = int(seq_len_dir)
            except ValueError:
                continue
            
            # Scan datasets in this sequence length
            for dataset_dir in os.listdir(seq_len_path):
                dataset_path = os.path.join(seq_len_path, dataset_dir)
                
                if not os.path.isdir(dataset_path):
                    continue
                
                # Check if valid cache (has metadata)
                metadata_file = os.path.join(dataset_path, "cache_metadata.json")
                if os.path.exists(metadata_file):
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        cache_info = {
                            'dataset_name': dataset_dir,
                            'sequence_length': sequence_length,
                            'path': os.path.normpath(dataset_path),  # FIXED: Normalize path for OS
                            'total_sequences': metadata.get('total_sequences', 0),
                            'total_chunks': metadata.get('total_chunks', 0),
                            'utilization': metadata.get('utilization_stats', {}).get('avg_utilization', 0.0),
                            'created': metadata.get('creation_time', 0)
                        }
                        
                        registry['available_caches'].append(cache_info)
                        
                    except Exception as e:
                        print(f"⚠️  Error reading metadata for {dataset_path}: {e}")
    
    # Save registry
    os.makedirs(os.path.dirname(registry_file), exist_ok=True)
    with open(registry_file, 'w') as f:
        json.dump(registry, f, indent=2)
    
    return registry


def find_cache_by_sequence_length(sequence_length: int) -> Optional[Dict]:
    """
    Find a cache with the specified sequence length.
    
    Args:
        sequence_length: Desired sequence length
        
    Returns:
        Cache info or None if not found
    """
    caches = load_cache_registry()
    
    for cache in caches:
        if cache.get('sequence_length') == sequence_length:
            return cache
    
    return None


def get_default_cache() -> Optional[Dict]:
    """
    Get the default cache (first available).
    
    Returns:
        Default cache info or None
    """
    caches = load_cache_registry()
    
    if caches:
        return caches[0]
    
    return None
