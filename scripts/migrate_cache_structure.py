"""
Migrate Packed Cache to New Structure

Migrates existing packed cache files to the new organized structure:
cache/packed_sequences/512/FineWeb/

This script:
1. Creates the new directory structure
2. Moves existing files to the correct location
3. Updates metadata with dataset name
4. Creates a registry of available caches
"""

import os
import json
import shutil
import glob
from typing import Dict, List


def migrate_cache_structure():
    """Migrate existing cache to new organized structure."""
    
    print("🔄 MIGRATING PACKED CACHE TO NEW STRUCTURE")
    print("=" * 50)
    
    # Paths
    old_cache_dir = "cache/packed_sequences"
    new_base_dir = "cache/packed_sequences"
    
    # Check if old structure exists
    if not os.path.exists(old_cache_dir):
        print("❌ No existing cache found")
        return False
    
    # Load existing metadata to determine sequence length
    metadata_file = os.path.join(old_cache_dir, "cache_metadata.json")
    if not os.path.exists(metadata_file):
        print("❌ No metadata found - cannot determine sequence length")
        return False
    
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        sequence_length = metadata.get('max_length', 512)
        print(f"📏 Detected sequence length: {sequence_length}")
        
    except Exception as e:
        print(f"❌ Error reading metadata: {e}")
        return False
    
    # Create new directory structure
    new_cache_dir = os.path.join(new_base_dir, str(sequence_length), "FineWeb")
    
    print(f"📁 Creating new structure: {new_cache_dir}")
    os.makedirs(new_cache_dir, exist_ok=True)
    
    # Find all files to migrate
    chunk_files = glob.glob(os.path.join(old_cache_dir, "packed_chunk_*.pt"))
    other_files = [
        os.path.join(old_cache_dir, "cache_metadata.json"),
        os.path.join(old_cache_dir, "cache_metadata.json.backup")
    ]
    
    print(f"📦 Found {len(chunk_files)} chunk files to migrate")
    
    # Migrate chunk files
    migrated_chunks = 0
    for chunk_file in chunk_files:
        filename = os.path.basename(chunk_file)
        new_path = os.path.join(new_cache_dir, filename)
        
        try:
            shutil.move(chunk_file, new_path)
            migrated_chunks += 1
            if migrated_chunks % 10 == 0:
                print(f"   Migrated {migrated_chunks}/{len(chunk_files)} chunks...")
        except Exception as e:
            print(f"⚠️  Error migrating {filename}: {e}")
    
    # Migrate metadata files
    for other_file in other_files:
        if os.path.exists(other_file):
            filename = os.path.basename(other_file)
            new_path = os.path.join(new_cache_dir, filename)
            try:
                shutil.move(other_file, new_path)
                print(f"✅ Migrated {filename}")
            except Exception as e:
                print(f"⚠️  Error migrating {filename}: {e}")
    
    # Update metadata with dataset info
    new_metadata_file = os.path.join(new_cache_dir, "cache_metadata.json")
    if os.path.exists(new_metadata_file):
        try:
            with open(new_metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Add dataset information
            metadata.update({
                'dataset_name': 'FineWeb',
                'sequence_length': sequence_length,
                'cache_structure_version': '2.0',
                'migration_timestamp': __import__('time').time()
            })
            
            with open(new_metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print("✅ Updated metadata with dataset info")
            
        except Exception as e:
            print(f"⚠️  Error updating metadata: {e}")
    
    # Create cache registry
    create_cache_registry()
    
    # Clean up old directory if empty
    try:
        remaining_files = os.listdir(old_cache_dir)
        if not remaining_files:
            os.rmdir(old_cache_dir)
            print("🧹 Cleaned up old directory")
        else:
            print(f"⚠️  Old directory not empty: {remaining_files}")
    except:
        pass
    
    print(f"\n🎉 Migration completed!")
    print(f"   Migrated: {migrated_chunks} chunk files")
    print(f"   New location: {new_cache_dir}")
    
    return True


def create_cache_registry():
    """Create a registry of available caches."""
    
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
                            'path': dataset_path,
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
    
    print(f"📋 Created cache registry: {len(registry['available_caches'])} caches found")
    
    return registry


def list_available_caches():
    """List all available caches."""
    
    registry_file = "cache/packed_sequences/cache_registry.json"
    
    if not os.path.exists(registry_file):
        print("❌ No cache registry found. Run migration first.")
        return []
    
    try:
        with open(registry_file, 'r') as f:
            registry = json.load(f)
        
        caches = registry.get('available_caches', [])
        
        if not caches:
            print("📭 No caches available")
            return []
        
        print("📋 AVAILABLE PACKED CACHES:")
        print("=" * 50)
        
        for i, cache in enumerate(caches, 1):
            print(f"{i}. {cache['dataset_name']} (seq_len: {cache['sequence_length']})")
            print(f"   Sequences: {cache['total_sequences']:,}")
            print(f"   Chunks: {cache['total_chunks']}")
            print(f"   Utilization: {cache['utilization']:.1%}")
            print(f"   Path: {cache['path']}")
            print()
        
        return caches
        
    except Exception as e:
        print(f"❌ Error reading registry: {e}")
        return []


def main():
    """Main migration script."""
    
    print("🔄 PACKED CACHE STRUCTURE MIGRATION")
    print("=" * 50)
    
    # Check if migration needed
    old_structure = os.path.exists("cache/packed_sequences/cache_metadata.json")
    new_structure = os.path.exists("cache/packed_sequences/cache_registry.json")
    
    if new_structure and not old_structure:
        print("✅ New structure already exists")
        list_available_caches()
        return
    
    if old_structure:
        print("🔄 Old structure detected - migrating...")
        success = migrate_cache_structure()
        
        if success:
            print("\n📋 Available caches after migration:")
            list_available_caches()
        else:
            print("❌ Migration failed")
    else:
        print("❌ No cache structure found")
        print("Create a new cache first using:")
        print("python scripts/create_packed_cache.py")


if __name__ == "__main__":
    main()
