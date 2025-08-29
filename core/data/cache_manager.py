"""
Cache Manager Module

Contains the CacheManager class for handling packed sequence caches.
Manages cache creation, validation, and loading for optimized training.
"""

import os
import json
import time
import torch
from typing import Dict, List, Optional
from config import training_config


class CacheManager:
    """Manager für Packed Sequence Caches."""
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or training_config.packed_cache_dir
        self.metadata_file = os.path.join(self.cache_dir, 'cache_metadata.json')
    
    def cache_exists(self) -> bool:
        """Prüft ob Cache existiert."""
        return os.path.exists(self.cache_dir) and os.path.exists(self.metadata_file)
    
    def get_cache_files(self) -> List[str]:
        """Gibt Liste der Cache-Dateien zurück."""
        if not os.path.exists(self.cache_dir):
            return []
        
        cache_files = [
            f for f in os.listdir(self.cache_dir) 
            if f.startswith('packed_chunk_') and f.endswith('.pt')
        ]
        
        return sorted(cache_files)
    
    def load_metadata(self) -> Optional[Dict]:
        """Lädt Cache-Metadaten."""
        if not os.path.exists(self.metadata_file):
            return None
        
        try:
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Fehler beim Laden der Cache-Metadaten: {e}")
            return None
    
    def save_metadata(self, metadata: Dict):
        """Speichert Cache-Metadaten."""
        os.makedirs(self.cache_dir, exist_ok=True)
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def create_minimal_metadata(self, total_chunks: int):
        """Erstellt minimale Metadaten für existierende Cache-Chunks."""
        metadata = {
            'version': '1.0',
            'creation_time': time.time(),
            'total_chunks': total_chunks,
            'max_length': training_config.sequence_length,
            'tokenizer_name': 'HuggingFaceTB/SmolLM-135M',
            'compression_enabled': False,
            'file_info': {
                'chunk_pattern': 'packed_chunk_XXXXXX.pt',
                'compression': 'none',
                'integrity_check': 'none'
            },
            'note': 'Minimal metadata auto-generated for existing chunks'
        }
        
        self.save_metadata(metadata)
        print(f"   ✅ Minimal metadata created: {total_chunks} chunks")
    
    def validate_cache_integrity(self) -> bool:
        """Validiert Cache-Integrität."""
        if not training_config.packed_cache_validation:
            return True
        
        try:
            metadata = self.load_metadata()
            if not metadata:
                return False
            
            cache_files = self.get_cache_files()
            expected_chunks = metadata.get('total_chunks', 0)
            
            if len(cache_files) != expected_chunks:
                print(f"⚠️ Cache-Integrität: {len(cache_files)} Dateien, {expected_chunks} erwartet")
                return False
            
            # Teste ersten Chunk
            if cache_files:
                first_chunk_path = os.path.join(self.cache_dir, cache_files[0])
                try:
                    test_data = torch.load(first_chunk_path, map_location='cpu')
                    if not isinstance(test_data, dict) or 'input_ids' not in test_data:
                        print("⚠️ Cache-Format ungültig")
                        return False
                except Exception as e:
                    print(f"⚠️ Cache-Chunk nicht ladbar: {e}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"⚠️ Cache-Validierung fehlgeschlagen: {e}")
            return False
    
    def get_cache_stats(self) -> Dict:
        """Gibt Cache-Statistiken zurück."""
        if not self.cache_exists():
            return {'exists': False}
        
        metadata = self.load_metadata()
        cache_files = self.get_cache_files()
        
        # Berechne Cache-Größe
        total_size = 0
        for filename in cache_files:
            filepath = os.path.join(self.cache_dir, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
        
        return {
            'exists': True,
            'total_chunks': len(cache_files),
            'total_size_mb': total_size / (1024 * 1024),
            'metadata': metadata,
            'integrity_valid': self.validate_cache_integrity()
        }
    
    def cleanup_cache(self):
        """Bereinigt Cache-Verzeichnis."""
        if not os.path.exists(self.cache_dir):
            return
        
        cache_files = self.get_cache_files()
        
        for filename in cache_files:
            filepath = os.path.join(self.cache_dir, filename)
            try:
                os.remove(filepath)
            except OSError:
                pass
        
        # Entferne Metadaten
        if os.path.exists(self.metadata_file):
            try:
                os.remove(self.metadata_file)
            except OSError:
                pass
        
        print(f"✅ Cache bereinigt: {len(cache_files)} Dateien entfernt")
    
    def prepare_cache_for_loading(self):
        """Bereitet Cache für das Laden vor."""
        if not self.cache_exists():
            return False
        
        # Erstelle minimale Metadaten falls fehlend
        if not os.path.exists(self.metadata_file):
            cache_files = self.get_cache_files()
            if cache_files:
                self.create_minimal_metadata(len(cache_files))
        
        # Validiere Cache
        if not self.validate_cache_integrity():
            print("⚠️ Cache-Integrität fehlgeschlagen")
            return False
        
        return True
    
    def create_dataloader_from_cache(self):
        """Erstellt DataLoader aus Cache."""
        if not self.prepare_cache_for_loading():
            return None
        
        try:
            # Import hier um zirkuläre Imports zu vermeiden
            from sequence_packing_cache import create_packed_dataloader
            
            dataloader = create_packed_dataloader(
                cache_dir=self.cache_dir,
                batch_size=training_config.batch_size,
                device='cuda',
                num_workers=0  # Windows compatibility
            )
            
            return dataloader
            
        except Exception as e:
            print(f"⚠️ Cache-DataLoader Erstellung fehlgeschlagen: {e}")
            return None
