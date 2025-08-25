#!/usr/bin/env python3
"""
🌐 FineWeb-Edu Dataset Downloader
Manueller Download und Caching von FineWeb-Edu Varianten
"""

import os
import time
from datasets import load_dataset
from tqdm import tqdm
import argparse
from config import dataset_config

def download_fineweb_sample(sample_name: str, cache_dir: str = None, force_download: bool = False):
    """
    Lädt eine spezifische FineWeb-Edu Sample-Version herunter.
    
    Args:
        sample_name: "sample-10BT", "sample-100BT", "sample-350BT", oder "default"
        cache_dir: Cache-Verzeichnis (default: aus config)
        force_download: Erzwinge Neudownload auch wenn bereits gecacht
    """
    
    cache_dir = cache_dir or dataset_config.fineweb_cache_dir
    
    # Dataset-Informationen
    dataset_info = {
        "sample-10BT": {
            "size": "~27GB",
            "tokens": "~10B",
            "description": "Kleinste Sample-Version für Tests"
        },
        "sample-100BT": {
            "size": "~277GB", 
            "tokens": "~100B",
            "description": "Mittlere Sample-Version für Training"
        },
        "sample-350BT": {
            "size": "~388GB",
            "tokens": "~350B", 
            "description": "Große Sample-Version für Production"
        },
        "default": {
            "size": "~10.4TB",
            "tokens": "~1.3T",
            "description": "Komplettes FineWeb-Edu Dataset"
        }
    }
    
    if sample_name not in dataset_info:
        print(f"❌ Unbekannte Sample-Version: {sample_name}")
        print(f"✅ Verfügbare Versionen: {list(dataset_info.keys())}")
        return False
    
    info = dataset_info[sample_name]
    print(f"🌐 FineWeb-Edu Download: {sample_name}")
    print(f"   Größe: {info['size']}")
    print(f"   Tokens: {info['tokens']}")
    print(f"   Beschreibung: {info['description']}")
    print(f"   Cache: {cache_dir}")
    print()
    
    # Warnung bei großen Downloads
    if sample_name in ["sample-350BT", "default"]:
        print("⚠️  WARNUNG: Sehr großer Download!")
        print(f"   Größe: {info['size']}")
        print("   Dies kann mehrere Stunden dauern und viel Speicherplatz benötigen.")
        
        response = input("   Möchten Sie fortfahren? (y/N): ")
        if response.lower() != 'y':
            print("❌ Download abgebrochen.")
            return False
    
    # Check ob bereits vorhanden
    if not force_download:
        try:
            print("🔍 Prüfe ob Dataset bereits gecacht ist...")
            dataset = load_dataset(
                "HuggingFaceFW/fineweb-edu",
                name=sample_name if sample_name != "default" else None,
                split="train",
                streaming=True,
                cache_dir=cache_dir
            )
            # Test ob wir Daten lesen können
            next(iter(dataset))
            print("✅ Dataset bereits verfügbar im Cache!")
            return True
        except:
            print("📥 Dataset nicht im Cache gefunden, starte Download...")
    
    try:
        print(f"🚀 Starte Download von {sample_name}...")
        start_time = time.time()
        
        # Download mit Progress-Anzeige
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name=sample_name if sample_name != "default" else None,
            split="train",
            streaming=False,  # Vollständiger Download
            cache_dir=cache_dir
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ Download erfolgreich abgeschlossen!")
        print(f"   Dauer: {duration/60:.1f} Minuten")
        print(f"   Samples: {len(dataset):,}")
        print(f"   Cache: {cache_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Download fehlgeschlagen: {e}")
        print("💡 Mögliche Lösungen:")
        print("   - Internetverbindung prüfen")
        print("   - Speicherplatz prüfen")
        print("   - Cache-Verzeichnis leeren und erneut versuchen")
        return False

def list_available_samples():
    """Zeigt verfügbare FineWeb-Edu Samples an."""
    print("📋 Verfügbare FineWeb-Edu Samples:")
    print("=" * 60)
    
    samples = [
        ("sample-10BT", "~27GB", "~10B tokens", "Für Tests & Entwicklung"),
        ("sample-100BT", "~277GB", "~100B tokens", "Für Training (EMPFOHLEN)"),
        ("sample-350BT", "~388GB", "~350B tokens", "Für Production Training"),
        ("default", "~10.4TB", "~1.3T tokens", "Komplettes Dataset")
    ]
    
    for name, size, tokens, desc in samples:
        print(f"🎯 {name}")
        print(f"   Größe: {size}")
        print(f"   Tokens: {tokens}")
        print(f"   Verwendung: {desc}")
        print()

def check_disk_space(path: str = "."):
    """Prüft verfügbaren Speicherplatz."""
    import shutil
    
    total, used, free = shutil.disk_usage(path)
    
    print(f"💾 Speicherplatz-Check ({path}):")
    print(f"   Gesamt: {total // (1024**3):.1f} GB")
    print(f"   Verwendet: {used // (1024**3):.1f} GB")
    print(f"   Frei: {free // (1024**3):.1f} GB")
    print()
    
    return free // (1024**3)  # GB

def main():
    parser = argparse.ArgumentParser(description="FineWeb-Edu Dataset Downloader")
    parser.add_argument("--sample", type=str, 
                       choices=["sample-10BT", "sample-100BT", "sample-350BT", "default"],
                       help="Sample-Version zum Download")
    parser.add_argument("--list", action="store_true", 
                       help="Zeige verfügbare Samples")
    parser.add_argument("--cache-dir", type=str, 
                       help="Cache-Verzeichnis (default: aus config)")
    parser.add_argument("--force", action="store_true",
                       help="Erzwinge Neudownload")
    parser.add_argument("--check-space", action="store_true",
                       help="Prüfe verfügbaren Speicherplatz")
    
    args = parser.parse_args()
    
    print("🌐 FineWeb-Edu Dataset Downloader")
    print("=" * 50)
    
    if args.check_space:
        check_disk_space()
    
    if args.list:
        list_available_samples()
        return
    
    if not args.sample:
        print("❌ Keine Sample-Version angegeben.")
        print("💡 Verwenden Sie --list um verfügbare Versionen zu sehen.")
        print("💡 Oder --sample <version> zum Download.")
        return
    
    # Speicherplatz-Check
    cache_dir = args.cache_dir or dataset_config.fineweb_cache_dir
    free_space = check_disk_space(cache_dir)
    
    # Empfohlener Speicherplatz
    space_requirements = {
        "sample-10BT": 30,
        "sample-100BT": 300,
        "sample-350BT": 400,
        "default": 11000
    }
    
    required = space_requirements.get(args.sample, 0)
    if free_space < required:
        print(f"⚠️  WARNUNG: Möglicherweise nicht genug Speicherplatz!")
        print(f"   Benötigt: ~{required} GB")
        print(f"   Verfügbar: {free_space} GB")
        
        response = input("   Trotzdem fortfahren? (y/N): ")
        if response.lower() != 'y':
            print("❌ Download abgebrochen.")
            return
    
    # Download starten
    success = download_fineweb_sample(
        args.sample, 
        args.cache_dir, 
        args.force
    )
    
    if success:
        print("\n🎉 Download erfolgreich!")
        print("🚀 Sie können jetzt mit dem Training beginnen:")
        print(f"   python gpu_training_optimized.py")
    else:
        print("\n❌ Download fehlgeschlagen.")

if __name__ == "__main__":
    main()
