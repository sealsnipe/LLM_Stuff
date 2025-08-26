#!/usr/bin/env python3
"""
🧪 Test Dataset Profile Configuration
Zeigt alle verfügbaren Profile und die aktuelle Auswahl
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import dataset_config

def test_dataset_profiles():
    print("🧪 Testing Dataset Profile Configuration")
    print("=" * 60)
    
    # Zeige aktuelle Auswahl
    current_profile = dataset_config.default_dataset_size
    print(f"📊 Current Profile: {current_profile.upper()}")
    print()
    
    # Zeige alle verfügbaren Profile
    print("📋 Available Dataset Profiles:")
    print("-" * 60)
    
    for profile_name, config in dataset_config.dataset_sizes.items():
        samples = config['num_samples']
        description = config['description']
        training_time = config['training_time']
        tokens = config['estimated_tokens']
        
        # Markiere aktuelles Profil
        marker = "👉 " if profile_name == current_profile else "   "
        
        print(f"{marker}{profile_name.upper():<12} | {samples or 'All':>10} samples | {training_time:<15} | {tokens}")
        print(f"{'':>15} | {description}")
        
        # Zeige Dataset-Größe wenn verfügbar
        if 'dataset_size' in config:
            print(f"{'':>15} | Dataset Size: {config['dataset_size']}")
        
        print()
    
    # Zeige wie man es ändert
    print("🔧 How to change the profile:")
    print("   Edit config.py:")
    print('   default_dataset_size: str = "large"    # für 1M samples')
    print('   default_dataset_size: str = "small"    # für 10k samples')
    print('   default_dataset_size: str = "tiny"     # für 1k samples')
    print()
    
    # Zeige aktuelle Konfiguration Details
    if current_profile in dataset_config.dataset_sizes:
        current_config = dataset_config.dataset_sizes[current_profile]
        print(f"🎯 Current Training Configuration ({current_profile.upper()}):")
        print(f"   Samples: {current_config['num_samples'] or 'All':,}")
        print(f"   Estimated Tokens: {current_config['estimated_tokens']}")
        print(f"   Training Time: {current_config['training_time']}")
        if 'dataset_size' in current_config:
            print(f"   Dataset Size: {current_config['dataset_size']}")
        print(f"   Description: {current_config['description']}")
    
    return current_profile

def show_profile_recommendations():
    """Zeigt Empfehlungen für verschiedene Use Cases."""
    print("\n💡 Profile Recommendations:")
    print("-" * 40)
    print("🚀 Quick Testing:     tiny (1k samples, 5 min)")
    print("🔧 Development:       small (10k samples, 30 min)")
    print("🎯 Serious Training:  medium (300k samples, 15 hours)")
    print("🏭 Production Ready:  large (1M samples, 2 days)")
    print("🌟 Full Production:   production (10M samples, 1-2 weeks)")
    print("🌍 Research/Complete: full (All samples, months)")

if __name__ == "__main__":
    current = test_dataset_profiles()
    show_profile_recommendations()
    
    print("\n" + "="*60)
    print(f"✅ Ready to train with {current.upper()} profile!")
    print("🔧 Change profile in config.py if needed")
