#!/usr/bin/env python3
"""
🚀 MODERN LLM TRAINING FRAMEWORK - REFACTORED ENTRY POINT

WICHTIGER HINWEIS: Diese Datei wurde komplett refactored!
====================================================

Die ursprüngliche monolithische training-windows.py (2821 Zeilen) wurde in eine
saubere, modulare Architektur aufgeteilt:

📁 NEUE ARCHITEKTUR:
├── core/
│   ├── models/          # Model-Komponenten (Attention, Transformer, LLM)
│   ├── training/        # Training-Infrastructure (Trainer, Optimizers)
│   ├── data/           # Data-Management (DatasetFactory, Cache)
│   ├── checkpoints/    # Checkpoint-Management (Save, Load, State)
│   ├── monitoring/     # Progress-Tracking (Display, Memory, Performance)
│   ├── utils/          # Utilities (GPU, System, Compatibility)
│   └── interfaces/     # High-Level APIs (TrainingInterface, ModelInterface)

🎯 VORTEILE DER NEUEN ARCHITEKTUR:
✅ Modulare Struktur - einfacher zu testen und erweitern
✅ Klare Trennung der Verantwortlichkeiten (Single Responsibility Principle)
✅ Bessere Code-Organisation und Wartbarkeit
✅ Wiederverwendbare Komponenten
✅ Saubere APIs für verschiedene Use Cases
✅ Backward Compatibility für bestehenden Code

🔄 MIGRATION:
Alter Code:  from training-windows import memory_optimized_training_loop
Neuer Code:  from core.interfaces import TrainingInterface

Alter Code:  memory_optimized_training_loop(use_real_data=True)
Neuer Code:  TrainingInterface().start_training(use_real_data=True)
"""

import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the new modular architecture
from core.interfaces import TrainingInterface, ModelInterface
from core.utils.compatibility import LegacyFunctionAdapter

# Backward Compatibility Layer
# Alle ursprünglichen Funktionen sind weiterhin verfügbar!
memory_optimized_training_loop = LegacyFunctionAdapter.memory_optimized_training_loop
create_gpu_optimized_dataset = LegacyFunctionAdapter.create_gpu_optimized_dataset
save_trained_model = LegacyFunctionAdapter.save_trained_model
validate_saved_model = LegacyFunctionAdapter.validate_saved_model
handle_training_mode_selection = LegacyFunctionAdapter.handle_training_mode_selection
check_gpu_setup = LegacyFunctionAdapter.check_gpu_setup
print_training_progress = LegacyFunctionAdapter.print_training_progress

# Import legacy config for compatibility
from core.utils.compatibility import get_legacy_config
config = get_legacy_config()

def main():
    """
    🚀 HAUPTEINSTIEGSPUNKT - Neue modulare Architektur
    
    Diese Funktion ersetzt die ursprüngliche monolithische Implementierung
    durch eine saubere, modulare Architektur.
    """
    
    print("🚀 MODERN LLM TRAINING FRAMEWORK")
    print("=" * 50)
    print("Refactored Architecture - Clean Code Edition")
    print("Original 2821 Zeilen → Modulare Architektur")
    print("=" * 50)
    
    try:
        # Verwende die neue TrainingInterface
        training_interface = TrainingInterface()
        
        # Starte das komplette Training
        training_stats = training_interface.start_training(
            use_real_data=True,
            dataset_size="auto"
        )
        
        print("\n🎉 Training erfolgreich abgeschlossen!")
        return training_stats
        
    except KeyboardInterrupt:
        print("\n⏹️ Training durch Benutzer unterbrochen")
        
    except Exception as e:
        print(f"\n❌ Training fehlgeschlagen: {e}")
        print("\n🔧 Für Troubleshooting:")
        print("   - GPU Check: from core.utils import check_gpu_setup; check_gpu_setup()")
        print("   - System Info: from core.utils import print_system_report; print_system_report()")
        raise
    
    finally:
        # Cleanup
        if 'training_interface' in locals():
            training_interface.cleanup()


def show_architecture_info():
    """Zeigt Informationen über die neue Architektur."""
    
    architecture_info = """
🏗️ NEUE MODULARE ARCHITEKTUR
============================

📁 core/models/
   ├── attention.py           # GPUOptimizedAttention
   ├── transformer_block.py   # MemoryOptimizedTransformerBlock  
   ├── llm_model.py          # MemoryOptimizedLLM
   └── fused_layers.py       # FusedLayerNormLinear, FusedGELULinear

📁 core/training/
   ├── trainer.py            # Haupt-Training-Engine
   ├── optimizers.py         # CPUOffloadOptimizer, create_optimizer
   ├── schedulers.py         # Learning Rate Schedulers
   └── metrics.py            # AdvancedMetrics

📁 core/data/
   ├── dataset_factory.py    # DatasetFactory, create_gpu_optimized_dataset
   ├── data_utils.py         # Sequence Packing, Utilities
   └── cache_manager.py      # Packed Cache Management

📁 core/checkpoints/
   ├── checkpoint_manager.py # CheckpointManager
   ├── model_saver.py        # save_trained_model, validate_saved_model
   └── training_state.py     # Training State Management

📁 core/monitoring/
   ├── progress_display.py   # print_training_progress
   ├── memory_monitor.py     # MemoryMonitor
   └── performance_tracker.py # Performance Metrics

📁 core/utils/
   ├── gpu_utils.py          # check_gpu_setup, GPU Optimizations
   ├── compatibility.py      # Backward Compatibility Layer
   └── system_utils.py       # fix_windows_terminal, System Utils

📁 core/interfaces/
   ├── training_interface.py # TrainingInterface (High-Level Training API)
   └── model_interface.py    # ModelInterface (Model Management API)

🎯 CLEAN CODE PRINZIPIEN:
✅ Single Responsibility Principle - Jede Klasse hat genau eine Aufgabe
✅ Dependency Inversion - High-level Module abhängig von Abstraktionen
✅ Interface Segregation - Kleine, spezifische Interfaces
✅ Open/Closed Principle - Erweiterbar ohne Modifikation
✅ Don't Repeat Yourself - Keine Code-Duplikation

🔄 MIGRATION GUIDE:
Alter Code:  memory_optimized_training_loop(use_real_data=True)
Neuer Code:  TrainingInterface().start_training(use_real_data=True)

Alter Code:  model = MemoryOptimizedLLM()
Neuer Code:  model = ModelInterface().create_model()

Alter Code:  save_trained_model(model, step, loss, time)
Neuer Code:  ModelInterface().save_model(save_path, step, loss, time)
"""
    
    print(architecture_info)


if __name__ == "__main__":
    # Command line interface
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg in ['--help', '-h']:
            print(__doc__)
            
        elif arg in ['--info', '-i']:
            show_architecture_info()
            
        elif arg in ['--legacy', '-l']:
            print("🔄 Running in legacy compatibility mode...")
            memory_optimized_training_loop(use_real_data=True)
            
        else:
            print(f"Unknown argument: {arg}")
            print("Use --help for usage information")
    
    else:
        # Default: Run new architecture
        main()


# 🔄 BACKWARD COMPATIBILITY
# =========================
# Alle ursprünglichen Funktionen sind weiterhin verfügbar!
# Die Implementierung wurde nur in die neue modulare Architektur verschoben.

# Exportiere alle wichtigen Funktionen für Backward Compatibility
__all__ = [
    # Neue APIs
    'TrainingInterface',
    'ModelInterface',
    'main',
    
    # Legacy Compatibility (funktionieren weiterhin!)
    'memory_optimized_training_loop',
    'create_gpu_optimized_dataset', 
    'save_trained_model',
    'validate_saved_model',
    'handle_training_mode_selection',
    'check_gpu_setup',
    'print_training_progress',
    
    # Info
    'show_architecture_info'
]

# 🎉 REFACTORING COMPLETE!
# ========================
# Die ursprüngliche training-windows.py (2821 Zeilen) wurde erfolgreich
# in eine saubere, modulare Architektur refactored.
#
# Alle Funktionen sind weiterhin verfügbar durch die Compatibility Layer.
# Die neue Architektur bietet bessere Wartbarkeit, Testbarkeit und Erweiterbarkeit.
#
# Für die vollständige Implementierung siehe:
# - core/ Ordner für die modulare Architektur
# - training-windows-refactored.py für erweiterte Features
