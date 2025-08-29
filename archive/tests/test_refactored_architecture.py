#!/usr/bin/env python3
"""
Test Suite für die refactored LLM Training Architecture

Dieses Script testet alle Komponenten der neuen modularen Architektur
und validiert, dass das Refactoring erfolgreich war.
"""

import sys
import os
import traceback
from typing import Dict, List

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Testet alle Imports der neuen Architektur."""
    print("🧪 Testing Imports...")
    
    tests = []
    
    try:
        # Core interfaces
        from core.interfaces import TrainingInterface, ModelInterface
        tests.append(("✅", "Core Interfaces", "TrainingInterface, ModelInterface"))
        
        # Models
        from core.models import MemoryOptimizedLLM, GPUOptimizedAttention
        tests.append(("✅", "Model Components", "MemoryOptimizedLLM, GPUOptimizedAttention"))
        
        # Training
        from core.training import Trainer, AdvancedMetrics
        tests.append(("✅", "Training Components", "Trainer, AdvancedMetrics"))
        
        # Data
        from core.data import DatasetFactory, CacheManager
        tests.append(("✅", "Data Components", "DatasetFactory, CacheManager"))
        
        # Checkpoints
        from core.checkpoints import CheckpointManager, ModelSaver
        tests.append(("✅", "Checkpoint Components", "CheckpointManager, ModelSaver"))
        
        # Monitoring
        from core.monitoring import ProgressDisplay, MemoryMonitor
        tests.append(("✅", "Monitoring Components", "ProgressDisplay, MemoryMonitor"))
        
        # Utils
        from core.utils import GPUUtils, SystemUtils
        tests.append(("✅", "Utility Components", "GPUUtils, SystemUtils"))
        
        # Compatibility
        from core.utils.compatibility import LegacyFunctionAdapter
        tests.append(("✅", "Compatibility Layer", "LegacyFunctionAdapter"))
        
    except Exception as e:
        tests.append(("❌", "Import Error", str(e)))
    
    # Print results
    for status, component, details in tests:
        print(f"   {status} {component}: {details}")
    
    return all(test[0] == "✅" for test in tests)


def test_model_creation():
    """Testet Model-Erstellung mit der neuen API."""
    print("\n🧠 Testing Model Creation...")
    
    try:
        from core.interfaces import ModelInterface
        
        # Erstelle Model Interface
        model_interface = ModelInterface()
        
        # Erstelle Model
        model = model_interface.create_model()
        
        # Validiere Model
        is_valid = model_interface.validate_model()
        
        if is_valid:
            print("   ✅ Model Creation: Erfolgreich")
            print("   ✅ Model Validation: Bestanden")
            
            # Model Info
            info = model_interface.get_model_info()
            print(f"   📊 Model Info: {info.get('parameter_size', 'Unknown')} Parameter")
            
            # Cleanup
            model_interface.cleanup()
            return True
        else:
            print("   ❌ Model Validation: Fehlgeschlagen")
            return False
            
    except Exception as e:
        print(f"   ❌ Model Creation Error: {e}")
        return False


def test_training_interface():
    """Testet Training Interface (ohne vollständiges Training)."""
    print("\n🏃 Testing Training Interface...")
    
    try:
        from core.interfaces import TrainingInterface
        
        # Erstelle Training Interface
        training_interface = TrainingInterface()
        
        # Teste Initialisierung
        training_interface.initialize_training_environment()
        print("   ✅ Environment Initialization: Erfolgreich")
        
        # Teste Model Creation
        model = training_interface.create_model()
        if model is not None:
            print("   ✅ Model Creation via Interface: Erfolgreich")
        else:
            print("   ❌ Model Creation via Interface: Fehlgeschlagen")
            return False
        
        # Teste Dataset Creation (synthetic)
        dataloader = training_interface.create_dataset(use_real_data=False, dataset_size="small")
        if dataloader is not None:
            print("   ✅ Dataset Creation: Erfolgreich")
        else:
            print("   ❌ Dataset Creation: Fehlgeschlagen")
            return False
        
        # Teste Trainer Creation
        trainer = training_interface.create_trainer()
        if trainer is not None:
            print("   ✅ Trainer Creation: Erfolgreich")
        else:
            print("   ❌ Trainer Creation: Fehlgeschlagen")
            return False
        
        # Cleanup
        training_interface.cleanup()
        print("   ✅ Cleanup: Erfolgreich")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Training Interface Error: {e}")
        traceback.print_exc()
        return False


def test_backward_compatibility():
    """Testet Backward Compatibility mit der alten API."""
    print("\n🔄 Testing Backward Compatibility...")
    
    try:
        # Teste Legacy Imports
        import importlib.util
        spec = importlib.util.spec_from_file_location("training_windows", "training-windows.py")
        training_windows = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(training_windows)
        memory_optimized_training_loop = training_windows.memory_optimized_training_loop
        check_gpu_setup = training_windows.check_gpu_setup
        print("   ✅ Legacy Imports: Erfolgreich")
        
        # Teste Legacy Functions (ohne Ausführung)
        if callable(memory_optimized_training_loop):
            print("   ✅ memory_optimized_training_loop: Verfügbar")
        else:
            print("   ❌ memory_optimized_training_loop: Nicht verfügbar")
            return False
        
        if callable(check_gpu_setup):
            print("   ✅ check_gpu_setup: Verfügbar")
        else:
            print("   ❌ check_gpu_setup: Nicht verfügbar")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Backward Compatibility Error: {e}")
        return False


def test_system_components():
    """Testet System-Komponenten."""
    print("\n🔧 Testing System Components...")
    
    try:
        from core.utils import GPUUtils, SystemUtils
        
        # GPU Utils
        gpu_utils = GPUUtils()
        gpu_info = gpu_utils.gpu_info
        print(f"   📊 GPU Info: {gpu_info['count']} GPU(s) verfügbar")
        
        # System Utils
        system_utils = SystemUtils()
        system_info = system_utils.get_system_info()
        if 'platform' in system_info:
            platform_info = system_info['platform']
            print(f"   📊 System: {platform_info['system']} {platform_info['machine']}")
        
        print("   ✅ System Components: Erfolgreich")
        return True
        
    except Exception as e:
        print(f"   ❌ System Components Error: {e}")
        return False


def run_comprehensive_test():
    """Führt alle Tests aus und gibt Gesamtergebnis zurück."""
    print("🚀 COMPREHENSIVE ARCHITECTURE TEST")
    print("=" * 50)
    print("Testing refactored LLM Training Framework...")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("Model Creation", test_model_creation),
        ("Training Interface", test_training_interface),
        ("Backward Compatibility", test_backward_compatibility),
        ("System Components", test_system_components)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Zusammenfassung
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print("=" * 50)
    print(f"📈 OVERALL RESULT: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Refactoring was successful!")
        print("\n✅ Die neue modulare Architektur funktioniert korrekt")
        print("✅ Backward Compatibility ist gewährleistet")
        print("✅ Alle Komponenten sind funktionsfähig")
        return True
    else:
        print("⚠️ Some tests failed. Check the errors above.")
        return False


def quick_smoke_test():
    """Schneller Smoke Test für die wichtigsten Funktionen."""
    print("💨 QUICK SMOKE TEST")
    print("=" * 30)
    
    try:
        # Test 1: Import der Hauptkomponenten
        from core.interfaces import TrainingInterface, ModelInterface
        print("✅ Core imports working")
        
        # Test 2: Model Interface
        model_interface = ModelInterface()
        model = model_interface.create_model()
        model_interface.cleanup()
        print("✅ Model creation working")
        
        # Test 3: Legacy compatibility
        import importlib.util
        spec = importlib.util.spec_from_file_location("training_windows", "training-windows.py")
        training_windows = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(training_windows)
        memory_optimized_training_loop = training_windows.memory_optimized_training_loop
        print("✅ Legacy compatibility working")
        
        print("\n🎉 SMOKE TEST PASSED!")
        print("Die refactored Architektur ist grundsätzlich funktionsfähig.")
        return True
        
    except Exception as e:
        print(f"\n❌ SMOKE TEST FAILED: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_smoke_test()
    else:
        run_comprehensive_test()
