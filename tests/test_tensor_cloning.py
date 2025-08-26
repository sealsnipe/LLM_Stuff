#!/usr/bin/env python3
"""
🧪 Test Plattformspezifisches Tensor-Cloning
Testet ob die CUDAGraphs-Fixes korrekt aktiviert werden
"""

import platform
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import training_config

def test_tensor_cloning_detection():
    print("🧪 Testing Tensor-Cloning Detection")
    print("=" * 50)
    
    print(f"Platform: {platform.system()}")
    print(f"torch.compile enabled: {training_config.use_torch_compile}")
    
    # Simuliere die Logik aus modern_llm.py
    needs_cloning = platform.system() == "Linux" and training_config.use_torch_compile
    
    print(f"Tensor cloning needed: {needs_cloning}")
    
    if needs_cloning:
        print("✅ Linux + torch.compile detected")
        print("   → Tensor cloning AKTIVIERT")
        print("   → CUDAGraphs-Fixes werden angewendet")
    else:
        print("✅ Windows oder torch.compile deaktiviert")
        print("   → Tensor cloning DEAKTIVIERT")
        print("   → Normale Performance-optimierte Ausführung")
    
    print("\n🎯 Expected behavior:")
    if platform.system() == "Linux":
        print("   Linux: Cloning nur wenn torch.compile=True")
    else:
        print("   Windows: Cloning immer deaktiviert")
    
    return needs_cloning

if __name__ == "__main__":
    test_tensor_cloning_detection()
