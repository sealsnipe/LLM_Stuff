#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test-Skript für Model-Speicherung und -Validierung.

Dieses Skript testet die kritischen Model-Speicherungsfunktionen
ohne ein vollständiges Training durchzuführen.
"""

import torch
import torch.nn as nn
import os
import sys
import time
from datetime import datetime

# Import der notwendigen Komponenten
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import model_config, training_config

# Import der Klassen aus training-windows.py
import importlib.util
spec = importlib.util.spec_from_file_location("training_windows", "training-windows.py")
training_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training_module)

MemoryOptimizedLLM = training_module.MemoryOptimizedLLM
save_trained_model = training_module.save_trained_model
validate_saved_model = training_module.validate_saved_model

def test_model_saving():
    """Testet Model-Speicherung mit einem kleinen Dummy-Training."""
    
    print("🧪 TEST: Model-Speicherung und -Validierung")
    print("=" * 60)
    
    # 1. Erstelle Test-Modell
    print("\n1️⃣ Erstelle Test-Modell...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    model = MemoryOptimizedLLM().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ Modell erstellt: {total_params:,} Parameter")
    
    # 2. Simuliere kurzes Training
    print("\n2️⃣ Simuliere Mini-Training (10 Steps)...")
    model.train()
    
    # Dummy-Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Dummy-Training Loop
    start_time = time.time()
    total_loss = 0.0
    
    for step in range(1, 11):  # 10 Steps
        # Dummy-Daten
        input_ids = torch.randint(0, model_config.vocab_size, (2, 64), device=device)
        labels = input_ids.clone()
        
        # Forward Pass
        outputs = model(input_ids, labels=labels)
        loss = outputs['loss']
        
        # Backward Pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        
        if step % 5 == 0:
            avg_loss = total_loss / step
            print(f"   Step {step}/10: Loss = {avg_loss:.4f}")
    
    training_time = time.time() - start_time
    final_loss = total_loss / 10
    
    print(f"   ✅ Mini-Training abgeschlossen")
    print(f"   Trainingszeit: {training_time:.2f}s")
    print(f"   Finale Loss: {final_loss:.4f}")
    
    # 3. Teste Model-Speicherung
    print("\n3️⃣ Teste Model-Speicherung...")
    
    try:
        # Erstelle Test-Verzeichnis
        test_dir = os.path.expanduser("~/AI/llm-coding/test_models")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_save_path = os.path.join(test_dir, f"test_model_{timestamp}")
        
        saved_path = save_trained_model(
            model=model,
            step=10,
            final_loss=final_loss,
            training_time=training_time,
            save_dir=model_save_path
        )
        
        print(f"   ✅ Model-Speicherung erfolgreich!")
        
    except Exception as e:
        print(f"   ❌ Model-Speicherung fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. Teste Model-Validierung
    print("\n4️⃣ Teste Model-Validierung...")
    
    try:
        validation_success = validate_saved_model(saved_path)
        
        if validation_success:
            print(f"   ✅ Model-Validierung erfolgreich!")
        else:
            print(f"   ❌ Model-Validierung fehlgeschlagen!")
            return False
            
    except Exception as e:
        print(f"   ❌ Model-Validierung Fehler: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. Teste Model-Laden
    print("\n5️⃣ Teste Model-Laden...")
    
    try:
        # Lade das gespeicherte Modell
        checkpoint_file = os.path.join(saved_path, 'model.pt')
        checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
        
        # Erstelle neues Modell und lade State
        new_model = MemoryOptimizedLLM()
        new_model.load_state_dict(checkpoint['model_state_dict'])
        new_model = new_model.to(device)
        new_model.eval()
        
        # Teste Inference
        test_input = torch.randint(0, model_config.vocab_size, (1, 10), device=device)
        
        with torch.no_grad():
            outputs = new_model(test_input)
            logits = outputs['logits']
        
        print(f"   ✅ Model erfolgreich geladen und getestet!")
        print(f"   Output Shape: {logits.shape}")
        
        # Vergleiche Parameter
        original_params = sum(p.numel() for p in model.parameters())
        loaded_params = sum(p.numel() for p in new_model.parameters())
        
        if original_params == loaded_params:
            print(f"   ✅ Parameter-Anzahl stimmt überein: {loaded_params:,}")
        else:
            print(f"   ❌ Parameter-Anzahl unterschiedlich: {original_params} vs {loaded_params}")
            return False
        
    except Exception as e:
        print(f"   ❌ Model-Laden fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 6. Teste gespeicherte Dateien
    print("\n6️⃣ Prüfe gespeicherte Dateien...")
    
    expected_files = ['model.pt', 'config.json', 'training_info.json', 'README.md']
    
    for filename in expected_files:
        filepath = os.path.join(saved_path, filename)
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            print(f"   ✅ {filename}: {file_size:,} bytes")
        else:
            print(f"   ❌ {filename}: Datei fehlt!")
            return False
    
    # 7. Zusammenfassung
    print("\n" + "=" * 60)
    print("🎉 ALLE TESTS ERFOLGREICH!")
    print(f"   Model gespeichert in: {saved_path}")
    print(f"   Bereit für produktives Training!")
    print("=" * 60)
    
    return True


def cleanup_test_models():
    """Bereinigt Test-Modelle (optional)."""
    
    test_dir = os.path.expanduser("~/AI/llm-coding/test_models")
    if os.path.exists(test_dir):
        import shutil
        try:
            shutil.rmtree(test_dir)
            print(f"🧹 Test-Modelle bereinigt: {test_dir}")
        except Exception as e:
            print(f"⚠️  Bereinigung fehlgeschlagen: {e}")


if __name__ == "__main__":
    # Windows UTF-8 Fix
    if os.name == 'nt':
        os.system('chcp 65001 > nul')
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    
    print("🔧 LLM Model-Speicherung Test")
    print(f"Datum: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA verfügbar: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # Führe Tests aus
    success = test_model_saving()
    
    if success:
        print("\n✅ Model-Speicherung ist bereit für produktives Training!")
        
        # Frage nach Bereinigung
        try:
            response = input("\n🧹 Test-Modelle löschen? (y/N): ").strip().lower()
            if response in ['y', 'yes', 'ja', 'j']:
                cleanup_test_models()
        except KeyboardInterrupt:
            print("\n👋 Test beendet.")
    else:
        print("\n❌ Model-Speicherung muss behoben werden!")
        sys.exit(1)
