#!/usr/bin/env python3
"""
Test der Progress-Anzeige
"""

import time
from progress_display import CleanProgressDisplay

def test_progress():
    """Testet die Progress-Anzeige mit simulierten Training-Steps."""
    
    total_steps = 20
    progress = CleanProgressDisplay(total_steps, update_interval=0.1)
    
    print("🧪 Testing Progress Display...")
    print("Sollte sich inline aktualisieren ohne neue Zeilen zu erstellen.")
    print()
    
    for step in range(1, total_steps + 1):
        # Simuliere Training-Step
        loss = 5.0 - (step * 0.2) + (0.1 * (step % 3))  # Fallender Loss mit etwas Variation
        lr = 0.001 * (0.95 ** (step // 5))  # Decaying learning rate
        
        # Update Progress
        progress.update(
            step=step,
            loss=loss,
            learning_rate=lr,
            extra_info={'epoch': step // 5 + 1}
        )
        
        # Simuliere Verarbeitungszeit
        time.sleep(0.5)
    
    print("\n\n✅ Test completed!")
    print("Wenn die Progress-Anzeige korrekt funktioniert, sollten Sie:")
    print("1. Nur 2 Zeilen gesehen haben, die sich aktualisiert haben")
    print("2. Keine neuen Zeilen für jeden Step")
    print("3. Einen sauberen Progress-Balken und Info-Zeile")

if __name__ == "__main__":
    test_progress()
