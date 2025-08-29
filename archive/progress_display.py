#!/usr/bin/env python3
"""
🎯 Clean Progress Display System
Sauberes, einzeiliges Progress-Display ohne Flackern
"""

import sys
import time
import torch
from typing import Optional

class CleanProgressDisplay:
    """Sauberes Progress-Display ohne Terminal-Flackern."""
    
    def __init__(self, total_steps: int, update_interval: float = 1.0):
        self.total_steps = total_steps
        self.update_interval = update_interval
        self.last_update = 0
        self.start_time = time.time()
        self.step_times = []
        
        # Terminal setup
        self.terminal_width = 120  # Feste Breite für Konsistenz
        
        # Zeige statische Header-Info
        self._print_header()
    
    def _print_header(self):
        """Zeigt statische Header-Informationen (minimal)."""
        print()  # Leerzeile für Progress
        print()  # Zweite Leerzeile für Info-Zeile
    
    def _get_gpu_memory(self) -> tuple:
        """Holt korrekte GPU Memory-Werte."""
        if not torch.cuda.is_available():
            return 0.0, 0.0, 0.0
        
        # Aktuelle Memory-Nutzung
        allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(0) / 1024**3    # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        return allocated, reserved, total
    
    def _format_time(self, seconds: float) -> str:
        """Formatiert Zeit in lesbarem Format."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds//60:.0f}m{seconds%60:.0f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours:.0f}h{minutes:.0f}m"
    
    def _calculate_eta(self, current_step: int) -> str:
        """Berechnet geschätzte verbleibende Zeit."""
        if current_step == 0:
            return "calculating..."
        
        elapsed = time.time() - self.start_time
        steps_per_sec = current_step / elapsed
        remaining_steps = self.total_steps - current_step
        eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
        
        return self._format_time(eta_seconds)
    
    def _create_progress_bar(self, current_step: int, width: int = 30) -> str:
        """Erstellt ASCII Progress Bar - KEINE UNICODE ZEICHEN."""
        progress = current_step / self.total_steps
        filled = int(width * progress)
        bar = "=" * filled + "-" * (width - filled)
        return f"[{bar}]"
    
    def update(self, step: int, loss: float, learning_rate: float = None,
               extra_info: dict = None):
        """
        Aktualisiert Progress-Display (2 Zeilen, in-place).

        Args:
            step: Aktueller Schritt
            loss: Aktueller Loss
            learning_rate: Aktuelle Learning Rate
            extra_info: Zusätzliche Infos (dict)
        """
        current_time = time.time()

        # Rate-Limiting: Nur alle X Sekunden aktualisieren
        if current_time - self.last_update < self.update_interval and step < self.total_steps:
            return

        self.last_update = current_time

        # Berechne Statistiken
        progress_pct = (step / self.total_steps) * 100
        elapsed = current_time - self.start_time
        eta = self._calculate_eta(step)

        # GPU Memory
        gpu_alloc, gpu_reserved, gpu_total = self._get_gpu_memory()
        gpu_usage_pct = (gpu_alloc / gpu_total) * 100 if gpu_total > 0 else 0

        # Steps per second
        steps_per_sec = step / elapsed if elapsed > 0 else 0

        # KURZE Progress Bar - KEINE EMOJIS
        progress_bar = self._create_progress_bar(step, width=30)  # Kürzer!

        # Speed in s/step statt steps/s
        sec_per_step = 1.0 / steps_per_sec if steps_per_sec > 0 else 0

        # KOMPAKTE eine Zeile - ALLES ZUSAMMEN
        combined_line = f"TRAIN {progress_bar} {progress_pct:5.1f}% | Loss: {loss:6.3f} | {sec_per_step:4.1f}s/step | ETA: {eta:>6} | GPU: {gpu_alloc:3.1f}GB"

        # Learning Rate und Extra Info in combined_line integrieren
        if learning_rate:
            combined_line += f" | LR: {learning_rate:.2e}"

        if extra_info:
            for key, value in extra_info.items():
                # Batch rausfiltern
                if key.lower() == 'batch':
                    continue
                if isinstance(value, float):
                    combined_line += f" | {key}: {value:.3f}"
                else:
                    combined_line += f" | {key}: {value}"

        # ECHTE INLINE LÖSUNG: Carriage Return ohne Newline
        if step == 1:
            # Erste Ausgabe - OHNE newline
            sys.stdout.write(f"{combined_line}")
        else:
            # Update: Carriage Return + überschreibe
            sys.stdout.write(f"\r{combined_line}")

        sys.stdout.flush()

        # Bei letztem Schritt: Zusammenfassung
        if step >= self.total_steps:
            self._print_summary(elapsed, step)
    
    def _print_summary(self, elapsed: float, final_step: int):
        """Zeigt finale Zusammenfassung."""
        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETED")
        print(f"📊 Total Steps: {final_step:,}")
        print(f"⏱️  Total Time: {self._format_time(elapsed)}")
        print(f"⚡ Average Speed: {final_step/elapsed:.1f} steps/sec")
        
        # Finale GPU Memory
        gpu_alloc, gpu_reserved, gpu_total = self._get_gpu_memory()
        if gpu_total > 0:
            print(f"💾 Final GPU Usage: {gpu_alloc:.1f}GB / {gpu_total:.1f}GB ({gpu_alloc/gpu_total*100:.1f}%)")
        
        print("=" * 80)

def test_progress_display():
    """Test des Progress-Displays."""
    import random
    
    print("🧪 Testing Clean Progress Display")
    print("Simulating training progress...\n")
    
    total_steps = 100
    progress = CleanProgressDisplay(total_steps, update_interval=0.1)
    
    for step in range(1, total_steps + 1):
        # Simuliere Training
        time.sleep(0.05)  # Simuliere Trainingszeit
        
        # Simuliere Loss-Verbesserung
        loss = 10.0 * (0.95 ** step) + random.uniform(-0.1, 0.1)
        lr = 0.001 * (0.99 ** step)
        
        # Zusätzliche Infos
        extra = {
            "Acc": random.uniform(0.1, 0.9),
            "Batch": step % 32
        }
        
        progress.update(step, loss, lr, extra)
    
    print("\n🎉 Test completed!")

if __name__ == "__main__":
    test_progress_display()
