#!/usr/bin/env python3
"""
🎯 Super Simple Progress Display
Eine einzige Zeile, die sich sauber aktualisiert
"""

import sys
import time
import torch

class SimpleProgress:
    """Einfachste Progress-Anzeige - eine Zeile, kein Flackern."""
    
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.start_time = time.time()
        
        # Zeige Header einmal
        print("🚀 TRAINING STARTED")
        print("=" * 80)
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🖥️  GPU: {gpu_name} ({gpu_total:.1f}GB)")
        print(f"📊 Total Steps: {total_steps:,}")
        print("=" * 80)
    
    def update(self, step: int, loss: float, lr: float = None):
        """Aktualisiert eine einzige Zeile."""
        
        # Berechne Statistiken
        progress = step / self.total_steps
        elapsed = time.time() - self.start_time
        speed = step / elapsed if elapsed > 0 else 0
        
        # ETA
        if step > 0:
            eta_seconds = (self.total_steps - step) / speed if speed > 0 else 0
            if eta_seconds < 60:
                eta = f"{eta_seconds:.0f}s"
            elif eta_seconds < 3600:
                eta = f"{eta_seconds//60:.0f}m{eta_seconds%60:.0f}s"
            else:
                eta = f"{eta_seconds//3600:.0f}h{(eta_seconds%3600)//60:.0f}m"
        else:
            eta = "calculating..."
        
        # GPU Memory
        if torch.cuda.is_available():
            gpu_used = torch.cuda.memory_allocated(0) / 1024**3
            gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_pct = (gpu_used / gpu_total) * 100
            gpu_info = f"GPU: {gpu_used:.1f}GB ({gpu_pct:.1f}%)"
        else:
            gpu_info = "GPU: N/A"
        
        # Progress Bar
        bar_width = 30
        filled = int(bar_width * progress)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        # Hauptzeile
        line = (f"\r🚀 [{bar}] {progress*100:5.1f}% │ "
                f"Step {step:4d}/{self.total_steps} │ "
                f"Loss {loss:6.4f} │ "
                f"{speed:4.1f} steps/s │ "
                f"ETA {eta:>6} │ "
                f"{gpu_info}")
        
        if lr:
            line += f" │ LR {lr:.2e}"
        
        # Schreibe ohne neue Zeile
        sys.stdout.write(line)
        sys.stdout.flush()
        
        # Bei Abschluss: neue Zeile
        if step >= self.total_steps:
            print()  # Neue Zeile
            self._print_summary(elapsed)
    
    def _print_summary(self, elapsed: float):
        """Finale Zusammenfassung."""
        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETED")
        print(f"⏱️  Total Time: {elapsed//60:.0f}m{elapsed%60:.0f}s")
        print(f"⚡ Average Speed: {self.total_steps/elapsed:.1f} steps/sec")
        
        if torch.cuda.is_available():
            gpu_used = torch.cuda.memory_allocated(0) / 1024**3
            gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"💾 Final GPU Usage: {gpu_used:.1f}GB / {gpu_total:.1f}GB")
        
        print("=" * 80)

def test_simple_progress():
    """Test der einfachen Progress-Anzeige."""
    import random
    
    print("🧪 Testing Simple Progress Display")
    print("Simulating smooth training progress...\n")
    
    total_steps = 50
    progress = SimpleProgress(total_steps)
    
    for step in range(1, total_steps + 1):
        # Simuliere Training
        time.sleep(0.1)  # Simuliere Trainingszeit
        
        # Simuliere Loss-Verbesserung
        loss = 10.0 * (0.9 ** step) + random.uniform(-0.05, 0.05)
        lr = 0.001 * (0.99 ** step)
        
        progress.update(step, loss, lr)
    
    print("\n🎉 Test completed!")

if __name__ == "__main__":
    test_simple_progress()
