"""
Smart Training Logger - Lazy but Powerful Progress Tracking
Logs training metrics to JSON for easy analysis and visualization.
"""

import json
import time
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
import pandas as pd

class TrainingLogger:
    """
    Lazy but powerful training logger.
    - Logs every N steps to JSON
    - Auto-generates plots
    - Easy to query and analyze
    """
    
    def __init__(self, log_dir: str = "logs", model_name: str = "model", log_every: int = 10, run_id: int = None):
        self.log_dir = log_dir
        self.model_name = model_name
        self.log_every = log_every

        # Auto-determine run_id if not provided
        if run_id is None:
            self.run_id = self._get_next_run_id()
        else:
            self.run_id = run_id

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

        # Log file paths with run_id
        self.log_file = os.path.join(log_dir, f"{model_name}_run_{self.run_id}.json")
        self.summary_file = os.path.join(log_dir, f"{model_name}_run_{self.run_id}_summary.json")
        
        # Initialize log data
        self.training_data = []
        self.start_time = time.time()
        self.last_log_step = 0
        
        # Training metadata
        self.metadata = {
            "model_name": model_name,
            "start_time": datetime.now().isoformat(),
            "log_every": log_every,
            "total_steps": None,
            "target_tokens": None
        }
        
        print(f"📊 Training Logger initialized: {self.log_file}")

    def _get_next_run_id(self) -> int:
        """Ermittelt die nächste Run-ID für dieses Modell (DEPRECATED - sollte von Training übergeben werden)."""
        print("⚠️  TrainingLogger._get_next_run_id() ist deprecated. Run-ID sollte vom Training übergeben werden.")

        if not os.path.exists(self.log_dir):
            return 1

        # Finde alle existierenden Runs für dieses Modell
        existing_runs = []
        for filename in os.listdir(self.log_dir):
            if filename.startswith(f"{self.model_name}_run_") and filename.endswith(".json"):
                try:
                    # Extrahiere Run-ID aus Dateiname
                    run_part = filename.replace(f"{self.model_name}_run_", "").replace(".json", "").replace("_summary", "")
                    run_id = int(run_part)
                    existing_runs.append(run_id)
                except ValueError:
                    continue

        # Nächste verfügbare Run-ID
        next_run = max(existing_runs) + 1 if existing_runs else 1

        # WICHTIG: Erstelle sofort eine leere Datei um Race Conditions zu vermeiden
        placeholder_file = os.path.join(self.log_dir, f"{self.model_name}_run_{next_run}.json")
        if not os.path.exists(placeholder_file):
            with open(placeholder_file, 'w') as f:
                json.dump({"metadata": {"model_name": self.model_name, "run_id": next_run}, "training_data": []}, f)

        return next_run

    def set_training_config(self, total_steps: int, target_tokens: int = None):
        """Set training configuration."""
        self.metadata["total_steps"] = total_steps
        self.metadata["target_tokens"] = target_tokens
        self._save_metadata()
    
    def log_step(self, step: int, loss: float, lr: float, tokens_per_sec: float, 
                 step_time: float, gpu_memory: float = None, **kwargs):
        """
        Log a training step (only every N steps to avoid overhead).
        
        Args:
            step: Current training step
            loss: Training loss
            lr: Learning rate
            tokens_per_sec: Tokens processed per second
            step_time: Time per step in seconds
            gpu_memory: GPU memory usage in GB
            **kwargs: Additional metrics (clip_rate, flash_hit_rate, etc.)
        """
        # Only log every N steps
        if step % self.log_every != 0 and step != 1:
            return
        
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # Calculate progress
        progress_pct = (step / self.metadata.get("total_steps", step)) * 100 if self.metadata.get("total_steps") else 0
        
        # Create log entry
        log_entry = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "elapsed_time": elapsed_time,
            "loss": loss,
            "lr": lr,
            "tokens_per_sec": tokens_per_sec,
            "step_time": step_time,
            "progress_pct": progress_pct,
            "gpu_memory_gb": gpu_memory,
            **kwargs  # Additional metrics
        }
        
        # Add to training data
        self.training_data.append(log_entry)
        self.last_log_step = step
        
        # Save to file (append mode for safety)
        self._save_log_entry(log_entry)
        
        # Auto-generate plots every 100 logged steps
        if len(self.training_data) % 10 == 0:
            self.generate_plots()
    
    def _save_log_entry(self, entry: Dict[str, Any]):
        """Save single log entry to file."""
        try:
            # Load existing data
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    data = json.load(f)
            else:
                data = {"metadata": self.metadata, "training_data": []}
            
            # Add new entry
            data["training_data"].append(entry)
            
            # Save back
            with open(self.log_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"⚠️  Log save failed: {e}")
    
    def _save_metadata(self):
        """Save metadata to summary file."""
        try:
            with open(self.summary_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            print(f"⚠️  Metadata save failed: {e}")
    
    def generate_plots(self, save_plots: bool = True):
        """Generate training plots from logged data."""
        if not MATPLOTLIB_AVAILABLE:
            print("⚠️  Matplotlib not available - skipping plot generation")
            return

        if len(self.training_data) < 2:
            return

        try:
            # Convert to DataFrame for easy plotting
            df = pd.DataFrame(self.training_data)
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Training Progress: {self.model_name}', fontsize=16)
            
            # Plot 1: Loss curve
            axes[0, 0].plot(df['step'], df['loss'], 'b-', linewidth=2)
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_yscale('log')  # Log scale for loss
            
            # Plot 2: Tokens/sec
            axes[0, 1].plot(df['step'], df['tokens_per_sec'], 'g-', linewidth=2)
            axes[0, 1].set_title('Training Speed')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Tokens/sec')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Learning Rate
            axes[1, 0].plot(df['step'], df['lr'], 'r-', linewidth=2)
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Step Time
            axes[1, 1].plot(df['step'], df['step_time'], 'm-', linewidth=2)
            axes[1, 1].set_title('Step Time')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Seconds/Step')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_plots:
                plot_file = os.path.join(self.log_dir, f"{self.model_name}_training_plots.png")
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                # Silent plot saving - no console spam during training
            else:
                plt.show()
                
        except Exception as e:
            print(f"⚠️  Plot generation failed: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get training summary statistics."""
        if not self.training_data:
            return {}
        
        df = pd.DataFrame(self.training_data)
        
        return {
            "total_logged_steps": len(self.training_data),
            "last_step": df['step'].iloc[-1],  # FIX: Verwende echten letzten Step aus Daten
            "current_loss": df['loss'].iloc[-1],
            "best_loss": df['loss'].min(),
            "avg_tokens_per_sec": df['tokens_per_sec'].mean(),
            "peak_tokens_per_sec": df['tokens_per_sec'].max(),
            "avg_step_time": df['step_time'].mean(),
            "total_elapsed_time": time.time() - self.start_time,
            "estimated_completion": self._estimate_completion()
        }
    
    def _estimate_completion(self) -> Optional[str]:
        """Estimate training completion time."""
        if not self.training_data or not self.metadata.get("total_steps"):
            return None
        
        df = pd.DataFrame(self.training_data)
        avg_step_time = df['step_time'].tail(10).mean()  # Last 10 steps
        remaining_steps = self.metadata["total_steps"] - self.last_log_step
        remaining_seconds = remaining_steps * avg_step_time
        
        hours = int(remaining_seconds // 3600)
        minutes = int((remaining_seconds % 3600) // 60)
        
        return f"{hours}h{minutes:02d}m"
    
    def load_from_file(self, log_file: str = None) -> bool:
        """Load training data from existing log file."""
        file_path = log_file or self.log_file
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            self.metadata = data.get("metadata", {})
            self.training_data = data.get("training_data", [])
            
            print(f"📊 Loaded {len(self.training_data)} log entries from {file_path}")
            return True
            
        except Exception as e:
            print(f"⚠️  Failed to load log file: {e}")
            return False

# Convenience functions for quick analysis
def analyze_training_log(log_file: str):
    """Quick analysis of training log."""
    logger = TrainingLogger()
    if logger.load_from_file(log_file):
        summary = logger.get_summary()
        print("\n📊 TRAINING SUMMARY:")
        for key, value in summary.items():
            print(f"   {key}: {value}")
        
        logger.generate_plots(save_plots=False)

def compare_training_runs(log_files: List[str]):
    """Compare multiple training runs."""
    if not MATPLOTLIB_AVAILABLE:
        print("⚠️  Matplotlib not available - cannot compare training runs")
        return

    plt.figure(figsize=(15, 5))
    
    for i, log_file in enumerate(log_files):
        logger = TrainingLogger()
        if logger.load_from_file(log_file):
            df = pd.DataFrame(logger.training_data)
            plt.subplot(1, 3, 1)
            plt.plot(df['step'], df['loss'], label=f"Run {i+1}", linewidth=2)
            plt.title('Loss Comparison')
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.yscale('log')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 3, 2)
            plt.plot(df['step'], df['tokens_per_sec'], label=f"Run {i+1}", linewidth=2)
            plt.title('Speed Comparison')
            plt.xlabel('Step')
            plt.ylabel('Tokens/sec')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 3, 3)
            plt.plot(df['step'], df['step_time'], label=f"Run {i+1}", linewidth=2)
            plt.title('Step Time Comparison')
            plt.xlabel('Step')
            plt.ylabel('Seconds/Step')
            plt.legend()
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    logger = TrainingLogger(model_name="test_model", log_every=10)
    logger.set_training_config(total_steps=1000, target_tokens=100_000_000)
    
    # Simulate some training steps
    for step in range(1, 101, 10):
        loss = 10.0 / (step + 1)  # Decreasing loss
        lr = 3e-4 * (1 - step/1000)  # Decaying LR
        tokens_per_sec = 8000 + step * 10  # Increasing speed
        step_time = 2.0 - step/1000  # Decreasing step time
        
        logger.log_step(step, loss, lr, tokens_per_sec, step_time, 
                       gpu_memory=12.5, clip_rate=0.15, flash_hit_rate=0.95)
    
    # Generate final summary
    summary = logger.get_summary()
    print("\n📊 FINAL SUMMARY:")
    for key, value in summary.items():
        print(f"   {key}: {value}")
