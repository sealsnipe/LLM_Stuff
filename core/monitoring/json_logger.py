"""
Professional JSON Training Logger

Logs training metrics to JSON files for analysis and monitoring.
Compatible with existing log format.
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional
from config import training_config


class JSONTrainingLogger:
    """Professional JSON logger for training metrics."""
    
    def __init__(self, model_name: str, run_id: int, total_steps: int, target_tokens: Optional[int] = None):
        self.model_name = model_name
        self.run_id = run_id
        self.total_steps = total_steps
        self.target_tokens = target_tokens
        self.start_time = datetime.now()
        self.training_data = []
        
        # Create log directory
        self.log_dir = "training_logs"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Log file paths
        self.log_file = os.path.join(self.log_dir, f"{model_name}_run_{run_id}.json")
        self.summary_file = os.path.join(self.log_dir, f"{model_name}_run_{run_id}_summary.json")
        
        # Initialize log files
        self._initialize_logs()
    
    def _initialize_logs(self):
        """Initialize log files with metadata."""
        metadata = {
            "model_name": self.model_name,
            "start_time": self.start_time.isoformat(),
            "log_every": training_config.log_every,
            "total_steps": self.total_steps,
            "target_tokens": self.target_tokens
        }
        
        # Create main log file
        log_data = {
            "metadata": metadata,
            "training_data": []
        }
        
        with open(self.log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        # Create summary file
        with open(self.summary_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def log_step(self, step: int, loss: float, lr: float, tokens_per_sec: float,
                 step_time: float, gpu_memory_gb: float, real_tokens: int,
                 clip_rate: float = 0.0, flash_hit_rate: float = 0.0):
        """Log a training step."""
        
        current_time = datetime.now()
        elapsed_time = (current_time - self.start_time).total_seconds()
        progress_pct = (step / self.total_steps) * 100 if self.total_steps > 0 else 0
        
        step_data = {
            "step": step,
            "timestamp": current_time.isoformat(),
            "elapsed_time": elapsed_time,
            "loss": loss,
            "lr": lr,
            "tokens_per_sec": tokens_per_sec,
            "step_time": step_time,
            "progress_pct": progress_pct,
            "gpu_memory_gb": gpu_memory_gb,
            "clip_rate": clip_rate,
            "flash_hit_rate": flash_hit_rate,
            "real_tokens": real_tokens
        }
        
        self.training_data.append(step_data)
        
        # Update log file
        self._update_log_file()
    
    def _update_log_file(self):
        """Update the main log file with new data."""
        try:
            # Read existing data
            with open(self.log_file, 'r') as f:
                log_data = json.load(f)
            
            # Update training data
            log_data["training_data"] = self.training_data
            
            # Write back
            with open(self.log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
                
        except Exception as e:
            # Fallback: create new file
            log_data = {
                "metadata": {
                    "model_name": self.model_name,
                    "start_time": self.start_time.isoformat(),
                    "log_every": training_config.log_every,
                    "total_steps": self.total_steps,
                    "target_tokens": self.target_tokens
                },
                "training_data": self.training_data
            }
            
            with open(self.log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
    
    def finalize_logs(self, final_step: int, final_loss: float, training_time: float):
        """Finalize logs at end of training."""
        
        # Update summary with final stats
        summary_data = {
            "model_name": self.model_name,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "log_every": training_config.log_every,
            "total_steps": self.total_steps,
            "final_step": final_step,
            "target_tokens": self.target_tokens,
            "final_loss": final_loss,
            "training_time_seconds": training_time,
            "training_time_hours": training_time / 3600,
            "total_data_points": len(self.training_data)
        }
        
        with open(self.summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current training statistics."""
        if not self.training_data:
            return {}
        
        latest = self.training_data[-1]
        
        # Calculate averages
        losses = [d['loss'] for d in self.training_data]
        tokens_per_sec = [d['tokens_per_sec'] for d in self.training_data]
        
        return {
            "current_step": latest['step'],
            "current_loss": latest['loss'],
            "current_lr": latest['lr'],
            "current_tokens_per_sec": latest['tokens_per_sec'],
            "avg_loss": sum(losses) / len(losses),
            "avg_tokens_per_sec": sum(tokens_per_sec) / len(tokens_per_sec),
            "elapsed_time": latest['elapsed_time'],
            "progress_pct": latest['progress_pct'],
            "total_logged_steps": len(self.training_data)
        }


# Global logger instance
_global_logger: Optional[JSONTrainingLogger] = None


def initialize_json_logger(model_name: str, run_id: int, total_steps: int, target_tokens: Optional[int] = None):
    """Initialize global JSON logger."""
    global _global_logger
    # FIXED: Always create new logger (important for resume with different total_steps)
    _global_logger = JSONTrainingLogger(model_name, run_id, total_steps, target_tokens)
    return _global_logger


def log_training_step(step: int, loss: float, lr: float, tokens_per_sec: float,
                     step_time: float, gpu_memory_gb: float, real_tokens: int,
                     clip_rate: float = 0.0, flash_hit_rate: float = 0.0):
    """Log a training step to JSON."""
    if _global_logger:
        _global_logger.log_step(step, loss, lr, tokens_per_sec, step_time, 
                               gpu_memory_gb, real_tokens, clip_rate, flash_hit_rate)


def finalize_json_logs(final_step: int, final_loss: float, training_time: float):
    """Finalize JSON logs."""
    if _global_logger:
        _global_logger.finalize_logs(final_step, final_loss, training_time)


def get_json_logger_stats() -> Dict[str, Any]:
    """Get current logger statistics."""
    if _global_logger:
        return _global_logger.get_stats()
    return {}
