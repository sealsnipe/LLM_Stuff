"""
Professional Terminal Display System

Provides clean, compact terminal output for professional environments.
No emojis, minimal lines, inline status updates.
"""

import sys
import time
from typing import Optional, Dict


class ProfessionalDisplay:
    """Professional terminal display with inline updates."""
    
    def __init__(self):
        self.current_line = ""
        self.status_width = 80
        
    def print_header(self):
        """Print professional header."""
        print("=" * self.status_width)
        print("LLM Training Framework - Professional Edition")
        print("=" * self.status_width)
    
    def status_line(self, message: str, status: str = "LOADING"):
        """Print status line with inline updates."""
        if status == "LOADING":
            line = f"{message:<60} [{status}]"
        elif status == "COMPLETE":
            line = f"{message:<60} [COMPLETE]"
        elif status == "ERROR":
            line = f"{message:<60} [ERROR]"
        elif status == "WARNING":
            line = f"{message:<60} [WARNING]"
        else:
            line = f"{message:<60} [{status}]"
        
        # Clear current line and print new one
        if self.current_line:
            sys.stdout.write('\r' + ' ' * len(self.current_line) + '\r')
        
        sys.stdout.write(line)
        sys.stdout.flush()
        self.current_line = line
        
        if status in ["COMPLETE", "ERROR", "WARNING"]:
            print()  # New line for completed status
            self.current_line = ""
    
    def update_status(self, status: str):
        """Update current status inline."""
        if self.current_line:
            message = self.current_line.split('[')[0].strip()
            self.status_line(message, status)
    
    def print_info(self, message: str):
        """Print info message."""
        if self.current_line:
            print()  # Ensure we're on a new line
            self.current_line = ""
        print(f"INFO: {message}")
    
    def print_error(self, message: str):
        """Print error message."""
        if self.current_line:
            print()
            self.current_line = ""
        print(f"ERROR: {message}")
    
    def print_warning(self, message: str):
        """Print warning message."""
        if self.current_line:
            print()
            self.current_line = ""
        print(f"WARNING: {message}")


class TrainingProgressDisplay:
    """Professional training progress display."""

    def __init__(self):
        self.start_time = None
        self.last_update = 0
        self.start_step = None  # Track starting step for resume scenarios
        
    def start_training(self, total_steps: int):
        """Start training progress."""
        self.start_time = time.time()
        print(f"\nTraining started - Target steps: {total_steps:,}")
        print("-" * 80)
    
    def update_progress(self, step: int, total_steps: int, loss: float, lr: float,
                       gpu_memory: float, tokens_per_sec: float):
        """Update training progress inline."""
        # FIXED: Stronger rate limiting and step-based deduplication
        current_time = time.time()
        if current_time - self.last_update < 2.0:  # Limit to every 2 seconds
            return

        # FIXED: Prevent duplicate updates for same step
        if hasattr(self, 'last_displayed_step') and step == self.last_displayed_step:
            return

        self.last_displayed_step = step
            
        elapsed = time.time() - self.start_time
        progress_pct = (step / total_steps) * 100
        
        # FIXED: Calculate ETA correctly for resume scenarios
        if step > 0:
            # Track starting step for resume scenarios
            if self.start_step is None:
                self.start_step = step

            # Calculate steps completed since this session started
            steps_completed = step - self.start_step

            if steps_completed > 0 and elapsed > 0:
                steps_per_sec = steps_completed / elapsed
                remaining_steps = total_steps - step
                eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                eta_hours = int(eta_seconds // 3600)
                eta_mins = int((eta_seconds % 3600) // 60)
                eta = f"{eta_hours:02d}:{eta_mins:02d}h"
            else:
                eta = "--:--"
        else:
            eta = "--:--"
        
        # FIXED: Much shorter progress line without GPU memory
        progress_line = f"Step {step:5,}/{total_steps:,} ({progress_pct:4.1f}%) | Loss: {loss:5.3f} | {tokens_per_sec:5.0f} tok/s | ETA: {eta}"

        # FIXED: Always use \r for inline updates (no special first update)
        sys.stdout.write(f'\r{progress_line:<100}')  # Always use \r for inline updates
        sys.stdout.flush()

        self.last_update = current_time
    
    def finish_training(self):
        """Finish training progress."""
        print("\n" + "-" * 80)
        if self.start_time:
            total_time = time.time() - self.start_time
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            seconds = int(total_time % 60)
            print(f"Training completed in {hours:02d}:{minutes:02d}:{seconds:02d}")


# Global instances
display = ProfessionalDisplay()
progress = TrainingProgressDisplay()


def print_professional_header():
    """Print professional header."""
    display.print_header()


def status_update(message: str, status: str = "LOADING"):
    """Update status with professional display."""
    display.status_line(message, status)


def complete_status(status: str = "COMPLETE"):
    """Complete current status."""
    display.update_status(status)


def print_info(message: str):
    """Print info message."""
    display.print_info(message)


def print_error(message: str):
    """Print error message."""
    display.print_error(message)


def print_warning(message: str):
    """Print warning message."""
    display.print_warning(message)


def start_training_progress(total_steps: int):
    """Start training progress display."""
    progress.start_training(total_steps)


def update_training_progress(step: int, total_steps: int, loss: float, lr: float,
                           gpu_memory: float, tokens_per_sec: float):
    """Update training progress."""
    progress.update_progress(step, total_steps, loss, lr, gpu_memory, tokens_per_sec)


def finish_training_progress():
    """Finish training progress."""
    progress.finish_training()


def print_training_progress(step: int, total_steps: int, loss: float, lr: float, 
                          gpu_memory_gb: float, start_time: float, metrics=None, real_tokens=None):
    """Legacy compatibility function."""
    tokens_per_sec = 0
    if metrics:
        stats = metrics.get_stats()
        tokens_per_sec = stats.get('tokens_per_sec_mean', 0)
    
    update_training_progress(step, total_steps, loss, lr, gpu_memory_gb, tokens_per_sec)
