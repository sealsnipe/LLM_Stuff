"""
Progress Display Module

Contains the ProgressDisplay class and print_training_progress function.
Handles real-time training progress visualization with advanced metrics.
"""

import sys
import time
from typing import Optional
from config import training_config


class ProgressDisplay:
    """Klasse für erweiterte Progress-Anzeige."""
    
    def __init__(self):
        self.last_step = 0
        self.display_initialized = False
        self.start_step = None  # Track starting step for ETA calculation
    
    def print_progress(self, step: int, total_steps: int, loss: float, lr: float, 
                      gpu_memory_gb: float, start_time: float, metrics=None, real_tokens=None):
        """Production-Ready Progress-Anzeige mit Token-basierter Kontrolle."""

        # Berechne Statistiken
        progress_pct = (step / total_steps) * 100
        elapsed = time.time() - start_time

        # Token-basierte Progress falls verfügbar
        if real_tokens and training_config.use_token_based_schedule:
            token_progress_pct = (real_tokens / training_config.target_real_tokens) * 100
            progress_line = f"Progress: {step}/{total_steps} ({progress_pct:4.1f}%) | Tokens: {token_progress_pct:4.1f}%"
        else:
            progress_line = f"Progress: {step}/{total_steps} ({progress_pct:4.1f}%)"

        # ETA berechnen - FIXED: Use steps completed since start
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

                # FIXED: Better ETA formatting for long training runs
                if eta_seconds > 3600:  # More than 1 hour
                    eta_hours = int(eta_seconds // 3600)
                    eta_mins = int((eta_seconds % 3600) // 60)
                    eta = f"{eta_hours:02d}:{eta_mins:02d}h"
                else:  # Less than 1 hour
                    eta_mins = int(eta_seconds // 60)
                    eta_secs = int(eta_seconds % 60)
                    eta = f"{eta_mins:02d}:{eta_secs:02d}m"

                sec_per_step = 1.0 / steps_per_sec if steps_per_sec > 0 else 0

                # DEBUG: Print calculation details every 100 steps
                if step % 100 == 0:
                    print(f"\nDEBUG ETA: steps_completed={steps_completed}, elapsed={elapsed:.1f}s, steps_per_sec={steps_per_sec:.3f}, remaining={remaining_steps}, eta_seconds={eta_seconds:.0f}")
            else:
                eta = "  --  "
                sec_per_step = 0

            # Tokens/sec berechnen - FIXED: Use actual sequence length
            tokens_per_step = training_config.batch_size * training_config.gradient_accumulation_steps * training_config.sequence_length
            tokens_per_sec = tokens_per_step / sec_per_step if sec_per_step > 0 else 0
        else:
            eta = "  --  "
            sec_per_step = 0
            tokens_per_sec = 0

        # Info-Zeile
        info_line = f"Loss: {loss:5.3f} | {sec_per_step:4.1f}s/step | {tokens_per_sec:,.0f} tok/s | ETA: {eta} | LR: {lr:.2e}"

        # Advanced Metrics alle 100 Steps
        if step % 100 == 0 and step > 0 and metrics:
            try:
                from torch.backends.cuda import sdp_kernel
                flash_active = sdp_kernel.is_flash_sdp_enabled()
                stats = metrics.get_stats()

                flash_rate = stats.get('flash_hit_rate', 0) * 100
                clip_rate = stats.get('clip_rate', 0) * 100

                info_line += f" | Flash: {flash_rate:.0f}% | Clip: {clip_rate:.0f}%"
            except:
                pass

        # FIXED: Simplified single-line progress display
        if not self.display_initialized:
            # First output
            print("Training started - Target steps: {:,}".format(total_steps))
            print("-" * 80)
            self.display_initialized = True

        # FIXED: Much shorter progress line
        progress_text = f"Step {step:5,}/{total_steps:,} ({progress_pct:4.1f}%) | Loss: {loss:5.3f} | {tokens_per_sec:5.0f} tok/s | ETA: {eta}"

        # Clear line and write new progress
        sys.stdout.write(f"\r{progress_text:<80}")  # Much shorter padding
        sys.stdout.flush()
        self.last_step = step
    
    def finalize_display(self):
        """Beendet die Progress-Anzeige sauber."""
        if self.display_initialized:
            print()  # Neue Zeile für sauberen Abschluss
            print()  # Extra Zeile für Abstand
            self.display_initialized = False


def print_training_progress(step: int, total_steps: int, loss: float, lr: float,
                          gpu_memory_gb: float, start_time: float, metrics=None, real_tokens=None):
    """
    DEACTIVATED: Legacy progress display - now handled by professional_display.py
    This function is kept for backward compatibility but does nothing.
    """
    # DEACTIVATED: Prevent duplicate progress output
    # The new professional_display.py handles all progress display
    pass


def finalize_training_progress():
    """Beendet die globale Progress-Anzeige."""
    if hasattr(print_training_progress, '_display'):
        print_training_progress._display.finalize_display()


def format_time(seconds: float) -> str:
    """Formatiert Zeit in lesbares Format."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m{secs:02d}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h{mins:02d}m"


def format_number(num: float, precision: int = 1) -> str:
    """Formatiert große Zahlen mit K/M/B Suffixen."""
    if num >= 1e9:
        return f"{num/1e9:.{precision}f}B"
    elif num >= 1e6:
        return f"{num/1e6:.{precision}f}M"
    elif num >= 1e3:
        return f"{num/1e3:.{precision}f}K"
    else:
        return f"{num:.{precision}f}"


def calculate_eta(current_step: int, total_steps: int, elapsed_time: float) -> str:
    """Berechnet geschätzte verbleibende Zeit."""
    if current_step <= 0:
        return "Unknown"
    
    steps_per_sec = current_step / elapsed_time
    remaining_steps = total_steps - current_step
    
    if steps_per_sec <= 0:
        return "Unknown"
    
    eta_seconds = remaining_steps / steps_per_sec
    return format_time(eta_seconds)
