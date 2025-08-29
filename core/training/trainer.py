"""
Main Trainer Module

Contains the Trainer class that orchestrates the complete training process.
This is the core training engine that coordinates all components.
"""

import torch
import torch.nn as nn
import time
import os
import sys
import gc
from typing import Dict, Optional

from config import training_config, model_config
from .metrics import AdvancedMetrics
from .optimizers import create_optimizer
from .schedulers import create_lr_scheduler, get_current_lr
from ..monitoring import print_training_progress, MemoryMonitor
from ..checkpoints import CheckpointManager
from ..data import DatasetFactory


class Trainer:
    """Main Training Engine für Memory-Optimized LLM Training."""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        
        # Monitoring
        self.metrics = AdvancedMetrics()
        self.memory_monitor = MemoryMonitor()
        
        # Checkpoint management
        self.checkpoint_manager = CheckpointManager()
        
        # Training state
        self.current_step = 0  # Optimizer steps (parameter updates)
        self.micro_step = 0    # Micro-steps (per-batch before accumulation)
        self.start_time = None
        self.training_mode = None
        self.cache_info = None
        self.warmup_steps = None  # Store warmup steps for status display

        # Actual sequence length used by the active dataset/cache
        self.actual_sequence_length = training_config.sequence_length
        
        # Setup mixed precision if enabled
        if training_config.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
    
    def setup_training(self, training_mode: Dict):
        """Setup training components based on training mode."""
        self.training_mode = training_mode
        self.cache_info = training_mode.get('cache_info')

        # FIXED: Update cache_info with current registry data (for expanded datasets)
        if self.cache_info:
            from ..utils.cache_registry import load_cache_registry
            current_caches = load_cache_registry()
            for current_cache in current_caches:
                if (current_cache['dataset_name'] == self.cache_info['dataset_name'] and
                    current_cache['sequence_length'] == self.cache_info['sequence_length']):
                    old_sequences = self.cache_info.get('total_sequences', 0)
                    new_sequences = current_cache.get('total_sequences', 0)

                    if new_sequences > old_sequences:
                        print(f"🔧 Trainer: Cache expanded {old_sequences:,} → {new_sequences:,} sequences")
                        self.cache_info.update(current_cache)  # Update with current data
                    break
        
        # Create optimizer
        self.optimizer = create_optimizer(self.model)
        
        # Create scheduler (will be updated with actual steps later)
        self.scheduler = create_lr_scheduler(self.optimizer, training_config.max_steps)

        # Store warmup steps for status display
        if training_config.use_epoch_based_training:
            self.warmup_steps = training_config.dynamic_warmup_steps
        else:
            self.warmup_steps = training_config.warmup_steps
        
        # Setup torch compile if enabled and triton available
        if training_config.use_torch_compile:
            try:
                # Check if triton is available
                import triton

                # Configure dynamo for Windows
                import torch._dynamo
                torch._dynamo.config.suppress_errors = True
                torch._dynamo.config.verbose = False

                self.model = torch.compile(
                    self.model,
                    mode=training_config.torch_compile_mode,
                    dynamic=False
                )
            except ImportError:
                # Triton not available - skip compile silently
                pass
            except Exception as e:
                # Other compile errors - skip silently
                pass
    
    def load_checkpoint_if_needed(self):
        """Load checkpoint if resuming training."""
        if self.training_mode['mode'] == 'checkpoint':
            checkpoint_info = self.training_mode['checkpoint_info']

            # Load model and optimizer state
            checkpoint = torch.load(checkpoint_info['filepath'], map_location='cpu', weights_only=False)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.current_step = checkpoint['step']

            # FIXED: Load scheduler state if available
            if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"✅ Scheduler state loaded (continuing from step {self.current_step})")

                # Show warmup status
                warmup_info = self._get_warmup_status()
                if warmup_info:
                    print(f"📈 {warmup_info}")

            print(f"✅ Checkpoint geladen: Step {self.current_step:,}")
            return checkpoint_info['model_name'], checkpoint_info['run_id']
        else:
            # New training - generate model name and run ID
            if hasattr(self.training_mode, 'get') and 'model_name' in self.training_mode:
                model_name = self.training_mode['model_name']
            else:
                model_name = "415M_FineWeb_512"  # Default fallback

            run_id = 1  # Start with run 1 for new training
            print(f"🚀 Starting new training: {model_name} (Run {run_id})")
            return model_name, run_id

    def _get_warmup_status(self):
        """Get warmup status information from stored warmup steps."""
        try:
            # Use stored warmup steps (set during setup)
            if self.warmup_steps is None:
                return "Warmup status unknown (warmup_steps not set)"

            warmup_steps = self.warmup_steps
            current_step = self.current_step

            if current_step < warmup_steps:
                remaining_steps = warmup_steps - current_step
                progress_pct = (current_step / warmup_steps) * 100
                return f"Warmup: {current_step:,}/{warmup_steps:,} steps ({progress_pct:.1f}%) - {remaining_steps:,} steps remaining"
            else:
                return f"Warmup completed at step {warmup_steps:,} - now in main training phase"

        except Exception as e:
            return f"Warmup status unknown (error: {e})"
        else:
            # New training
            from ..checkpoints.training_state import get_next_run_id
            model_name = self.training_mode['model_name']
            run_id = get_next_run_id(model_name)
            return model_name, run_id
    
    def train_step(self, batch):
        """Execute a single micro-step (forward/backward). Optimizer step happens in optimizer_step()."""
        step_start_time = time.time()

        # Handle different batch formats
        if isinstance(batch, dict):
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            labels = input_ids.clone()
        else:
            input_ids, labels = [x.to(self.device, non_blocking=True) for x in batch]

        # Forward pass with mixed precision
        if training_config.use_mixed_precision and self.scaler:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = self.model(input_ids, labels=labels)
                loss = outputs['loss']

            # Scale loss for gradient accumulation
            scaled_loss = loss / training_config.gradient_accumulation_steps
            self.scaler.scale(scaled_loss).backward()
        else:
            outputs = self.model(input_ids, labels=labels)
            loss = outputs['loss']

            # Scale loss for gradient accumulation
            scaled_loss = loss / training_config.gradient_accumulation_steps
            scaled_loss.backward()

        # Calculate step metrics
        step_time = time.time() - step_start_time

        # Update micro-step counter
        self.micro_step += 1

        # Only compute tokens/sec on optimizer steps (stable throughput)
        if self.micro_step % training_config.gradient_accumulation_steps == 0:
            effective_batch_size = training_config.batch_size * training_config.gradient_accumulation_steps
            tokens_per_step = effective_batch_size * self.actual_sequence_length
            tokens_per_sec = tokens_per_step / step_time if step_time > 0 else 0
        else:
            tokens_per_sec = 0

        # Get gradient norm on optimizer steps only
        grad_norm = 0.0
        if self.micro_step % training_config.gradient_accumulation_steps == 0:
            if training_config.use_mixed_precision and self.scaler:
                self.scaler.unscale_(self.optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                training_config.max_grad_norm
            ).item()

        # Check for gradient clipping and NaN
        was_clipped = grad_norm > training_config.max_grad_norm
        has_nan = torch.isnan(loss).item()

        # Update metrics
        self.metrics.update(
            step_time=step_time,
            tokens_per_sec=tokens_per_sec,
            grad_norm=grad_norm,
            was_clipped=was_clipped,
            flash_active=True,
            has_nan=has_nan
        )

        return loss.item(), grad_norm
    
    def optimizer_step(self):
        """Execute optimizer step with gradient accumulation.
        Increments current_step only when an optimizer update happens."""
        if self.micro_step % training_config.gradient_accumulation_steps == 0:
            if training_config.use_mixed_precision and self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad(set_to_none=True)

            if self.scheduler:
                self.scheduler.step()

            # Count an optimizer step
            self.current_step += 1
    
    def should_log(self):
        """Check if we should log this step."""
        return self.current_step % training_config.log_interval == 0
    
    def should_save_checkpoint(self):
        """Check if we should save checkpoint this step."""
        return self.current_step % training_config.save_interval == 0
    
    def log_progress(self, loss, total_steps):
        """Log training progress."""
        if self.should_log():
            lr = get_current_lr(self.scheduler) if self.scheduler else training_config.learning_rate
            gpu_memory = self.memory_monitor.get_memory_stats()['gpu_allocated']

            # Calculate tokens per second and step time
            tokens_per_sec = 0
            step_time = 0
            if self.metrics:
                stats = self.metrics.get_stats()
                tokens_per_sec = stats.get('tokens_per_sec_mean', 0)
                step_time = stats.get('step_time_mean', 0)

            # Calculate real tokens processed (optimizer steps only)
            effective_batch_size = training_config.batch_size * training_config.gradient_accumulation_steps
            real_tokens = self.current_step * effective_batch_size * self.actual_sequence_length

            # Update professional display
            from ..monitoring.professional_display import update_training_progress
            update_training_progress(
                step=self.current_step,
                total_steps=total_steps,
                loss=loss,
                lr=lr,
                gpu_memory=gpu_memory,
                tokens_per_sec=tokens_per_sec
            )

            # Log to JSON
            from ..monitoring.json_logger import log_training_step
            log_training_step(
                step=self.current_step,
                loss=loss,
                lr=lr,
                tokens_per_sec=tokens_per_sec,
                step_time=step_time,
                gpu_memory_gb=gpu_memory,
                real_tokens=real_tokens,
                clip_rate=0.0,  # TODO: Add gradient clipping stats
                flash_hit_rate=0.0  # TODO: Add flash attention stats
            )
    
    def cleanup_memory(self):
        """Cleanup memory periodically."""
        if self.current_step % 100 == 0:
            self.memory_monitor.cleanup_memory()
    
    def get_training_stats(self):
        """Get current training statistics."""
        return {
            'step': self.current_step,
            'metrics': self.metrics.get_stats(),
            'memory': self.memory_monitor.get_memory_stats(),
            'lr': get_current_lr(self.scheduler) if self.scheduler else training_config.learning_rate
        }

    def train(self, dataloader, total_steps=None):
        """Main training loop."""
        if total_steps is None:
            # FIXED: Always use current dynamic max_steps with cache info
            if training_config.use_epoch_based_training:
                from ..utils.dataset_utils import get_dataset_calculator
                calculator = get_dataset_calculator(self.cache_info)
                total_steps, _ = calculator.calculate_epoch_based_steps(training_config.target_epochs)
            else:
                total_steps = training_config.max_steps



        # Setup training
        model_name, run_id = self.load_checkpoint_if_needed()
        self.start_time = time.time()

        # Update scheduler with actual total_steps and warmup_steps
        self.scheduler = create_lr_scheduler(self.optimizer, total_steps)

        # Update warmup steps for status display
        if training_config.use_epoch_based_training:
            from ..utils.dataset_utils import get_dataset_calculator
            calculator = get_dataset_calculator(self.cache_info)
            self.warmup_steps = calculator.calculate_dynamic_warmup_steps(total_steps)
        else:
            self.warmup_steps = training_config.warmup_steps

        from ..monitoring.professional_display import start_training_progress
        from ..monitoring.json_logger import initialize_json_logger

        start_training_progress(total_steps)

        # Initialize JSON logger (use effective tokens per optimizer step)
        effective_batch_size = training_config.batch_size * training_config.gradient_accumulation_steps
        target_tokens = total_steps * effective_batch_size * self.actual_sequence_length
        self.json_logger = initialize_json_logger(model_name, run_id, total_steps, target_tokens)

        # Training loop
        self.model.train()
        epoch = 0
        final_loss = 0.0  # Initialize final_loss
        training_time = 0.0  # Initialize training_time
        checkpoint_saved = False  # Track if checkpoint was saved

        try:
            while self.current_step < total_steps:
                epoch += 1

                for batch_idx, batch in enumerate(dataloader):
                    if self.current_step >= total_steps:
                        break

                    # Training step (micro step)
                    loss, grad_norm = self.train_step(batch)

                    # Optimizer step (with gradient accumulation)
                    self.optimizer_step()

                    # Update final_loss for potential interruption
                    final_loss = loss

                    # Logging
                    self.log_progress(loss, total_steps)

                    # Checkpointing
                    if self.should_save_checkpoint():
                        self.checkpoint_manager.save_checkpoint(
                            model=self.model,
                            optimizer=self.optimizer,
                            step=self.current_step,
                            loss=loss,
                            model_name=model_name,
                            run_id=run_id,
                            cache_info=self.cache_info,
                            scheduler=self.scheduler  # FIXED: Include scheduler
                        )

                    # Memory cleanup
                    self.cleanup_memory()

                pass  # Silent epoch completion

        except KeyboardInterrupt:
            from ..monitoring.professional_display import print_warning
            print_warning(f"Training interrupted at step {self.current_step:,}")

            # Calculate training time for interruption
            training_time = time.time() - self.start_time

            # Save checkpoint on interruption
            self._save_checkpoint_on_exit(model_name, run_id, final_loss, training_time)
            checkpoint_saved = True

        except Exception as e:
            from ..monitoring.professional_display import print_error
            print_error(f"Training error: {e}")

            # Calculate training time for error
            training_time = time.time() - self.start_time

            # Save checkpoint on error
            self._save_checkpoint_on_exit(model_name, run_id, final_loss, training_time)
            checkpoint_saved = True
            raise

        finally:
            # Final checkpoint
            if self.current_step > 0:
                final_loss = self.get_training_stats().get('last_loss', 0.0)
                training_time = time.time() - self.start_time

                from ..checkpoints.model_saver import save_trained_model
                from ..monitoring.professional_display import finish_training_progress
                from ..monitoring.json_logger import finalize_json_logs

                # Save final checkpoint first (only if not already saved)
                if not checkpoint_saved:
                    self._save_checkpoint_on_exit(model_name, run_id, final_loss, training_time)

                # Save trained model
                save_trained_model(
                    model=self.model,
                    step=self.current_step,
                    final_loss=final_loss,
                    training_time=training_time
                )

                # Finalize JSON logs
                finalize_json_logs(self.current_step, final_loss, training_time)

                # Create training plots
                try:
                    from ..monitoring.training_plotter import create_plots_for_latest_run
                    plot_path = create_plots_for_latest_run(model_name)
                    if plot_path:
                        from ..monitoring.professional_display import print_info
                        print_info(f"Training plots saved: {plot_path}")
                except Exception as e:
                    from ..monitoring.professional_display import print_warning
                    print_warning(f"Failed to create plots: {e}")

                finish_training_progress()

        return self.get_training_stats()

    def _save_checkpoint_on_exit(self, model_name: str, run_id: int, final_loss: float, training_time: float):
        """Save checkpoint on training exit (interruption or completion)."""
        try:
            from ..checkpoints.checkpoint_manager import CheckpointManager

            checkpoint_manager = CheckpointManager()

            # Save checkpoint with run info
            checkpoint_path = checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                step=self.current_step,
                loss=final_loss,
                model_name=model_name,
                run_id=run_id,
                cache_info=self.cache_info,
                scheduler=self.scheduler  # FIXED: Include scheduler
            )

            from ..monitoring.professional_display import print_info
            print_info(f"Checkpoint saved: {checkpoint_path}")

        except Exception as e:
            from ..monitoring.professional_display import print_warning
            print_warning(f"Failed to save checkpoint: {e}")
