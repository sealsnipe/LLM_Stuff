"""
Training Interface Module

Contains the TrainingInterface class - the main high-level API for training.
This is the primary entry point for all training operations in the new architecture.
"""

import os
import time
from typing import Dict, Optional

from config import training_config
from ..models import MemoryOptimizedLLM
from ..training import Trainer
from ..data import DatasetFactory
from ..checkpoints import CheckpointManager
from ..monitoring import MemoryMonitor
from ..utils import GPUUtils, SystemUtils
from ..checkpoints.training_state import handle_training_mode_selection
from ..monitoring.professional_display import (
    print_professional_header, status_update, complete_status,
    print_info, print_error, print_warning
)


class TrainingInterface:
    """
    High-Level Training Interface - Haupteinstiegspunkt für Training.
    
    Diese Klasse stellt eine saubere, einfache API für das komplette Training bereit.
    Sie orchestriert alle Komponenten und versteckt die Komplexität der internen Architektur.
    """
    
    def __init__(self):
        self.model = None
        self.trainer = None
        self.dataloader = None
        self.training_mode = None
        
        # Komponenten
        self.gpu_utils = GPUUtils()
        self.system_utils = SystemUtils()
        self.memory_monitor = MemoryMonitor()
        self.checkpoint_manager = CheckpointManager()
        self.dataset_factory = DatasetFactory()
        
        # Status
        self.is_initialized = False
        self.training_stats = {}
    
    def initialize_training_environment(self):
        """Initialisiert die komplette Training-Umgebung."""
        print_professional_header()

        # System-Optimierungen
        status_update("Initializing system environment")
        self.system_utils.optimize_system_for_training()
        complete_status("COMPLETE")

        # GPU-Setup und Optimierung
        status_update("Configuring GPU settings")
        if not self.gpu_utils.check_gpu_setup():
            complete_status("ERROR")
            raise RuntimeError("GPU setup failed")

        self.gpu_utils.optimize_gpu_settings()
        complete_status("COMPLETE")

        self.is_initialized = True
    
    def setup_training_mode(self):
        """Setup Training Mode mit User-Interaktion."""
        if not self.is_initialized:
            self.initialize_training_environment()

        # Training Mode Selection
        status_update("Selecting training mode")
        self.training_mode = handle_training_mode_selection()
        complete_status("COMPLETE")

        return self.training_mode
    
    def create_model(self):
        """Erstellt das LLM-Model."""
        status_update("Creating model architecture")

        self.model = MemoryOptimizedLLM()

        # Move to GPU
        device = 'cuda' if self.gpu_utils.gpu_info['available'] else 'cpu'
        self.model = self.model.to(device)

        # Model Info
        total_params = sum(p.numel() for p in self.model.parameters())
        param_size = f"{total_params / 1e9:.2f}B" if total_params >= 1e9 else f"{total_params / 1e6:.0f}M"

        complete_status("COMPLETE")
        print_info(f"Model created: {param_size} parameters on {device}")
        return self.model
    
    def create_dataset(self, use_real_data=True, dataset_size="auto", cache_info=None):
        """Erstellt den Dataset mit Cache-Auswahl."""
        status_update("Loading dataset")

        # Auto-calculate samples if needed
        if dataset_size == "auto":
            num_samples = None  # Will be auto-calculated
        else:
            num_samples = None  # Let factory handle it

        # Try packed cache first, then FineWeb, then synthetic
        if use_real_data and training_config.use_packed_cache:
            # FIXED: Verwende spezifischen Cache-Pfad wenn verfügbar
            cache_path = None
            if cache_info:
                cache_path = cache_info.get('path')
                print(f"📦 Using cache: {cache_info.get('dataset_name')} (seq_len: {cache_info.get('sequence_length')})")

            self.dataloader = self.dataset_factory.create_packed_cache_dataset(cache_path)
            if self.dataloader:
                self._print_dataset_info(self.dataloader, "Packed Cache")
                complete_status("COMPLETE")
                return self.dataloader

        if use_real_data:
            self.dataloader = self.dataset_factory.create_fineweb_dataset(num_samples)
            if self.dataloader:
                self._print_dataset_info(self.dataloader, "FineWeb-Edu")
                complete_status("COMPLETE")
                return self.dataloader

        # Fallback to synthetic
        print_warning("Using synthetic dataset (no real data available)")
        self.dataloader = self.dataset_factory.create_synthetic_dataset(10000)
        self._print_dataset_info(self.dataloader, "Synthetic")

        complete_status("COMPLETE")

        # Show packing status
        if training_config.use_sequence_packing:
            print_info("Sequence packing enabled for improved efficiency")

        return self.dataloader

    def _print_dataset_info(self, dataloader, dataset_type):
        """Print detailed dataset information with epoch-based or token-based strategy."""
        from ..monitoring.professional_display import print_info

        # Calculate dataset metrics DIRECTLY from dataloader (RELIABLE)
        total_samples = len(dataloader.dataset)
        total_batches = len(dataloader)
        batch_size = dataloader.batch_size

        # Get sequence length from first batch (ACTUAL length)
        try:
            for batch in dataloader:
                if isinstance(batch, dict):
                    seq_length = batch['input_ids'].shape[1]
                else:
                    seq_length = batch[0].shape[1]
                # Propagate actual sequence length to trainer
                if hasattr(self, 'trainer') and self.trainer is not None:
                    self.trainer.actual_sequence_length = int(seq_length)
                break
        except:
            seq_length = training_config.sequence_length

        # Calculate REAL dataset tokens (not from buggy metadata)
        dataset_tokens = total_samples * seq_length

        print_info(f"Dataset: {dataset_type}")
        print_info(f"Samples: {total_samples:,} | Batches: {total_batches:,} | Seq Length: {seq_length}")
        print_info(f"Dataset tokens: {dataset_tokens:,}")

        if training_config.use_epoch_based_training:
            # Epoch-based training info with dataset fraction
            target_epochs = training_config.target_epochs
            epoch_fraction = training_config.epoch_dataset_fraction

            # Each epoch uses epoch_fraction of the dataset
            tokens_per_epoch = int(dataset_tokens * epoch_fraction)
            total_training_tokens = tokens_per_epoch * target_epochs

            # Calculate steps - FIXED: Use ACTUAL sequence length from dataloader
            effective_batch_size = training_config.batch_size * training_config.gradient_accumulation_steps
            tokens_per_step = effective_batch_size * seq_length  # Use REAL seq_length from cache!
            steps_per_epoch = tokens_per_epoch // tokens_per_step
            total_steps = steps_per_epoch * target_epochs

            print_info(f"Training Mode: EPOCH-BASED ({target_epochs} epochs)")
            print_info(f"Dataset fraction per epoch: {epoch_fraction:.1f} ({epoch_fraction*100:.0f}%)")
            print_info(f"Tokens per epoch: {tokens_per_epoch:,}")
            print_info(f"Total training tokens: {total_training_tokens:,}")
            print_info(f"Steps per epoch: {steps_per_epoch:,}")
            print_info(f"Total steps (current dataset): {total_steps:,}")

            # NOTE: Actual training steps will be calculated dynamically based on full dataset
        else:
            # Token-based training info
            target_tokens = training_config.target_tokens
            estimated_epochs = target_tokens / dataset_tokens
            total_steps = training_config.max_steps
            warmup_steps = training_config.warmup_steps

            print_info(f"Training Mode: TOKEN-BASED ({target_tokens:,} tokens)")
            print_info(f"Estimated epochs: {estimated_epochs:.1f}")
            print_info(f"Total steps: {total_steps:,}")
            print_info(f"Static warmup steps: {warmup_steps:,}")

    def create_trainer(self):
        """Erstellt den Trainer."""
        if self.model is None:
            raise RuntimeError("Model must be created first")

        status_update("Initializing trainer")

        device = 'cuda' if self.gpu_utils.gpu_info['available'] else 'cpu'
        self.trainer = Trainer(self.model, device=device)

        # Setup training components
        self.trainer.setup_training(self.training_mode)

        complete_status("COMPLETE")
        return self.trainer
    
    def start_training(self, use_real_data=True, dataset_size="auto", training_mode=None):
        """
        Startet das komplette Training.
        
        Args:
            use_real_data: Verwende echte FineWeb-Daten
            dataset_size: Größe des Datasets ("auto", "small", "medium", etc.)
            training_mode: Training Mode (None = User-Auswahl)
        
        Returns:
            Training-Statistiken
        """
        
        try:
            # 1. Environment Setup
            if not self.is_initialized:
                self.initialize_training_environment()
            
            # 2. Training Mode
            if training_mode:
                self.training_mode = training_mode
            else:
                self.setup_training_mode()
            
            # 3. Model Creation
            if self.model is None:
                self.create_model()
            
            # 4. Dataset Creation
            if self.dataloader is None:
                cache_info = self.training_mode.get('cache_info') if self.training_mode else None
                self.create_dataset(use_real_data, dataset_size, cache_info)
            
            # 5. Trainer Creation
            if self.trainer is None:
                self.create_trainer()
            
            # 6. Start Training
            status_update("Starting training pipeline")
            complete_status("COMPLETE")

            # Main Training Loop - FIXED: Use dynamic steps for epoch-based training
            if training_config.use_epoch_based_training:
                from ..utils.dataset_utils import get_dataset_calculator

                # FIXED: Use FRESH cache info calculation, same as dataset creation
                if training_config.use_packed_cache:
                    # Force fresh cache info calculation
                    from ..utils.dataset_utils import DatasetSizeCalculator
                    fresh_calculator = DatasetSizeCalculator()
                    fresh_cache_size = fresh_calculator.get_packed_cache_size()
                    if fresh_cache_size:
                        total_sequences, total_tokens = fresh_cache_size
                        cache_info = {
                            'total_sequences': total_sequences,
                            'total_tokens': total_tokens,
                            'sequence_length': 512,
                            'dataset_name': 'FineWeb'
                        }
                        print(f"🔄 Using FRESH cache info: {total_sequences:,} sequences")
                    else:
                        cache_info = None
                        print(f"🔄 No fresh cache info available")
                else:
                    # Use cache info from training mode
                    cache_info = self.training_mode.get('cache_info') if self.training_mode else None

                # FIXED: Only reset cache_info if we're NOT using packed cache AND NOT loading checkpoint
                # If we're using packed cache, always keep cache_info for correct step calculation
                if (self.training_mode.get('mode') != 'checkpoint' and
                    not training_config.use_packed_cache):
                    cache_info = None  # Reset cache_info for FineWeb-Edu training

                print(f"🔍 Debug: mode={self.training_mode.get('mode')}, use_packed_cache={training_config.use_packed_cache}, cache_info={'present' if cache_info else 'None'}")

                # FIXED: Ensure trainer uses the same cache_info
                if self.trainer:
                    self.trainer.cache_info = cache_info
                    print(f"🔄 Updated trainer cache_info: {'present' if cache_info else 'None'}")

                calculator = get_dataset_calculator(cache_info)
                dynamic_total_steps, _ = calculator.calculate_epoch_based_steps(training_config.target_epochs)

                # Calculate warmup based on ACTUAL training steps (not dataset display steps)
                warmup_steps = calculator.calculate_dynamic_warmup_steps(dynamic_total_steps)

                print_info(f"Dynamic warmup steps: {warmup_steps:,} ({warmup_steps/dynamic_total_steps*100:.1f}%)")
                print_info(f"Actual training steps: {dynamic_total_steps:,} (full dataset estimate)")
                self.training_stats = self.trainer.train(
                    dataloader=self.dataloader,
                    total_steps=dynamic_total_steps
                )
            else:
                self.training_stats = self.trainer.train(
                    dataloader=self.dataloader,
                    total_steps=training_config.max_steps
                )

            print_info("Training completed successfully")
            return self.training_stats
            
        except KeyboardInterrupt:
            print_warning("Training interrupted by user")
            return self.get_current_stats()

        except Exception as e:
            print_error(f"Training failed: {e}")
            raise
    
    def resume_training(self, checkpoint_info):
        """Setzt Training von einem Checkpoint fort."""
        print(f"🔄 Setze Training fort von: {checkpoint_info['filename']}")
        
        # Load checkpoint
        model, optimizer, start_step, model_name, run_id = self.checkpoint_manager.load_checkpoint(checkpoint_info)
        
        self.model = model
        
        # Create trainer with loaded components
        device = 'cuda' if self.gpu_utils.gpu_info['available'] else 'cpu'
        self.trainer = Trainer(self.model, device=device)
        self.trainer.optimizer = optimizer
        self.trainer.current_step = start_step
        
        # Setup training mode for resume
        self.training_mode = {
            'mode': 'checkpoint',
            'checkpoint_info': checkpoint_info
        }
        
        print(f"✅ Training wird fortgesetzt ab Step {start_step:,}")
        return model, optimizer, start_step, model_name, run_id
    
    def get_current_stats(self):
        """Gibt aktuelle Training-Statistiken zurück."""
        if self.trainer:
            return self.trainer.get_training_stats()
        return {}
    
    def get_system_status(self):
        """Gibt umfassenden System-Status zurück."""
        return {
            'initialized': self.is_initialized,
            'gpu_info': self.gpu_utils.gpu_info,
            'memory_stats': self.memory_monitor.get_memory_stats(),
            'system_info': self.system_utils.get_system_info(),
            'training_mode': self.training_mode,
            'model_loaded': self.model is not None,
            'trainer_ready': self.trainer is not None,
            'dataset_ready': self.dataloader is not None
        }
    
    def cleanup(self):
        """Bereinigt Ressourcen."""
        if self.memory_monitor:
            self.memory_monitor.cleanup_memory()
        
        print("🧹 Training-Interface bereinigt")


# Convenience Functions für einfache Nutzung
def start_training(use_real_data=True, dataset_size="auto"):
    """Convenience function für schnelles Training-Start."""
    interface = TrainingInterface()
    return interface.start_training(use_real_data=use_real_data, dataset_size=dataset_size)


def quick_training_setup():
    """Schnelles Training-Setup für Entwicklung."""
    interface = TrainingInterface()
    interface.initialize_training_environment()
    return interface
