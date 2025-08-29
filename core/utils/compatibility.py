"""
Compatibility Module

Contains compatibility layers and adapters for backward compatibility.
Ensures smooth transition from the monolithic training-windows.py to the new modular architecture.
"""

from config import model_config, training_config


class CompatConfig:
    """
    Compatibility layer für die alte Config-Struktur.
    Stellt die gleichen Attribute bereit wie die ursprüngliche training-windows.py.
    """
    
    def __init__(self):
        # Model settings - direkt von model_config
        self.vocab_size = model_config.vocab_size
        self.hidden_size = model_config.hidden_size
        self.num_layers = model_config.num_layers
        self.num_attention_heads = model_config.num_attention_heads
        self.num_key_value_heads = model_config.num_key_value_heads
        self.tie_word_embeddings = model_config.tie_word_embeddings

        # Training settings - direkt von training_config
        self.max_steps = training_config.max_steps
        self.batch_size = training_config.batch_size
        self.gradient_accumulation_steps = training_config.gradient_accumulation_steps
        self.sequence_length = training_config.sequence_length
        self.learning_rate = training_config.learning_rate
        self.weight_decay = training_config.weight_decay
        self.max_grad_norm = training_config.max_grad_norm
        self.use_torch_compile = training_config.use_torch_compile
        self.use_mixed_precision = training_config.use_mixed_precision
        self.use_activation_checkpointing = training_config.use_activation_checkpointing
        self.use_gradient_checkpointing = training_config.use_activation_checkpointing
        self.log_interval = training_config.log_interval
        self.adam_beta1 = training_config.adam_beta1
        self.adam_beta2 = training_config.adam_beta2
        self.adam_eps = training_config.adam_eps


class LegacyFunctionAdapter:
    """
    Adapter für Legacy-Funktionen aus der ursprünglichen training-windows.py.
    Stellt die gleichen Funktionssignaturen bereit, leitet aber an neue Module weiter.
    """
    
    @staticmethod
    def memory_optimized_training_loop(use_real_data=True, dataset_size="medium", training_mode=None):
        """Legacy-Wrapper für die neue Training-Pipeline."""
        from ..interfaces.training_interface import TrainingInterface
        
        # Erstelle Training Interface
        trainer_interface = TrainingInterface()
        
        # Konvertiere Legacy-Parameter
        if training_mode is None:
            from ..checkpoints.training_state import handle_training_mode_selection
            training_mode = handle_training_mode_selection()
        
        # Starte Training
        return trainer_interface.start_training(
            use_real_data=use_real_data,
            dataset_size=dataset_size,
            training_mode=training_mode
        )
    
    @staticmethod
    def create_gpu_optimized_dataset(num_samples=None, use_real_data=True, dataset_size="auto", return_splits=False):
        """Legacy-Wrapper für Dataset-Erstellung."""
        from ..data.dataset_factory import create_gpu_optimized_dataset as new_create_dataset
        
        return new_create_dataset(
            num_samples=num_samples,
            use_real_data=use_real_data,
            dataset_size=dataset_size,
            return_splits=return_splits
        )
    
    @staticmethod
    def save_trained_model(model, step, final_loss, training_time, save_dir=None):
        """Legacy-Wrapper für Model Saving."""
        from ..checkpoints.model_saver import save_trained_model as new_save_model
        
        return new_save_model(
            model=model,
            step=step,
            final_loss=final_loss,
            training_time=training_time,
            save_dir=save_dir
        )
    
    @staticmethod
    def validate_saved_model(model_path):
        """Legacy-Wrapper für Model Validation."""
        from ..checkpoints.model_saver import validate_saved_model as new_validate_model
        
        return new_validate_model(model_path)
    
    @staticmethod
    def handle_training_mode_selection():
        """Legacy-Wrapper für Training Mode Selection."""
        from ..checkpoints.training_state import handle_training_mode_selection as new_handle_selection
        
        return new_handle_selection()
    
    @staticmethod
    def check_gpu_setup():
        """Legacy-Wrapper für GPU Setup Check."""
        from ..utils.gpu_utils import check_gpu_setup as new_check_gpu
        
        return new_check_gpu()
    
    @staticmethod
    def print_training_progress(step, total_steps, loss, lr, gpu_memory_gb, start_time, metrics=None, real_tokens=None):
        """Legacy-Wrapper für Progress Display."""
        from ..monitoring.progress_display import print_training_progress as new_print_progress
        
        return new_print_progress(
            step=step,
            total_steps=total_steps,
            loss=loss,
            lr=lr,
            gpu_memory_gb=gpu_memory_gb,
            start_time=start_time,
            metrics=metrics,
            real_tokens=real_tokens
        )


class GlobalVariableAdapter:
    """
    Adapter für globale Variablen aus der ursprünglichen training-windows.py.
    Stellt Backward Compatibility für Code bereit, der auf globale Variablen zugreift.
    """
    
    def __init__(self):
        # Globale Variablen aus training-windows.py
        self._dynamic_max_steps = None
        self._intelligent_training_config = None
        self._dataloader_info = None
        
        # Fused Operations Availability
        self.FUSED_OPS_AVAILABLE = self._check_fused_ops()
    
    def _check_fused_ops(self):
        """Prüft Verfügbarkeit von Fused Operations."""
        try:
            from torch.nn.utils.fusion import fuse_conv_bn_eval
            return True
        except ImportError:
            return False
    
    @property
    def dynamic_max_steps(self):
        return self._dynamic_max_steps
    
    @dynamic_max_steps.setter
    def dynamic_max_steps(self, value):
        self._dynamic_max_steps = value
    
    @property
    def intelligent_training_config(self):
        return self._intelligent_training_config
    
    @intelligent_training_config.setter
    def intelligent_training_config(self, value):
        self._intelligent_training_config = value
    
    @property
    def dataloader_info(self):
        return self._dataloader_info
    
    @dataloader_info.setter
    def dataloader_info(self, value):
        self._dataloader_info = value


# Globale Instanzen für Backward Compatibility
_global_adapter = GlobalVariableAdapter()
_legacy_adapter = LegacyFunctionAdapter()

# Exportiere Legacy-Funktionen auf Modul-Ebene
memory_optimized_training_loop = _legacy_adapter.memory_optimized_training_loop
create_gpu_optimized_dataset = _legacy_adapter.create_gpu_optimized_dataset
save_trained_model = _legacy_adapter.save_trained_model
validate_saved_model = _legacy_adapter.validate_saved_model
handle_training_mode_selection = _legacy_adapter.handle_training_mode_selection
check_gpu_setup = _legacy_adapter.check_gpu_setup
print_training_progress = _legacy_adapter.print_training_progress

# Exportiere globale Variablen
FUSED_OPS_AVAILABLE = _global_adapter.FUSED_OPS_AVAILABLE


def get_legacy_config():
    """Gibt Legacy-kompatible Config zurück."""
    return CompatConfig()


def migrate_from_legacy():
    """
    Hilfsfunktion für Migration von Legacy-Code.
    Gibt Empfehlungen für die Umstellung auf die neue Architektur.
    """
    
    migration_guide = """
    🔄 MIGRATION GUIDE: Von training-windows.py zur neuen Architektur
    
    Alte Imports:
    from training-windows import memory_optimized_training_loop, MemoryOptimizedLLM
    
    Neue Imports:
    from core.interfaces import TrainingInterface
    from core.models import MemoryOptimizedLLM
    
    Alte Nutzung:
    memory_optimized_training_loop(use_real_data=True)
    
    Neue Nutzung:
    trainer = TrainingInterface()
    trainer.start_training(use_real_data=True)
    
    Vorteile der neuen Architektur:
    ✅ Modulare Struktur - einfacher zu testen und erweitern
    ✅ Klare Trennung der Verantwortlichkeiten
    ✅ Bessere Code-Organisation
    ✅ Einfachere Wartung und Debugging
    ✅ Wiederverwendbare Komponenten
    """
    
    print(migration_guide)
    return migration_guide
