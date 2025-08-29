"""
Model Interface Module

Contains the ModelInterface class - high-level API for model management.
Handles model creation, loading, saving, and inference operations.
"""

import torch
import os
from typing import Dict, Optional, List, Union

from config import model_config
from ..models import MemoryOptimizedLLM
from ..checkpoints import ModelSaver, CheckpointManager
from ..utils import GPUUtils


class ModelInterface:
    """
    High-Level Model Interface - API für Model Management.
    
    Diese Klasse stellt eine saubere API für alle Model-Operationen bereit:
    - Model-Erstellung und -Konfiguration
    - Model-Laden und -Speichern
    - Inference und Text-Generation
    - Model-Validierung und -Analyse
    """
    
    def __init__(self):
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_saver = ModelSaver()
        self.checkpoint_manager = CheckpointManager()
        self.gpu_utils = GPUUtils()
        
        # Model state
        self.is_loaded = False
        self.model_info = {}
    
    def create_model(self, config_override: Optional[Dict] = None) -> MemoryOptimizedLLM:
        """
        Erstellt ein neues LLM-Model.
        
        Args:
            config_override: Optionale Config-Überschreibungen
            
        Returns:
            Das erstellte Model
        """
        print("🧠 Erstelle neues Model...")
        
        # TODO: Config override implementation falls benötigt
        self.model = MemoryOptimizedLLM()
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.model_info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameter_size': f"{total_params / 1e9:.2f}B" if total_params >= 1e9 else f"{total_params / 1e6:.0f}M",
            'device': str(self.device),
            'architecture': 'MemoryOptimizedLLM',
            'config': {
                'vocab_size': model_config.vocab_size,
                'hidden_size': model_config.hidden_size,
                'num_layers': model_config.num_layers,
                'num_attention_heads': model_config.num_attention_heads,
                'num_key_value_heads': model_config.num_key_value_heads
            }
        }
        
        self.is_loaded = True
        
        print(f"✅ Model erstellt: {self.model_info['parameter_size']} Parameter")
        print(f"   Device: {self.device}")
        print(f"   Trainable: {trainable_params:,} Parameter")
        
        return self.model
    
    def load_model(self, model_path: str, device: Optional[str] = None) -> MemoryOptimizedLLM:
        """
        Lädt ein gespeichertes Model.
        
        Args:
            model_path: Pfad zum Model (Verzeichnis oder .pt Datei)
            device: Ziel-Device ('cuda', 'cpu', oder None für auto)
            
        Returns:
            Das geladene Model
        """
        print(f"📥 Lade Model von: {model_path}")
        
        # Determine device
        if device:
            self.device = device
        
        # Handle different path formats
        if os.path.isdir(model_path):
            # Model directory
            checkpoint_file = os.path.join(model_path, 'model.pt')
        else:
            # Direct .pt file
            checkpoint_file = model_path
        
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"Model file not found: {checkpoint_file}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
        
        # Create model
        self.model = MemoryOptimizedLLM()
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint  # Direct state dict
        
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        
        # Extract model info from checkpoint
        if 'training_info' in checkpoint:
            training_info = checkpoint['training_info']
            self.model_info = {
                'total_parameters': training_info.get('total_parameters', 0),
                'parameter_size': f"{training_info.get('total_parameters', 0) / 1e9:.2f}B" if training_info.get('total_parameters', 0) >= 1e9 else f"{training_info.get('total_parameters', 0) / 1e6:.0f}M",
                'device': str(self.device),
                'loaded_from': model_path,
                'training_steps': training_info.get('final_step', 0),
                'final_loss': training_info.get('final_loss', 0.0)
            }
        else:
            # Fallback info
            total_params = sum(p.numel() for p in self.model.parameters())
            self.model_info = {
                'total_parameters': total_params,
                'parameter_size': f"{total_params / 1e9:.2f}B" if total_params >= 1e9 else f"{total_params / 1e6:.0f}M",
                'device': str(self.device),
                'loaded_from': model_path
            }
        
        self.is_loaded = True
        
        print(f"✅ Model geladen: {self.model_info['parameter_size']} Parameter")
        print(f"   Device: {self.device}")
        
        return self.model
    
    def save_model(self, save_path: str, step: int = 0, final_loss: float = 0.0, training_time: float = 0.0) -> str:
        """
        Speichert das aktuelle Model.
        
        Args:
            save_path: Speicherpfad
            step: Training steps
            final_loss: Finaler Loss
            training_time: Trainingszeit
            
        Returns:
            Pfad zum gespeicherten Model
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Kein Model geladen")
        
        return self.model_saver.save_model(
            model=self.model,
            step=step,
            final_loss=final_loss,
            training_time=training_time,
            save_dir=save_path
        )
    
    def validate_model(self, model_path: Optional[str] = None) -> bool:
        """
        Validiert ein Model.
        
        Args:
            model_path: Pfad zum Model (None = aktuelles Model)
            
        Returns:
            True wenn valid, False sonst
        """
        if model_path:
            # Validate external model
            from ..checkpoints.model_saver import validate_saved_model
            return validate_saved_model(model_path)
        else:
            # Validate current model
            if not self.is_loaded or self.model is None:
                print("❌ Kein Model geladen")
                return False
            
            try:
                # Test forward pass
                test_input = torch.randint(0, model_config.vocab_size, (1, 10), device=self.device)
                
                with torch.no_grad():
                    outputs = self.model(test_input)
                    logits = outputs['logits']
                
                expected_shape = (1, 10, model_config.vocab_size)
                if logits.shape != expected_shape:
                    print(f"❌ Falsche Output-Shape: {logits.shape}, erwartet: {expected_shape}")
                    return False
                
                print("✅ Model-Validierung erfolgreich")
                return True
                
            except Exception as e:
                print(f"❌ Model-Validierung fehlgeschlagen: {e}")
                return False
    
    def generate_text(self, input_text: str, max_length: int = 50, temperature: float = 0.8) -> str:
        """
        Generiert Text mit dem Model (einfache Implementierung).
        
        Args:
            input_text: Input-Text
            max_length: Maximale Länge
            temperature: Sampling-Temperatur
            
        Returns:
            Generierter Text
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Kein Model geladen")
        
        # Simplified text generation (requires tokenizer integration)
        print("⚠️ Text-Generation benötigt Tokenizer-Integration")
        print(f"Input: {input_text}")
        print("Für vollständige Text-Generation siehe README.md im gespeicherten Model")
        
        return input_text + " [Generated text would appear here]"
    
    def get_model_info(self) -> Dict:
        """Gibt Model-Informationen zurück."""
        return self.model_info.copy()
    
    def get_model_summary(self) -> str:
        """Gibt Model-Zusammenfassung als String zurück."""
        if not self.is_loaded:
            return "Kein Model geladen"
        
        info = self.model_info
        summary = f"""
🧠 MODEL SUMMARY
================
Architecture: {info.get('architecture', 'Unknown')}
Parameters: {info.get('parameter_size', 'Unknown')}
Total Parameters: {info.get('total_parameters', 0):,}
Device: {info.get('device', 'Unknown')}
"""
        
        if 'training_steps' in info:
            summary += f"Training Steps: {info['training_steps']:,}\n"
        if 'final_loss' in info:
            summary += f"Final Loss: {info['final_loss']:.4f}\n"
        if 'loaded_from' in info:
            summary += f"Loaded From: {info['loaded_from']}\n"
        
        return summary
    
    def set_eval_mode(self):
        """Setzt Model in Evaluation-Modus."""
        if self.model:
            self.model.eval()
            print("✅ Model in Evaluation-Modus")
    
    def set_train_mode(self):
        """Setzt Model in Training-Modus."""
        if self.model:
            self.model.train()
            print("✅ Model in Training-Modus")
    
    def move_to_device(self, device: str):
        """Verschiebt Model zu anderem Device."""
        if self.model:
            self.device = device
            self.model = self.model.to(device)
            self.model_info['device'] = device
            print(f"✅ Model zu {device} verschoben")
    
    def cleanup(self):
        """Bereinigt Model-Ressourcen."""
        if self.model:
            del self.model
            self.model = None
        
        self.is_loaded = False
        self.model_info.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("🧹 Model-Interface bereinigt")


# Convenience Functions
def create_model() -> MemoryOptimizedLLM:
    """Convenience function für Model-Erstellung."""
    interface = ModelInterface()
    return interface.create_model()


def load_model(model_path: str) -> MemoryOptimizedLLM:
    """Convenience function für Model-Laden."""
    interface = ModelInterface()
    return interface.load_model(model_path)
