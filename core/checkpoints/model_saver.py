"""
Model Saver Module

Contains functions for saving and validating trained models.
Handles final model export with comprehensive metadata and documentation.
"""

import os
import json
import torch
from datetime import datetime
from typing import Optional

from config import model_config, training_config


class ModelSaver:
    """Klasse für das Speichern trainierter Modelle."""
    
    def __init__(self, base_dir: str = None):
        self.base_dir = base_dir or os.path.expanduser("~/AI/llm-coding/trained_models")
        os.makedirs(self.base_dir, exist_ok=True)
    
    def save_model(self, model, step: int, final_loss: float, training_time: float, save_dir: str = None) -> str:
        """Speichert das trainierte Modell mit allen Metadaten."""
        
        # Erstelle Ausgabe-Verzeichnis mit Timestamp
        if save_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            total_params = sum(p.numel() for p in model.parameters())
            param_size = f"{total_params / 1e9:.1f}B" if total_params >= 1e9 else f"{total_params / 1e6:.0f}M"
            model_name = f"modern_llm_{param_size}_{timestamp}"
            save_dir = os.path.join(self.base_dir, model_name)

        os.makedirs(save_dir, exist_ok=True)

        print(f"\n💾 Speichere trainiertes Modell...")
        print(f"   Pfad: {save_dir}")

        # 1. PyTorch Model State Dict speichern
        state_dict = self._clean_state_dict(model.state_dict())
        
        model_checkpoint = {
            'model_state_dict': state_dict,
            'model_config': self._get_model_config_dict(),
            'training_info': self._get_training_info_dict(step, final_loss, training_time, model),
            'training_config': self._get_training_config_dict()
        }

        model_path = os.path.join(save_dir, 'model.pt')
        torch.save(model_checkpoint, model_path)
        print(f"   ✅ PyTorch Model: model.pt")

        # 2. Model Config als JSON
        config_path = os.path.join(save_dir, 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(model_checkpoint['model_config'], f, indent=2)
        print(f"   ✅ Model Config: config.json")

        # 3. Training Info als JSON
        training_info_path = os.path.join(save_dir, 'training_info.json')
        with open(training_info_path, 'w', encoding='utf-8') as f:
            json.dump(model_checkpoint['training_info'], f, indent=2)
        print(f"   ✅ Training Info: training_info.json")

        # 4. README mit Nutzungsanleitung
        self._create_readme(save_dir, model_checkpoint)
        print(f"   ✅ Documentation: README.md")

        total_params = model_checkpoint['training_info']['total_parameters']
        param_size = f"{total_params / 1e9:.2f}B" if total_params >= 1e9 else f"{total_params / 1e6:.0f}M"
        training_time_formatted = model_checkpoint['training_info']['training_time_formatted']

        print(f"\n🎉 Model erfolgreich gespeichert!")
        print(f"   Verzeichnis: {save_dir}")
        print(f"   Parameter: {param_size}")
        print(f"   Trainingszeit: {training_time_formatted}")

        return save_dir
    
    def _clean_state_dict(self, state_dict):
        """Bereinigt State Dict von torch.compile Prefixes."""
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            # torch.compile fügt _orig_mod. Prefix hinzu - entferne es
            clean_state_dict = {}
            for key, value in state_dict.items():
                clean_key = key.replace('_orig_mod.', '') if key.startswith('_orig_mod.') else key
                clean_state_dict[clean_key] = value
            return clean_state_dict
        return state_dict
    
    def _get_model_config_dict(self):
        """Erstellt Model Config Dictionary."""
        return {
            'vocab_size': model_config.vocab_size,
            'hidden_size': model_config.hidden_size,
            'num_layers': model_config.num_layers,
            'num_attention_heads': model_config.num_attention_heads,
            'num_key_value_heads': model_config.num_key_value_heads,
            'intermediate_size': model_config.intermediate_size,
            'max_position_embeddings': model_config.max_position_embeddings,
            'tie_word_embeddings': model_config.tie_word_embeddings,
        }
    
    def _get_training_info_dict(self, step, final_loss, training_time, model):
        """Erstellt Training Info Dictionary."""
        return {
            'final_step': step,
            'final_loss': final_loss,
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'training_time_seconds': training_time,
            'training_time_formatted': f"{training_time//3600:.0f}h {(training_time%3600)//60:.0f}m {training_time%60:.0f}s",
            'timestamp': datetime.now().isoformat(),
            'pytorch_version': torch.__version__,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        }
    
    def _get_training_config_dict(self, cache_info=None):
        """Erstellt Training Config Dictionary mit Cache-Info."""
        config_dict = {
            'max_steps': training_config.max_steps,
            'batch_size': training_config.batch_size,
            'gradient_accumulation_steps': training_config.gradient_accumulation_steps,
            'sequence_length': training_config.sequence_length,
            'learning_rate': training_config.learning_rate,
            'optimizer_type': training_config.optimizer_type,
            'use_mixed_precision': training_config.use_mixed_precision,
            'use_torch_compile': training_config.use_torch_compile,
        }

        # FIXED: Cache-Info hinzufügen für Checkpoint-Kompatibilität
        if cache_info:
            config_dict.update({
                'cache_dataset_name': cache_info.get('dataset_name', 'Unknown'),
                'cache_sequence_length': cache_info.get('sequence_length', 512),
                'cache_path': cache_info.get('path', ''),
                'cache_total_sequences': cache_info.get('total_sequences', 0)
            })
        else:
            # Fallback-Werte für bestehende Checkpoints
            config_dict.update({
                'cache_dataset_name': 'FineWeb',
                'cache_sequence_length': 512,
                'cache_path': 'cache/packed_sequences/512/FineWeb',
                'cache_total_sequences': 352217
            })

        return config_dict
    
    def _create_readme(self, save_dir, model_checkpoint):
        """Erstellt README-Datei mit Nutzungsanleitung."""
        readme_path = os.path.join(save_dir, 'README.md')
        total_params = model_checkpoint['training_info']['total_parameters']
        param_size = f"{total_params / 1e9:.2f}B" if total_params >= 1e9 else f"{total_params / 1e6:.0f}M"
        
        step = model_checkpoint['training_info']['final_step']
        final_loss = model_checkpoint['training_info']['final_loss']
        training_time_formatted = model_checkpoint['training_info']['training_time_formatted']

        readme_content = f"""# {os.path.basename(save_dir)}

## Model Information
- **Parameters:** {param_size} ({total_params:,} total)
- **Architecture:** Modern LLM with GQA, RoPE, SwiGLU
- **Training Steps:** {step:,}
- **Final Loss:** {final_loss:.4f}
- **Training Time:** {training_time_formatted}
- **Training Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Architecture Details
- **Hidden Size:** {model_config.hidden_size}
- **Layers:** {model_config.num_layers}
- **Attention Heads:** {model_config.num_attention_heads}
- **Key-Value Heads:** {model_config.num_key_value_heads} (GQA)
- **Vocabulary Size:** {model_config.vocab_size:,}
- **Max Sequence Length:** {model_config.max_position_embeddings}

## Usage

### Load Model
```python
import torch
from core.models import MemoryOptimizedLLM

# Load checkpoint
checkpoint = torch.load('model.pt', map_location='cpu')

# Create model with same config
model = MemoryOptimizedLLM()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

### Generate Text
```python
# Example text generation (requires tokenizer)
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Encode input
input_text = "The future of AI is"
input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

# Generate
with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs['logits']

    # Simple greedy decoding
    next_token_id = torch.argmax(logits[0, -1, :])
    next_token = tokenizer.decode([next_token_id])
    print(f"{{input_text}}{{next_token}}")
```

## Files
- `model.pt` - PyTorch model checkpoint
- `config.json` - Model architecture configuration
- `training_info.json` - Training metadata and statistics
- `README.md` - This documentation

## Training Configuration
- **Batch Size:** {training_config.batch_size}
- **Gradient Accumulation:** {training_config.gradient_accumulation_steps}
- **Learning Rate:** {training_config.learning_rate}
- **Sequence Length:** {training_config.sequence_length}
- **Mixed Precision:** {training_config.use_mixed_precision}
- **Torch Compile:** {training_config.use_torch_compile}
"""

        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)


def save_trained_model(model, step: int, final_loss: float, training_time: float, save_dir: str = None) -> str:
    """Convenience function für Model Saving."""
    saver = ModelSaver()
    return saver.save_model(model, step, final_loss, training_time, save_dir)


def validate_saved_model(model_path: str) -> bool:
    """Validiert ein gespeichertes Modell durch Laden und Testen."""
    try:
        print(f"\n🔍 Validiere gespeichertes Modell...")

        # 1. Lade Checkpoint
        checkpoint_file = os.path.join(model_path, 'model.pt')
        if not os.path.exists(checkpoint_file):
            print(f"   ❌ Model-Datei nicht gefunden: {checkpoint_file}")
            return False

        checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
        print(f"   ✅ Checkpoint geladen")

        # 2. Erstelle Modell mit gleicher Konfiguration
        from ..models import MemoryOptimizedLLM
        model = MemoryOptimizedLLM()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"   ✅ Model-State geladen")

        # 3. Teste Forward Pass
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # Erstelle Test-Input
        test_input = torch.randint(0, model_config.vocab_size, (1, 10), device=device)

        with torch.no_grad():
            outputs = model(test_input)
            logits = outputs['logits']

        # 4. Prüfe Output-Shape
        expected_shape = (1, 10, model_config.vocab_size)
        if logits.shape != expected_shape:
            print(f"   ❌ Falsche Output-Shape: {logits.shape}, erwartet: {expected_shape}")
            return False

        print(f"   ✅ Forward Pass erfolgreich")
        print(f"   ✅ Output-Shape korrekt: {logits.shape}")

        # 5. Prüfe Metadaten
        training_info = checkpoint.get('training_info', {})
        total_params = training_info.get('total_parameters', 0)
        actual_params = sum(p.numel() for p in model.parameters())

        if total_params != actual_params:
            print(f"   ⚠️  Parameter-Anzahl stimmt nicht überein: {total_params} vs {actual_params}")
        else:
            print(f"   ✅ Parameter-Anzahl korrekt: {actual_params:,}")

        print(f"\n✅ Model-Validierung erfolgreich!")
        return True

    except Exception as e:
        print(f"\n❌ Model-Validierung fehlgeschlagen: {e}")
        return False
