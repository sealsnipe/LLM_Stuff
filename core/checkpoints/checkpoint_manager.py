"""
Checkpoint Manager Module

Contains the CheckpointManager class for handling training checkpoints.
Manages checkpoint saving, loading, scanning, and cleanup operations.
"""

import os
import torch
from datetime import datetime
from typing import List, Dict, Optional

from config import model_config, training_config


class CheckpointManager:
    """Manager für Training Checkpoints."""
    
    def __init__(self, checkpoint_dir: str = None):
        self.checkpoint_dir = checkpoint_dir or os.path.expanduser("~/AI/llm-coding/current_training/checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def scan_checkpoints(self) -> List[Dict]:
        """Scannt Checkpoint-Ordner und parsed verfügbare Checkpoints."""
        if not os.path.exists(self.checkpoint_dir):
            return []

        checkpoints = []
        for filename in os.listdir(self.checkpoint_dir):
            if filename.endswith('.pt') and '_checkpoint_' in filename:
                try:
                    # Parse: {model_name}_checkpoint_{step}_run_{run_id}.pt
                    parts = filename.replace('.pt', '').split('_')

                    # Finde checkpoint, step, run indices
                    checkpoint_idx = parts.index('checkpoint')
                    run_idx = parts.index('run')

                    model_name = '_'.join(parts[:checkpoint_idx])
                    step = int(parts[checkpoint_idx + 1])
                    run_id = int(parts[run_idx + 1])

                    filepath = os.path.join(self.checkpoint_dir, filename)

                    # Lade Metadaten
                    try:
                        checkpoint_data = torch.load(filepath, map_location='cpu', weights_only=False)
                        loss = checkpoint_data.get('loss', 0.0)
                        timestamp = checkpoint_data.get('timestamp', 'Unknown')

                        # Parse timestamp für Anzeige
                        if timestamp != 'Unknown':
                            try:
                                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                                display_time = dt.strftime('%Y-%m-%d %H:%M')
                            except:
                                display_time = timestamp[:16]
                        else:
                            display_time = 'Unknown'

                        checkpoints.append({
                            'filename': filename,
                            'filepath': filepath,
                            'model_name': model_name,
                            'step': step,
                            'run_id': run_id,
                            'loss': loss,
                            'timestamp': display_time
                        })
                    except Exception as e:
                        print(f"⚠️  Konnte Checkpoint {filename} nicht laden: {e}")
                        continue

                except (ValueError, IndexError):
                    # Ungültiger Dateiname, ignorieren
                    continue

        # Sortiere nach Timestamp (neueste zuerst)
        checkpoints.sort(key=lambda x: x['timestamp'], reverse=True)
        return checkpoints
    
    def save_checkpoint(self, model, optimizer, step: int, loss: float, model_name: str, run_id: int, cache_info=None, scheduler=None):
        """Speichert Checkpoint und löscht alte vom gleichen Run mit Cache-Info und Scheduler."""
        
        # Erstelle Checkpoint mit Cache-Info und Scheduler
        checkpoint = {
            'model_name': model_name,
            'run_id': run_id,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'model_config': model_config.__dict__,
            'training_config': training_config.__dict__,
            'timestamp': datetime.now().isoformat(),
            'pytorch_version': torch.__version__,
        }

        # FIXED: Save scheduler state if provided
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        # FIXED: Cache-Info hinzufügen
        if cache_info:
            checkpoint['cache_info'] = {
                'dataset_name': cache_info.get('dataset_name', 'Unknown'),
                'sequence_length': cache_info.get('sequence_length', 512),
                'path': cache_info.get('path', ''),
                'total_sequences': cache_info.get('total_sequences', 0)
            }
        else:
            # Fallback für bestehende Checkpoints
            checkpoint['cache_info'] = {
                'dataset_name': 'FineWeb',
                'sequence_length': 512,
                'path': 'cache/packed_sequences/512/FineWeb',
                'total_sequences': 352217
            }

        # Speichere neuen Checkpoint
        filename = f"{model_name}_checkpoint_{step}_run_{run_id}.pt"
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)

        # Cleanup: Lösche alte Checkpoints vom gleichen Run
        self._cleanup_old_checkpoints_same_run(model_name, run_id, step)

        return filepath
    
    def load_checkpoint(self, checkpoint_info: Dict):
        """Lädt Checkpoint für Training-Fortsetzung."""
        print(f"\n📥 Lade Checkpoint: {checkpoint_info['filename']}")

        try:
            checkpoint = torch.load(checkpoint_info['filepath'], map_location='cpu', weights_only=False)

            # Erstelle Modell
            from ..models import MemoryOptimizedLLM
            model = MemoryOptimizedLLM()
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(torch.device("cuda"))

            # Erstelle Optimizer
            from ..training.optimizers import create_optimizer
            optimizer = create_optimizer(model)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            start_step = checkpoint['step']
            model_name = checkpoint_info['model_name']
            run_id = checkpoint_info['run_id']

            print(f"✅ Modell geladen: {model_name}")
            print(f"✅ Training fortsetzung ab Step: {start_step:,}")
            print(f"✅ Run ID: {run_id}")
            print(f"✅ Letzter Loss: {checkpoint_info['loss']:.4f}")

            return model, optimizer, start_step, model_name, run_id

        except Exception as e:
            print(f"❌ Fehler beim Laden des Checkpoints: {e}")
            raise
    
    def _cleanup_old_checkpoints_same_run(self, model_name: str, run_id: int, current_step: int):
        """Löscht alte Checkpoints vom gleichen Run."""
        if not os.path.exists(self.checkpoint_dir):
            return

        for filename in os.listdir(self.checkpoint_dir):
            if filename.endswith('.pt') and '_checkpoint_' in filename:
                try:
                    # Parse Dateiname
                    parts = filename.replace('.pt', '').split('_')
                    checkpoint_idx = parts.index('checkpoint')
                    run_idx = parts.index('run')

                    file_model_name = '_'.join(parts[:checkpoint_idx])
                    file_step = int(parts[checkpoint_idx + 1])
                    file_run_id = int(parts[run_idx + 1])

                    # Lösche wenn: gleicher Modell-Name, gleiche Run-ID, aber älterer Step
                    if (file_model_name == model_name and
                        file_run_id == run_id and
                        file_step < current_step):

                        filepath = os.path.join(self.checkpoint_dir, filename)
                        try:
                            os.remove(filepath)
                            # Stille Löschung (kein Print um Progress nicht zu stören)
                        except OSError:
                            pass

                except (ValueError, IndexError):
                    continue
    
    def cleanup_old_checkpoints(self, keep_last: int = 3):
        """Entfernt alte Checkpoints, behält nur die neuesten."""
        if not os.path.exists(self.checkpoint_dir):
            return

        # Finde alle Checkpoint-Dateien
        checkpoint_files = []
        for filename in os.listdir(self.checkpoint_dir):
            if filename.startswith('checkpoint_step_') and filename.endswith('.pt'):
                try:
                    step = int(filename.replace('checkpoint_step_', '').replace('.pt', ''))
                    filepath = os.path.join(self.checkpoint_dir, filename)
                    checkpoint_files.append((step, filepath))
                except ValueError:
                    continue

        # Sortiere nach Step-Nummer
        checkpoint_files.sort(key=lambda x: x[0])

        # Entferne alte Checkpoints
        if len(checkpoint_files) > keep_last:
            for step, filepath in checkpoint_files[:-keep_last]:
                try:
                    os.remove(filepath)
                except OSError:
                    pass
    
    def get_checkpoint_stats(self) -> Dict:
        """Gibt Checkpoint-Statistiken zurück."""
        checkpoints = self.scan_checkpoints()
        
        if not checkpoints:
            return {'total_checkpoints': 0, 'models': {}}
        
        # Gruppiere nach Modell
        models = {}
        for cp in checkpoints:
            model_name = cp['model_name']
            if model_name not in models:
                models[model_name] = {'runs': {}, 'total_checkpoints': 0}
            
            run_id = cp['run_id']
            if run_id not in models[model_name]['runs']:
                models[model_name]['runs'][run_id] = []
            
            models[model_name]['runs'][run_id].append(cp)
            models[model_name]['total_checkpoints'] += 1
        
        return {
            'total_checkpoints': len(checkpoints),
            'models': models,
            'checkpoint_dir': self.checkpoint_dir
        }
