"""
Training State Management Module

Contains functions for managing training state, run IDs, and training mode selection.
Handles the user interface for checkpoint selection and training continuation.
"""

import os
from typing import Dict, List, Optional
from .checkpoint_manager import CheckpointManager


class TrainingState:
    """Manages training state and user interactions."""
    
    def __init__(self):
        self.checkpoint_manager = CheckpointManager()
    
    def get_model_name_from_user(self) -> str:
        """Fragt User nach Modell-Name für neues Training mit intelligenten Vorschlägen."""
        print("\n🏷️  MODELL-NAME FÜR NEUES TRAINING")
        print("=" * 50)

        # Generate intelligent suggestions
        suggestions = self._generate_model_name_suggestions()

        print("📋 VORSCHLÄGE (Zahlen wählen oder eigenen Namen eingeben):")
        print("-" * 50)
        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i}. {suggestion['name']} ({suggestion['description']})")

        # Scanne existierende Checkpoint-Namen
        existing_models = set()
        checkpoints = self.checkpoint_manager.scan_checkpoints()
        for cp in checkpoints:
            existing_models.add(cp['model_name'])

        if existing_models:
            print(f"\n⚠️  Existierende Modelle: {', '.join(sorted(existing_models))}")
            print("Wähle einen ANDEREN Namen um Konflikte zu vermeiden!")

        while True:
            user_input = input(f"\nWähle 1-{len(suggestions)} oder gib eigenen Namen ein: ").strip()

            # Check if user selected a number (suggestion)
            if user_input.isdigit():
                choice = int(user_input)
                if 1 <= choice <= len(suggestions):
                    model_name = suggestions[choice - 1]['name']
                    print(f"✅ Vorschlag gewählt: {model_name}")
                else:
                    print(f"❌ Ungültige Auswahl. Wähle 1-{len(suggestions)} oder gib einen Namen ein.")
                    continue
            else:
                # User entered custom name
                model_name = user_input

            # Basis-Validierung
            if not model_name or not model_name.replace('_', '').replace('-', '').isalnum():
                print("❌ Ungültiger Name. Nur Buchstaben, Zahlen, _ und - erlaubt.")
                continue

            # Kollisions-Prüfung
            if model_name in existing_models:
                print(f"❌ Modell '{model_name}' existiert bereits! Wähle einen anderen Namen.")
                continue

            print(f"✅ Name '{model_name}' ist verfügbar.")
            return model_name

    def _generate_model_name_suggestions(self) -> list:
        """Generiert intelligente Modell-Namen-Vorschläge basierend auf Config."""
        from config import model_config, training_config
        from ..utils.cache_registry import load_cache_registry

        suggestions = []

        # Calculate model parameters with tied embeddings
        def calculate_tied_parameters():
            h = model_config.hidden_size
            n = model_config.num_layers
            v = model_config.vocab_size
            i = model_config.intermediate_size

            # Embedding parameters (tied: only one embedding matrix)
            if model_config.tie_word_embeddings:
                embedding_params = v * h  # Only input embeddings
            else:
                embedding_params = 2 * v * h  # Input + output embeddings

            # Transformer layers
            attention_params = 4 * h * h  # q, k, v, o projections
            ff_params = 3 * h * i  # gate, up, down projections
            norm_params = 2 * h  # attention_norm + ffn_norm
            layer_params = attention_params + ff_params + norm_params

            # Total parameters
            total_params = embedding_params + (n * layer_params) + h  # +h for final norm
            return total_params

        # Get parameter count and format
        total_params = calculate_tied_parameters()
        if total_params >= 1e9:
            param_str = f"{total_params / 1e9:.1f}B"
        else:
            param_str = f"{total_params / 1e6:.0f}M"

        # Get dataset info from cache registry
        caches = load_cache_registry()
        dataset_info = "FineWeb"  # Default
        seq_len = training_config.sequence_length

        if caches:
            # Find cache matching sequence length
            for cache in caches:
                if cache['sequence_length'] == seq_len:
                    dataset_info = cache['dataset_name']
                    break

        # Generate suggestions
        base_name = f"{param_str}_{dataset_info}_{seq_len}"

        suggestions.extend([
            {
                'name': base_name,
                'description': f'{param_str} parameters, {dataset_info} dataset, {seq_len} seq_len'
            },
            {
                'name': f"experiment_{base_name}",
                'description': f'Experimental run with {param_str} parameters'
            },
            {
                'name': f"modern_llm_{param_str}",
                'description': f'Modern LLM architecture with {param_str} parameters'
            },
            {
                'name': f"test_{param_str}",
                'description': f'Test run with {param_str} parameters'
            }
        ])

        return suggestions
    
    def display_checkpoint_menu(self, checkpoints: List[Dict]) -> Optional[Dict]:
        """Zeigt verfügbare Checkpoints mit Details."""
        print("\n📋 VERFÜGBARE CHECKPOINTS")
        print("=" * 80)

        if not checkpoints:
            print("Keine Checkpoints gefunden.")
            return None

        for i, cp in enumerate(checkpoints, 1):
            print(f"{i:2d}. {cp['model_name']}_checkpoint_{cp['step']}_run_{cp['run_id']}")
            print(f"     Step: {cp['step']:,} | Loss: {cp['loss']:.4f} | Zeit: {cp['timestamp']}")
            print()

        print("N.  Neues Training starten")
        print("X.  Checkpoint-Management (Löschen)")
        print("=" * 80)

        while True:
            choice = input("Auswahl [1-{}, N oder X]: ".format(len(checkpoints))).strip().upper()

            if choice == 'N':
                return None
            elif choice == 'X':
                self.checkpoint_management_menu()
                # Nach Management zurück zum Hauptmenu
                return self.display_checkpoint_menu(self.checkpoint_manager.scan_checkpoints())

            try:
                idx = int(choice) - 1
                if 0 <= idx < len(checkpoints):
                    return checkpoints[idx]
                else:
                    print(f"❌ Ungültige Auswahl. Bitte 1-{len(checkpoints)}, N oder X eingeben.")
            except ValueError:
                print(f"❌ Ungültige Eingabe. Bitte 1-{len(checkpoints)}, N oder X eingeben.")

    def checkpoint_management_menu(self):
        """Checkpoint-Management Menu für Löschen von Checkpoints und Modellen."""
        print("\n🗑️  CHECKPOINT-MANAGEMENT")
        print("=" * 80)

        # Scanne Checkpoints und Modelle
        checkpoints = self.checkpoint_manager.scan_checkpoints()
        trained_models = self._scan_trained_models()

        if not checkpoints and not trained_models:
            print("Keine Checkpoints oder Modelle gefunden.")
            input("Drücke Enter um fortzufahren...")
            return

        print("📋 VERFÜGBARE ITEMS ZUM LÖSCHEN:")
        print("-" * 80)

        items = []

        # Checkpoints hinzufügen
        if checkpoints:
            print("🔄 CHECKPOINTS:")
            for cp in checkpoints:
                items.append({
                    'type': 'checkpoint',
                    'data': cp,
                    'display': f"Checkpoint: {cp['model_name']}_checkpoint_{cp['step']}_run_{cp['run_id']} (Step: {cp['step']:,})"
                })
                print(f"{len(items):2d}. {items[-1]['display']}")
            print()

        # Trained Models hinzufügen
        if trained_models:
            print("🎯 FERTIGE MODELLE:")
            for model in trained_models:
                items.append({
                    'type': 'model',
                    'data': model,
                    'display': f"Modell: {model['name']} ({model['size']:.1f}MB)"
                })
                print(f"{len(items):2d}. {items[-1]['display']}")
            print()

        print("A.  Alle Checkpoints löschen")
        print("B.  Alle Modelle löschen")
        print("L.  Alle Logs löschen")
        print("C.  Alles löschen (Checkpoints + Modelle + Logs)")
        print("R.  Zurück zum Hauptmenu")
        print("=" * 80)

        while True:
            choice = input(f"Auswahl [1-{len(items)}, A, B, L, C oder R]: ").strip().upper()

            if choice == 'R':
                return
            elif choice == 'A':
                self._delete_all_checkpoints()
                return
            elif choice == 'B':
                self._delete_all_models()
                return
            elif choice == 'L':
                self._delete_all_logs()
                return
            elif choice == 'C':
                self._delete_everything()
                return

            try:
                idx = int(choice) - 1
                if 0 <= idx < len(items):
                    self._delete_single_item(items[idx])
                    return
                else:
                    print(f"❌ Ungültige Auswahl. Bitte 1-{len(items)}, A, B, C oder R eingeben.")
            except ValueError:
                print(f"❌ Ungültige Eingabe. Bitte 1-{len(items)}, A, B, C oder R eingeben.")

    def _scan_trained_models(self):
        """Scannt verfügbare trainierte Modelle."""
        import os
        models = []
        trained_models_dir = "trained_models"

        if not os.path.exists(trained_models_dir):
            return models

        for item in os.listdir(trained_models_dir):
            model_path = os.path.join(trained_models_dir, item)
            if os.path.isdir(model_path):
                # Berechne Ordnergröße
                total_size = 0
                for dirpath, dirnames, filenames in os.walk(model_path):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        if os.path.exists(filepath):
                            total_size += os.path.getsize(filepath)

                models.append({
                    'name': item,
                    'path': model_path,
                    'size': total_size / (1024 * 1024)  # MB
                })

        return sorted(models, key=lambda x: x['name'])

    def _delete_single_item(self, item):
        """Löscht ein einzelnes Item (Checkpoint oder Modell) mit zugehörigen Logs."""
        import os
        import shutil

        # Zeige was gelöscht wird (inklusive Logs bei Checkpoints)
        print(f"\n⚠️  LÖSCHUNG BESTÄTIGEN:")
        print(f"Item: {item['display']}")

        if item['type'] == 'checkpoint':
            # Finde zugehörige Log-Dateien
            related_logs = self._find_related_logs(item['data'])
            if related_logs:
                print(f"\n📋 ZUGEHÖRIGE LOGS (werden mit gelöscht):")
                for log_file in related_logs:
                    log_size = os.path.getsize(log_file) / 1024 if os.path.exists(log_file) else 0
                    print(f"   • {os.path.basename(log_file)} ({log_size:.1f}KB)")

        confirm = input("\nWirklich löschen? [y/N]: ").strip().lower()

        if confirm != 'y':
            print("❌ Löschung abgebrochen.")
            input("Drücke Enter um fortzufahren...")
            return

        try:
            if item['type'] == 'checkpoint':
                # Lösche Checkpoint-Datei
                filepath = item['data']['filepath']
                if os.path.exists(filepath):
                    os.remove(filepath)
                    print(f"✅ Checkpoint gelöscht: {os.path.basename(filepath)}")
                else:
                    print(f"⚠️  Checkpoint-Datei nicht gefunden: {filepath}")

                # Lösche zugehörige Logs
                related_logs = self._find_related_logs(item['data'])
                deleted_logs = 0
                for log_file in related_logs:
                    try:
                        if os.path.exists(log_file):
                            os.remove(log_file)
                            deleted_logs += 1
                            print(f"✅ Log gelöscht: {os.path.basename(log_file)}")
                    except Exception as e:
                        print(f"⚠️  Fehler beim Löschen von {os.path.basename(log_file)}: {e}")

                if related_logs:
                    print(f"📊 {deleted_logs}/{len(related_logs)} Log-Dateien gelöscht.")

            elif item['type'] == 'model':
                # Lösche Modell-Ordner
                model_path = item['data']['path']
                if os.path.exists(model_path):
                    shutil.rmtree(model_path)
                    print(f"✅ Modell gelöscht: {item['data']['name']}")
                else:
                    print(f"⚠️  Modell-Ordner nicht gefunden: {model_path}")

        except Exception as e:
            print(f"❌ Fehler beim Löschen: {e}")

        input("Drücke Enter um fortzufahren...")

    def _find_related_logs(self, checkpoint_data):
        """Findet alle Log-Dateien die zu einem Checkpoint gehören."""
        import os
        import glob

        model_name = checkpoint_data['model_name']
        run_id = checkpoint_data['run_id']

        # Mögliche Log-Datei-Patterns
        log_patterns = [
            f"training_logs/{model_name}_run_{run_id}.json",
            f"training_logs/{model_name}_run_{run_id}_summary.json",
            f"training_logs/{model_name}_run_{run_id}_training_plots.png",
            f"training_logs/{model_name}_training_plots.png",  # Ohne run_id
        ]

        related_logs = []
        for pattern in log_patterns:
            # Verwende glob für flexible Suche
            matches = glob.glob(pattern)
            related_logs.extend(matches)

        # Entferne Duplikate und sortiere
        related_logs = sorted(list(set(related_logs)))

        return related_logs

    def _delete_all_checkpoints(self):
        """Löscht alle Checkpoints mit zugehörigen Logs."""
        import os

        checkpoints = self.checkpoint_manager.scan_checkpoints()
        if not checkpoints:
            print("Keine Checkpoints zum Löschen gefunden.")
            input("Drücke Enter um fortzufahren...")
            return

        # Sammle alle zugehörigen Log-Dateien
        all_related_logs = set()
        for cp in checkpoints:
            related_logs = self._find_related_logs(cp)
            all_related_logs.update(related_logs)

        print(f"\n⚠️  ALLE CHECKPOINTS LÖSCHEN:")
        print(f"Checkpoints: {len(checkpoints)}")
        if all_related_logs:
            total_log_size = sum(os.path.getsize(log) for log in all_related_logs if os.path.exists(log)) / 1024
            print(f"Zugehörige Logs: {len(all_related_logs)} Dateien ({total_log_size:.1f}KB)")

        confirm = input("Wirklich ALLE Checkpoints und Logs löschen? [y/N]: ").strip().lower()

        if confirm != 'y':
            print("❌ Löschung abgebrochen.")
            input("Drücke Enter um fortzufahren...")
            return

        # Lösche Checkpoints
        deleted_checkpoints = 0
        for cp in checkpoints:
            try:
                if os.path.exists(cp['filepath']):
                    os.remove(cp['filepath'])
                    deleted_checkpoints += 1
            except Exception as e:
                print(f"❌ Fehler beim Löschen von {cp['filename']}: {e}")

        # Lösche zugehörige Logs
        deleted_logs = 0
        for log_file in all_related_logs:
            try:
                if os.path.exists(log_file):
                    os.remove(log_file)
                    deleted_logs += 1
            except Exception as e:
                print(f"⚠️  Fehler beim Löschen von {os.path.basename(log_file)}: {e}")

        print(f"✅ {deleted_checkpoints}/{len(checkpoints)} Checkpoints gelöscht.")
        if all_related_logs:
            print(f"✅ {deleted_logs}/{len(all_related_logs)} Log-Dateien gelöscht.")

        input("Drücke Enter um fortzufahren...")

    def _delete_all_models(self):
        """Löscht alle trainierten Modelle."""
        import os
        import shutil

        models = self._scan_trained_models()
        if not models:
            print("Keine Modelle zum Löschen gefunden.")
            input("Drücke Enter um fortzufahren...")
            return

        total_size = sum(model['size'] for model in models)
        print(f"\n⚠️  ALLE MODELLE LÖSCHEN:")
        print(f"Anzahl: {len(models)} Modelle ({total_size:.1f}MB)")
        confirm = input("Wirklich ALLE Modelle löschen? [y/N]: ").strip().lower()

        if confirm != 'y':
            print("❌ Löschung abgebrochen.")
            input("Drücke Enter um fortzufahren...")
            return

        deleted_count = 0
        for model in models:
            try:
                if os.path.exists(model['path']):
                    shutil.rmtree(model['path'])
                    deleted_count += 1
            except Exception as e:
                print(f"❌ Fehler beim Löschen von {model['name']}: {e}")

        print(f"✅ {deleted_count}/{len(models)} Modelle gelöscht.")
        input("Drücke Enter um fortzufahren...")

    def _delete_all_logs(self):
        """Löscht alle Log-Dateien aus allen Log-Ordnern."""
        import os
        import shutil

        # Alle Log-Ordner definieren
        log_directories = [
            "training_logs",                    # Root-Level (aktuelle Session)
            "LLM_Stuff/training_logs",         # LLM_Stuff-Level (Archiv)
            "current_training/logs",           # Falls vorhanden
            "logs",                            # System-Config default
        ]

        # Sammle alle Log-Dateien
        all_log_files = []
        total_size = 0

        for log_dir in log_directories:
            if os.path.exists(log_dir):
                for item in os.listdir(log_dir):
                    item_path = os.path.join(log_dir, item)
                    if os.path.isfile(item_path):
                        # Nur Log-relevante Dateien
                        if any(item.endswith(ext) for ext in ['.json', '.log', '.png']):
                            file_size = os.path.getsize(item_path)
                            all_log_files.append({
                                'path': item_path,
                                'name': item,
                                'dir': log_dir,
                                'size': file_size
                            })
                            total_size += file_size

        if not all_log_files:
            print("Keine Log-Dateien zum Löschen gefunden.")
            input("Drücke Enter um fortzufahren...")
            return

        # Gruppiere nach Ordnern für bessere Übersicht
        dirs_with_files = {}
        for log_file in all_log_files:
            dir_name = log_file['dir']
            if dir_name not in dirs_with_files:
                dirs_with_files[dir_name] = []
            dirs_with_files[dir_name].append(log_file)

        print(f"\n⚠️  ALLE LOGS LÖSCHEN:")
        print(f"Gefundene Log-Ordner: {len(dirs_with_files)}")
        print(f"Gesamt Log-Dateien: {len(all_log_files)}")
        print(f"Gesamt Größe: {total_size / (1024*1024):.1f}MB")
        print()

        # Zeige Details pro Ordner
        for dir_name, files in dirs_with_files.items():
            dir_size = sum(f['size'] for f in files)
            print(f"📁 {dir_name}: {len(files)} Dateien ({dir_size / 1024:.1f}KB)")

        confirm = input(f"\nWirklich ALLE {len(all_log_files)} Log-Dateien löschen? [y/N]: ").strip().lower()

        if confirm != 'y':
            print("❌ Löschung abgebrochen.")
            input("Drücke Enter um fortzufahren...")
            return

        # Lösche alle Log-Dateien
        deleted_count = 0
        deleted_size = 0

        for log_file in all_log_files:
            try:
                if os.path.exists(log_file['path']):
                    os.remove(log_file['path'])
                    deleted_count += 1
                    deleted_size += log_file['size']
            except Exception as e:
                print(f"⚠️  Fehler beim Löschen von {log_file['name']}: {e}")

        # Lösche leere Log-Ordner (optional)
        empty_dirs_removed = 0
        for log_dir in log_directories:
            try:
                if os.path.exists(log_dir) and not os.listdir(log_dir):
                    os.rmdir(log_dir)
                    empty_dirs_removed += 1
                    print(f"📁 Leerer Ordner entfernt: {log_dir}")
            except Exception:
                pass  # Ordner nicht leer oder andere Fehler ignorieren

        print(f"✅ {deleted_count}/{len(all_log_files)} Log-Dateien gelöscht ({deleted_size / (1024*1024):.1f}MB).")
        if empty_dirs_removed > 0:
            print(f"📁 {empty_dirs_removed} leere Log-Ordner entfernt.")

        input("Drücke Enter um fortzufahren...")

    def _delete_everything(self):
        """Löscht ALLES: Checkpoints, Modelle, Logs und leert alle Ordner komplett."""
        import os
        import shutil

        checkpoints = self.checkpoint_manager.scan_checkpoints()
        models = self._scan_trained_models()

        # Sammle ALLE Log-Informationen (nicht nur checkpoint-related)
        log_directories = [
            "training_logs",
            "LLM_Stuff/training_logs",
            "current_training/logs",
            "logs",
        ]

        all_log_files = []
        total_log_size = 0

        for log_dir in log_directories:
            if os.path.exists(log_dir):
                for item in os.listdir(log_dir):
                    item_path = os.path.join(log_dir, item)
                    if os.path.isfile(item_path):
                        if any(item.endswith(ext) for ext in ['.json', '.log', '.png']):
                            file_size = os.path.getsize(item_path)
                            all_log_files.append(item_path)
                            total_log_size += file_size

        # Sammle alle Ordner die geleert werden
        directories_to_clear = [
            "current_training/checkpoints",
            "trained_models",
            "training_logs",
            "LLM_Stuff/training_logs",
            "current_training/logs",
            "logs",
        ]

        existing_dirs = [d for d in directories_to_clear if os.path.exists(d)]

        total_model_size = sum(model['size'] for model in models)

        print(f"\n⚠️  KOMPLETTE BEREINIGUNG:")
        print("=" * 50)
        print(f"📋 Checkpoints: {len(checkpoints)}")
        print(f"🎯 Modelle: {len(models)} ({total_model_size:.1f}MB)")
        print(f"📄 Log-Dateien: {len(all_log_files)} ({total_log_size / (1024*1024):.1f}MB)")
        print(f"📁 Ordner zu leeren: {len(existing_dirs)}")
        print()
        print("🗂️  BETROFFENE ORDNER:")
        for dir_name in existing_dirs:
            if os.path.exists(dir_name):
                item_count = len([f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))])
                print(f"   • {dir_name} ({item_count} Dateien)")

        print("\n⚠️  WARNUNG: Dies löscht ALLES und leert alle Training-Ordner!")
        confirm = input("Wirklich KOMPLETTE BEREINIGUNG durchführen? [y/N]: ").strip().lower()

        if confirm != 'y':
            print("❌ Bereinigung abgebrochen.")
            input("Drücke Enter um fortzufahren...")
            return

        print("\n🗑️  KOMPLETTE BEREINIGUNG GESTARTET:")
        print("=" * 50)

        # 1. Lösche alle Ordner komplett und erstelle sie neu
        for dir_name in existing_dirs:
            try:
                if os.path.exists(dir_name):
                    print(f"🗂️  Leere Ordner: {dir_name}")
                    shutil.rmtree(dir_name)
                    os.makedirs(dir_name, exist_ok=True)
                    print(f"✅ Ordner geleert und neu erstellt: {dir_name}")
            except Exception as e:
                print(f"⚠️  Fehler bei {dir_name}: {e}")

        # 2. Stelle sicher, dass wichtige Ordner existieren
        essential_dirs = [
            "current_training/checkpoints",
            "trained_models",
            "training_logs",
        ]

        for dir_name in essential_dirs:
            os.makedirs(dir_name, exist_ok=True)

        print("\n✅ KOMPLETTE BEREINIGUNG ABGESCHLOSSEN!")
        print("=" * 50)
        print("🎯 Alle Checkpoints gelöscht")
        print("🎯 Alle Modelle gelöscht")
        print("🎯 Alle Logs gelöscht")
        print("🎯 Alle Ordner geleert und neu erstellt")
        print("🎯 System bereit für neues Training")

        input("\nDrücke Enter um fortzufahren...")
    
    def handle_training_mode_selection(self) -> Dict:
        """Hauptfunktion für Training-Modus-Auswahl mit Log-Bereinigung."""
        print("\n🚀 TRAINING MODE SELECTION")
        print("=" * 50)

        # Scanne verfügbare Checkpoints
        checkpoints = self.checkpoint_manager.scan_checkpoints()

        if not checkpoints:
            print("Keine Checkpoints gefunden. Starte neues Training.")
            model_name = self.get_model_name_from_user()

            # Bereinige Logs für neues Modell (falls welche existieren)
            print(f"\n🧹 Bereinige verwaiste Logs für: {model_name}")
            self.cleanup_orphaned_logs(model_name)

            return {
                'mode': 'new',
                'model_name': model_name
            }

        # Zeige Checkpoint-Menu
        selected_checkpoint = self.display_checkpoint_menu(checkpoints)

        if selected_checkpoint is None:
            # Neues Training - Cache auswählen
            cache_info = self.select_cache_for_new_training()
            model_name = self.get_model_name_from_user()
            return {
                'mode': 'new',
                'model_name': model_name,
                'cache_info': cache_info
            }
        else:
            # Checkpoint fortsetzen
            print(f"\n✅ Checkpoint ausgewählt: {selected_checkpoint['filename']}")

            # Bereinige Logs für Checkpoint-Resume (vor Pipeline!)
            model_name = selected_checkpoint['model_name']
            print(f"\n🧹 Bereinige verwaiste Logs für: {model_name}")
            self.cleanup_orphaned_logs(model_name)

            # Checkpoint fortsetzen - Cache aus Checkpoint-Info extrahieren
            cache_info = self.extract_cache_from_checkpoint(selected_checkpoint)
            return {
                'mode': 'checkpoint',
                'checkpoint_info': selected_checkpoint,
                'cache_info': cache_info
            }
    
    def cleanup_orphaned_logs(self, model_name: str):
        """Bereinigt Logs ohne korrespondierenden Checkpoint."""
        print(f"🧹 Bereinige verwaiste Logs für Modell: {model_name}")

        # Scanne Checkpoints für dieses Modell
        checkpoints = self.checkpoint_manager.scan_checkpoints()
        checkpoint_runs = set()
        for cp in checkpoints:
            if cp['model_name'] == model_name:
                checkpoint_runs.add(cp['run_id'])

        # Scanne Log-Dateien
        log_dir = "training_logs"
        if not os.path.exists(log_dir):
            return

        deleted_count = 0
        for filename in os.listdir(log_dir):
            if filename.startswith(f"{model_name}_run_") and filename.endswith(".json"):
                try:
                    # Extrahiere Run-ID aus Log-Datei
                    run_part = filename.replace(f"{model_name}_run_", "").replace(".json", "").replace("_summary", "")
                    log_run_id = int(run_part)

                    # Lösche Log wenn kein korrespondierender Checkpoint existiert
                    if log_run_id not in checkpoint_runs:
                        log_path = os.path.join(log_dir, filename)
                        try:
                            os.remove(log_path)
                            print(f"   🗑️  Gelöscht: {filename} (kein Checkpoint für Run {log_run_id})")
                            deleted_count += 1
                        except OSError as e:
                            print(f"   ⚠️  Konnte {filename} nicht löschen: {e}")

                except ValueError:
                    continue

        if deleted_count == 0:
            print("   ✅ Keine verwaisten Logs gefunden")
        else:
            print(f"   ✅ {deleted_count} verwaiste Log-Dateien bereinigt")

    def select_cache_for_new_training(self):
        """Auswahl des Caches für neues Training."""
        from ..utils.cache_registry import load_cache_registry, display_cache_menu

        print(f"\n📦 CACHE AUSWAHL FÜR NEUES TRAINING")
        print("=" * 50)

        # Lade verfügbare Caches
        caches = load_cache_registry()

        if not caches:
            print("❌ Keine Caches verfügbar!")
            print("Erstelle zuerst einen Cache mit:")
            print("python scripts/create_packed_cache.py")
            return None

        # Zeige Cache-Menu
        selected_cache = display_cache_menu(caches)

        if selected_cache:
            print(f"✅ Cache ausgewählt: {selected_cache['dataset_name']} (seq_len: {selected_cache['sequence_length']})")
            return selected_cache
        else:
            print("❌ Kein Cache ausgewählt")
            return None

    def extract_cache_from_checkpoint(self, checkpoint_info):
        """Extrahiert Cache-Info aus Checkpoint mit robustem Fallback."""
        import torch
        from ..utils.cache_registry import load_cache_registry, find_cache_by_sequence_length

        try:
            # Lade Checkpoint um Cache-Info zu extrahieren
            checkpoint = torch.load(checkpoint_info['filepath'], map_location='cpu', weights_only=False)

            # Neue Checkpoints haben cache_info
            if 'cache_info' in checkpoint:
                cache_info = checkpoint['cache_info']
                print(f"📦 Cache aus Checkpoint: {cache_info['dataset_name']} (seq_len: {cache_info['sequence_length']})")

                # Validiere dass Cache noch existiert und aktualisiere Größe
                from ..utils.cache_registry import validate_cache, load_cache_registry
                print(f"🔍 Validating cache: {cache_info['dataset_name']} at {cache_info.get('path', 'unknown')}")
                if validate_cache(cache_info):
                    # FIXED: Update cache info with current size (for expanded datasets)
                    current_caches = load_cache_registry()
                    for current_cache in current_caches:
                        if (current_cache['dataset_name'] == cache_info['dataset_name'] and
                            current_cache['sequence_length'] == cache_info['sequence_length']):
                            old_sequences = cache_info.get('total_sequences', 0)
                            new_sequences = current_cache.get('total_sequences', 0)

                            if new_sequences > old_sequences:
                                print(f"📈 Dataset expanded: {old_sequences:,} → {new_sequences:,} sequences")
                                cache_info['total_sequences'] = new_sequences
                                cache_info['path'] = current_cache['path']  # Update path too

                            break

                    print(f"✅ Cache validation successful")
                    return cache_info
                else:
                    print(f"⚠️  Cache validation failed, suche Alternative...")

            # Fallback 1: Aus training_config sequence_length ableiten
            if 'training_config' in checkpoint:
                training_config = checkpoint['training_config']
                seq_len = training_config.get('sequence_length', 512)

                cache = find_cache_by_sequence_length(seq_len)
                if cache:
                    print(f"📦 Cache gefunden für seq_len {seq_len}: {cache['dataset_name']}")
                    return cache

            # Fallback 2: Ersten verfügbaren Cache verwenden
            caches = load_cache_registry()
            if caches:
                cache = caches[0]
                print(f"📦 Fallback Cache: {cache['dataset_name']} (seq_len: {cache['sequence_length']})")
                return cache

            # Fallback 3: Hardcoded Default
            print("⚠️  Kein Cache gefunden, verwende Fallback")
            return {
                'dataset_name': 'FineWeb',
                'sequence_length': 512,
                'path': 'cache/packed_sequences/512/FineWeb',
                'total_sequences': 352217
            }

        except Exception as e:
            print(f"⚠️  Fehler beim Extrahieren der Cache-Info: {e}")

            # Ultimate Fallback
            caches = load_cache_registry()
            if caches:
                return caches[0]

            return {
                'dataset_name': 'FineWeb',
                'sequence_length': 512,
                'path': 'cache/packed_sequences/512/FineWeb',
                'total_sequences': 352217
            }


def get_next_run_id(model_name: str) -> int:
    """Ermittelt nächste Run-ID für ein Modell basierend auf Checkpoints."""
    checkpoint_manager = CheckpointManager()
    checkpoints = checkpoint_manager.scan_checkpoints()

    # Finde höchste Run-ID für dieses Modell
    max_run_id = 0
    for cp in checkpoints:
        if cp['model_name'] == model_name:
            max_run_id = max(max_run_id, cp['run_id'])

    return max_run_id + 1


def handle_training_mode_selection() -> Dict:
    """Convenience function für Training Mode Selection."""
    training_state = TrainingState()
    return training_state.handle_training_mode_selection()


def load_checkpoint_for_training(checkpoint_info: Dict):
    """Lädt Checkpoint für Training-Fortsetzung."""
    checkpoint_manager = CheckpointManager()
    return checkpoint_manager.load_checkpoint(checkpoint_info)
