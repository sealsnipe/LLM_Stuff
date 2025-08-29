"""
System Utilities Module

Contains system-level utility functions and the SystemUtils class.
Handles platform-specific optimizations, terminal fixes, and system setup.
"""

import os
import sys
import platform
import subprocess
from typing import Dict, List, Optional


class SystemUtils:
    """Utility-Klasse für System-Management und -Optimierung."""
    
    def __init__(self):
        self.platform = platform.system().lower()
        self.is_windows = self.platform == 'windows'
        self.is_linux = self.platform == 'linux'
        self.is_mac = self.platform == 'darwin'
    
    def fix_windows_terminal(self):
        """Behebt Windows Terminal-Probleme für Progress-Anzeige."""
        if not self.is_windows:
            return
        
        try:
            # Windows Terminal ANSI Support
            import ctypes
            kernel32 = ctypes.windll.kernel32
            
            # Enable ANSI escape sequences
            STD_OUTPUT_HANDLE = -11
            ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
            
            handle = kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
            mode = ctypes.c_ulong()
            kernel32.GetConsoleMode(handle, ctypes.byref(mode))
            mode.value |= ENABLE_VIRTUAL_TERMINAL_PROCESSING
            kernel32.SetConsoleMode(handle, mode)
            
            pass
            
        except Exception as e:
            pass
    
    def optimize_system_for_training(self):
        """Optimiert System-Einstellungen für Training."""
        # Platform-spezifische Optimierungen
        if self.is_windows:
            self._optimize_windows()
        elif self.is_linux:
            self._optimize_linux()
        elif self.is_mac:
            self._optimize_mac()

        # Allgemeine Optimierungen
        self._set_environment_variables()
        self._optimize_python_settings()
    
    def _optimize_windows(self):
        """Windows-spezifische Optimierungen."""
        try:
            # Windows Terminal Fix
            self.fix_windows_terminal()
            
            # Process Priority (falls Administrator)
            try:
                import psutil
                current_process = psutil.Process()
                current_process.nice(psutil.HIGH_PRIORITY_CLASS)
                pass
            except:
                pass
            
            # Windows-spezifische Environment Variables
            windows_env = {
                "PYTHONUNBUFFERED": "1",
                "OMP_NUM_THREADS": str(os.cpu_count() // 2),
                "MKL_NUM_THREADS": str(os.cpu_count() // 2)
            }
            
            for key, value in windows_env.items():
                os.environ.setdefault(key, value)
                
        except Exception as e:
            pass
    
    def _optimize_linux(self):
        """Linux-spezifische Optimierungen."""
        try:
            # CPU Governor (falls verfügbar)
            try:
                subprocess.run(["sudo", "cpupower", "frequency-set", "-g", "performance"], 
                             check=False, capture_output=True)
                print("   ✅ CPU Governor auf Performance gesetzt")
            except:
                print("   ⚠️ CPU Governor nicht änderbar")
            
            # Linux-spezifische Environment Variables
            linux_env = {
                "OMP_NUM_THREADS": str(os.cpu_count()),
                "MKL_NUM_THREADS": str(os.cpu_count()),
                "OPENBLAS_NUM_THREADS": str(os.cpu_count())
            }
            
            for key, value in linux_env.items():
                os.environ.setdefault(key, value)
                print(f"   ✅ {key}={value}")
                
        except Exception as e:
            print(f"   ⚠️ Linux-Optimierung fehlgeschlagen: {e}")
    
    def _optimize_mac(self):
        """macOS-spezifische Optimierungen."""
        try:
            # macOS-spezifische Environment Variables
            mac_env = {
                "OMP_NUM_THREADS": str(os.cpu_count()),
                "MKL_NUM_THREADS": str(os.cpu_count()),
                "VECLIB_MAXIMUM_THREADS": str(os.cpu_count())
            }
            
            for key, value in mac_env.items():
                os.environ.setdefault(key, value)
                print(f"   ✅ {key}={value}")
                
        except Exception as e:
            print(f"   ⚠️ macOS-Optimierung fehlgeschlagen: {e}")
    
    def _set_environment_variables(self):
        """Setzt allgemeine Environment Variables."""
        general_env = {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",  # Verhindert Tokenizer-Warnings
            "TRANSFORMERS_OFFLINE": "0",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "PYTHONWARNINGS": "ignore",  # Alle Python Warnings ausblenden
            "TF_CPP_MIN_LOG_LEVEL": "3",  # TensorFlow Warnings ausblenden
            "TORCH_LOGS": "",  # Torch Logs ausblenden
            "TORCHDYNAMO_VERBOSE": "0",  # Dynamo Verbose ausblenden
            "TORCH_COMPILE_DEBUG": "0"  # Compile Debug ausblenden
        }
        
        for key, value in general_env.items():
            os.environ.setdefault(key, value)
    
    def _optimize_python_settings(self):
        """Optimiert Python-spezifische Einstellungen."""
        try:
            # Warnings unterdrücken
            import warnings
            warnings.filterwarnings("ignore")

            # Logging unterdrücken
            import logging
            logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
            logging.getLogger("torch._inductor").setLevel(logging.ERROR)
            logging.getLogger("torch.fx").setLevel(logging.ERROR)
            logging.getLogger().setLevel(logging.ERROR)

            # Garbage Collection Tuning
            import gc
            gc.set_threshold(700, 10, 10)  # Weniger aggressive GC
            pass

            # Threading Optimierung
            import threading
            threading.stack_size(8 * 1024 * 1024)  # 8MB Stack Size
            
        except Exception as e:
            pass
    
    def get_system_info(self) -> Dict:
        """Sammelt System-Informationen."""
        try:
            import psutil
            
            # CPU Info
            cpu_info = {
                'physical_cores': psutil.cpu_count(logical=False),
                'logical_cores': psutil.cpu_count(logical=True),
                'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                'cpu_percent': psutil.cpu_percent(interval=1)
            }
            
            # Memory Info
            memory = psutil.virtual_memory()
            memory_info = {
                'total_gb': memory.total / 1e9,
                'available_gb': memory.available / 1e9,
                'used_gb': memory.used / 1e9,
                'percent': memory.percent
            }
            
            # Disk Info
            disk = psutil.disk_usage('/')
            disk_info = {
                'total_gb': disk.total / 1e9,
                'free_gb': disk.free / 1e9,
                'used_gb': disk.used / 1e9,
                'percent': (disk.used / disk.total) * 100
            }
            
            return {
                'platform': {
                    'system': platform.system(),
                    'release': platform.release(),
                    'version': platform.version(),
                    'machine': platform.machine(),
                    'processor': platform.processor()
                },
                'python': {
                    'version': sys.version,
                    'executable': sys.executable
                },
                'cpu': cpu_info,
                'memory': memory_info,
                'disk': disk_info
            }
            
        except Exception as e:
            return {'error': f"Konnte System-Info nicht sammeln: {e}"}
    
    def check_system_requirements(self) -> Dict:
        """Prüft System-Anforderungen für Training."""
        requirements = {
            'python_version': (3, 8),
            'min_memory_gb': 16,
            'min_disk_space_gb': 50,
            'recommended_cores': 8
        }
        
        system_info = self.get_system_info()
        checks = {}
        
        try:
            # Python Version Check
            python_version = sys.version_info[:2]
            checks['python_version'] = {
                'required': requirements['python_version'],
                'current': python_version,
                'passed': python_version >= requirements['python_version']
            }
            
            # Memory Check
            if 'memory' in system_info:
                memory_gb = system_info['memory']['total_gb']
                checks['memory'] = {
                    'required_gb': requirements['min_memory_gb'],
                    'current_gb': memory_gb,
                    'passed': memory_gb >= requirements['min_memory_gb']
                }
            
            # Disk Space Check
            if 'disk' in system_info:
                free_gb = system_info['disk']['free_gb']
                checks['disk_space'] = {
                    'required_gb': requirements['min_disk_space_gb'],
                    'current_gb': free_gb,
                    'passed': free_gb >= requirements['min_disk_space_gb']
                }
            
            # CPU Cores Check
            if 'cpu' in system_info:
                cores = system_info['cpu']['logical_cores']
                checks['cpu_cores'] = {
                    'recommended': requirements['recommended_cores'],
                    'current': cores,
                    'passed': cores >= requirements['recommended_cores']
                }
            
        except Exception as e:
            checks['error'] = str(e)
        
        return checks
    
    def print_system_report(self):
        """Druckt umfassenden System-Report."""
        print("\n" + "="*60)
        print("🖥️  SYSTEM REPORT")
        print("="*60)
        
        # System Info
        system_info = self.get_system_info()
        if 'platform' in system_info:
            platform_info = system_info['platform']
            print(f"Platform: {platform_info['system']} {platform_info['release']}")
            print(f"Machine: {platform_info['machine']}")
        
        # Requirements Check
        requirements = self.check_system_requirements()
        print("\n📋 REQUIREMENTS CHECK:")
        
        for check_name, check_data in requirements.items():
            if check_name == 'error':
                print(f"❌ Error: {check_data}")
                continue
            
            if isinstance(check_data, dict) and 'passed' in check_data:
                status = "✅" if check_data['passed'] else "❌"
                print(f"{status} {check_name}: {check_data}")
        
        print("="*60)


def fix_windows_terminal():
    """Convenience function für Windows Terminal Fix."""
    system_utils = SystemUtils()
    system_utils.fix_windows_terminal()


def optimize_system_for_training():
    """Convenience function für System-Optimierung."""
    system_utils = SystemUtils()
    system_utils.optimize_system_for_training()


def get_system_info() -> Dict:
    """Convenience function für System-Info."""
    system_utils = SystemUtils()
    return system_utils.get_system_info()


def print_system_report():
    """Convenience function für System-Report."""
    system_utils = SystemUtils()
    system_utils.print_system_report()
