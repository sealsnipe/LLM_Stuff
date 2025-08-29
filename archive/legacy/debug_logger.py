#!/usr/bin/env python3
"""
🔧 DEBUG LOGGER SYSTEM
Handles detailed debug logs and error logs separately from main console output
"""

import os
import time
import traceback
from datetime import datetime
from typing import Optional
from config import training_config

class DebugLogger:
    """
    🔧 Debug Logger for detailed training information
    
    Features:
    - Debug logs (when debug_mode=True)
    - Error logs (always active)
    - Automatic file rotation
    - Clean startup (removes old logs)
    """
    
    def __init__(self, run_id: Optional[str] = None):
        self.run_id = run_id or f"run_{int(time.time())}"
        self.debug_enabled = training_config.debug_mode
        
        # Create log files with run_id
        self.debug_file = f"debug_{self.run_id}.log"
        self.error_file = f"error_{self.run_id}.log"
        
        # Clean old logs on startup
        self._cleanup_old_logs()
        
        # Initialize files
        if self.debug_enabled:
            self._init_debug_log()
        self._init_error_log()
    
    def _cleanup_old_logs(self):
        """Remove old debug/error logs to keep workspace clean"""
        for filename in os.listdir('.'):
            if filename.startswith(('debug_', 'error_')) and filename.endswith('.log'):
                try:
                    os.remove(filename)
                except OSError:
                    pass
    
    def _init_debug_log(self):
        """Initialize debug log file"""
        with open(self.debug_file, 'w', encoding='utf-8') as f:
            f.write(f"🔧 DEBUG LOG - {datetime.now().isoformat()}\n")
            f.write(f"Run ID: {self.run_id}\n")
            f.write("=" * 80 + "\n\n")
    
    def _init_error_log(self):
        """Initialize error log file"""
        with open(self.error_file, 'w', encoding='utf-8') as f:
            f.write(f"❌ ERROR LOG - {datetime.now().isoformat()}\n")
            f.write(f"Run ID: {self.run_id}\n")
            f.write("=" * 80 + "\n\n")
    
    def debug(self, message: str, category: str = "DEBUG"):
        """Log debug message (only if debug_mode=True)"""
        if not self.debug_enabled:
            return
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_line = f"[{timestamp}] [{category}] {message}\n"
        
        try:
            with open(self.debug_file, 'a', encoding='utf-8') as f:
                f.write(log_line)
        except OSError:
            pass  # Silent fail for debug logs
    
    def error(self, message: str, exception: Optional[Exception] = None):
        """Log error message (always active)"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_line = f"[{timestamp}] [ERROR] {message}\n"
        
        if exception:
            log_line += f"Exception: {str(exception)}\n"
            log_line += f"Traceback:\n{traceback.format_exc()}\n"
        
        log_line += "-" * 40 + "\n"
        
        try:
            with open(self.error_file, 'a', encoding='utf-8') as f:
                f.write(log_line)
        except OSError:
            pass  # Silent fail
    
    def packed_cache(self, message: str):
        """Log packed cache operations"""
        self.debug(message, "PACKED_CACHE")
    
    def intelligent_training(self, message: str):
        """Log intelligent training adjustments"""
        self.debug(message, "INTELLIGENT")
    
    def production_stats(self, message: str):
        """Log production statistics"""
        self.debug(message, "STATS")
    
    def checkpoint(self, message: str):
        """Log checkpoint operations"""
        self.debug(message, "CHECKPOINT")

# Global debug logger instance
_debug_logger: Optional[DebugLogger] = None

def get_debug_logger(run_id: Optional[str] = None) -> DebugLogger:
    """Get or create global debug logger instance"""
    global _debug_logger
    if _debug_logger is None:
        _debug_logger = DebugLogger(run_id)
    return _debug_logger

def debug_log(message: str, category: str = "DEBUG"):
    """Convenience function for debug logging"""
    get_debug_logger().debug(message, category)

def error_log(message: str, exception: Optional[Exception] = None):
    """Convenience function for error logging"""
    get_debug_logger().error(message, exception)

def packed_cache_log(message: str):
    """Convenience function for packed cache logging"""
    get_debug_logger().packed_cache(message)

def intelligent_training_log(message: str):
    """Convenience function for intelligent training logging"""
    get_debug_logger().intelligent_training(message)

def production_stats_log(message: str):
    """Convenience function for production stats logging"""
    get_debug_logger().production_stats(message)

def checkpoint_log(message: str):
    """Convenience function for checkpoint logging"""
    get_debug_logger().checkpoint(message)
