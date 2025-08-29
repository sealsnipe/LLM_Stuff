#!/usr/bin/env python3
"""
Quick Training Analysis Script
Usage: python analyze_training.py [log_file]
"""

import sys
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from training_logger import TrainingLogger, analyze_training_log, compare_training_runs

def quick_analysis(log_file: str = None):
    """Quick analysis of current training."""
    
    # Auto-detect log file if not provided
    if log_file is None:
        log_dir = "training_logs"
        if os.path.exists(log_dir):
            log_files = [f for f in os.listdir(log_dir) if f.endswith("_training.json")]
            if log_files:
                log_file = os.path.join(log_dir, sorted(log_files)[-1])  # Latest file
                print(f"📊 Auto-detected log file: {log_file}")
            else:
                print("❌ No training log files found in training_logs/")
                return
        else:
            print("❌ No training_logs directory found")
            return
    
    if not os.path.exists(log_file):
        print(f"❌ Log file not found: {log_file}")
        return
    
    # Load and analyze
    logger = TrainingLogger()
    if logger.load_from_file(log_file):
        
        # Print summary
        summary = logger.get_summary()
        print("\n" + "="*60)
        print("📊 TRAINING ANALYSIS SUMMARY")
        print("="*60)
        
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"   {key:25}: {value:.4f}")
            else:
                print(f"   {key:25}: {value}")
        
        # Generate plots
        print("\n📈 Generating training plots...")
        logger.generate_plots(save_plots=True)
        
        # Recent performance analysis
        df = pd.DataFrame(logger.training_data)
        if len(df) >= 10:
            recent_data = df.tail(10)
            print(f"\n📈 RECENT PERFORMANCE (Last 10 logged steps):")
            print(f"   Average Loss:        {recent_data['loss'].mean():.4f}")
            print(f"   Average Tokens/sec:  {recent_data['tokens_per_sec'].mean():.0f}")
            print(f"   Average Step Time:   {recent_data['step_time'].mean():.2f}s")
            
            # Performance trend
            if len(df) >= 20:
                old_data = df.iloc[-20:-10]
                new_data = df.tail(10)
                
                loss_trend = (new_data['loss'].mean() - old_data['loss'].mean()) / old_data['loss'].mean() * 100
                speed_trend = (new_data['tokens_per_sec'].mean() - old_data['tokens_per_sec'].mean()) / old_data['tokens_per_sec'].mean() * 100
                
                print(f"\n📊 PERFORMANCE TRENDS:")
                print(f"   Loss Change:         {loss_trend:+.1f}%")
                print(f"   Speed Change:        {speed_trend:+.1f}%")
        
        # Training health check
        print(f"\n🏥 TRAINING HEALTH CHECK:")
        
        # Check for loss spikes
        if len(df) >= 5:
            loss_std = df['loss'].tail(20).std() if len(df) >= 20 else df['loss'].std()
            recent_loss = df['loss'].iloc[-1]
            avg_loss = df['loss'].tail(10).mean()
            
            if recent_loss > avg_loss + 2 * loss_std:
                print("   ⚠️  WARNING: Recent loss spike detected!")
            else:
                print("   ✅ Loss stability: Good")
        
        # Check performance consistency
        if len(df) >= 10:
            speed_cv = df['tokens_per_sec'].tail(10).std() / df['tokens_per_sec'].tail(10).mean()
            if speed_cv > 0.1:
                print("   ⚠️  WARNING: Inconsistent training speed")
            else:
                print("   ✅ Speed consistency: Good")
        
        print("\n" + "="*60)

def live_monitor(log_file: str = None, refresh_seconds: int = 30):
    """Live monitoring of training progress."""
    import time
    
    print("🔴 LIVE TRAINING MONITOR")
    print("Press Ctrl+C to stop")
    print("="*60)
    
    try:
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')  # Clear screen
            print(f"🔴 LIVE MONITOR - {pd.Timestamp.now().strftime('%H:%M:%S')}")
            print("="*60)
            
            quick_analysis(log_file)
            
            print(f"\n⏱️  Refreshing in {refresh_seconds} seconds...")
            time.sleep(refresh_seconds)
            
    except KeyboardInterrupt:
        print("\n👋 Live monitor stopped")

def compare_runs():
    """Compare multiple training runs."""
    log_dir = "training_logs"
    if not os.path.exists(log_dir):
        print("❌ No training_logs directory found")
        return
    
    log_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith("_training.json")]
    
    if len(log_files) < 2:
        print("❌ Need at least 2 training runs to compare")
        return
    
    print(f"📊 Comparing {len(log_files)} training runs...")
    compare_training_runs(log_files)

def export_csv(log_file: str = None, output_file: str = None):
    """Export training data to CSV for external analysis."""
    
    # Auto-detect log file
    if log_file is None:
        log_dir = "training_logs"
        if os.path.exists(log_dir):
            log_files = [f for f in os.listdir(log_dir) if f.endswith("_training.json")]
            if log_files:
                log_file = os.path.join(log_dir, sorted(log_files)[-1])
            else:
                print("❌ No training log files found")
                return
    
    # Load data
    logger = TrainingLogger()
    if not logger.load_from_file(log_file):
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(logger.training_data)
    
    # Auto-generate output filename
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(log_file))[0]
        output_file = f"{base_name}.csv"
    
    # Export
    df.to_csv(output_file, index=False)
    print(f"📊 Training data exported to: {output_file}")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {list(df.columns)}")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments - quick analysis
        quick_analysis()
    
    elif sys.argv[1] == "live":
        # Live monitoring
        log_file = sys.argv[2] if len(sys.argv) > 2 else None
        refresh = int(sys.argv[3]) if len(sys.argv) > 3 else 30
        live_monitor(log_file, refresh)
    
    elif sys.argv[1] == "compare":
        # Compare runs
        compare_runs()
    
    elif sys.argv[1] == "export":
        # Export to CSV
        log_file = sys.argv[2] if len(sys.argv) > 2 else None
        output_file = sys.argv[3] if len(sys.argv) > 3 else None
        export_csv(log_file, output_file)
    
    else:
        # Analyze specific file
        quick_analysis(sys.argv[1])
