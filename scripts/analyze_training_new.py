#!/usr/bin/env python3
"""
OVERHAULED Training Analysis - Professional Training Insights
"""

import os
import sys
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
from training_logger import TrainingLogger, compare_training_runs

def get_available_models_and_runs():
    """Scannt verfügbare Modelle und Runs."""
    log_dir = "training_logs"
    if not os.path.exists(log_dir):
        return {}

    models = {}
    for filename in os.listdir(log_dir):
        if filename.endswith(".json") and "_run_" in filename:
            # Parse: MODELNAME_run_X.json
            try:
                base_name = filename.replace(".json", "").replace("_summary", "")
                if "_run_" in base_name:
                    model_name, run_part = base_name.rsplit("_run_", 1)
                    run_id = int(run_part)

                    if model_name not in models:
                        models[model_name] = []
                    if run_id not in models[model_name]:
                        models[model_name].append(run_id)
            except ValueError:
                continue

    # Sortiere Runs
    for model_name in models:
        models[model_name].sort()

    return models

def select_log_file():
    """Interaktive Auswahl von Modell und Run."""
    models = get_available_models_and_runs()

    if not models:
        print("❌ No training logs found in training_logs/")
        return None

    print("📊 Available Models and Runs:")
    print("="*50)

    # Zeige verfügbare Modelle
    model_list = list(models.keys())
    for i, model_name in enumerate(model_list, 1):
        runs = models[model_name]
        print(f"  {i}. {model_name} (Runs: {', '.join(map(str, runs))})")

    # Modell auswählen
    try:
        model_choice = int(input(f"\nSelect model (1-{len(model_list)}): ")) - 1
        if model_choice < 0 or model_choice >= len(model_list):
            print("❌ Invalid model selection")
            return None

        selected_model = model_list[model_choice]
        available_runs = models[selected_model]

        # Run auswählen
        print(f"\nAvailable runs for {selected_model}: {', '.join(map(str, available_runs))}")
        run_choice = int(input(f"Select run (or press Enter for latest): ") or str(max(available_runs)))

        if run_choice not in available_runs:
            print("❌ Invalid run selection")
            return None

        # Konstruiere Dateiname
        log_file = os.path.join("training_logs", f"{selected_model}_run_{run_choice}.json")
        print(f"📊 Selected: {log_file}")
        return log_file

    except (ValueError, KeyboardInterrupt):
        print("❌ Selection cancelled")
        return None

def enhanced_analysis(log_file: str = None):
    """OVERHAULED Training Analysis with detailed insights."""

    # Auto-detect or select log file
    if log_file is None:
        log_file = select_log_file()
        if log_file is None:
            return
    
    if not os.path.exists(log_file):
        print(f"❌ Log file not found: {log_file}")
        return
    
    # Load and analyze
    logger = TrainingLogger()
    if logger.load_from_file(log_file):
        
        # Extract metadata for better naming
        df = pd.DataFrame(logger.training_data)
        model_name = logger.metadata.get('model_name', 'model')
        run_id = f"run_{hash(log_file) % 1000:03d}"  # Simple run ID from file hash
        first_step = df['step'].min()
        last_step = df['step'].max()
        total_logged = len(logger.training_data)
        
        print("="*75)
        print("📊 TRAINING ANALYSIS SUMMARY")
        print("="*75)
        print(f"   total_logged_steps       : {total_logged}")
        print(f"   last_step                : {last_step}")
        
        # RECENT PERFORMANCE (Last 10 logged steps)
        if len(logger.training_data) >= 10:
            recent_data = df.tail(10)
            print(f"\n📈 RECENT PERFORMANCE (Last 10 logged steps):")
            print(f"   Average Loss:        {recent_data['loss'].mean():.4f}")
            print(f"   Average Tokens/sec:  {recent_data['tokens_per_sec'].mean():.0f}")
            print(f"   Average Step Time:   {recent_data['step_time'].mean():.2f}s")
        
        # OVERALL PERFORMANCE (All logged steps)
        print(f"\n📈 PERFORMANCE OVERALL (All {total_logged} logged steps):")
        print(f"   Average Loss:        {df['loss'].mean():.4f}")
        print(f"   Average Tokens/sec:  {df['tokens_per_sec'].mean():.0f}")
        print(f"   Average Step Time:   {df['step_time'].mean():.2f}s")
        
        # TRENDS (Recent vs Earlier)
        if len(df) >= 20:
            old_data = df.iloc[-20:-10] if len(df) >= 20 else df.iloc[:len(df)//2]
            new_data = df.tail(10)
            
            loss_trend = (new_data['loss'].mean() - old_data['loss'].mean()) / old_data['loss'].mean() * 100
            speed_trend = (new_data['tokens_per_sec'].mean() - old_data['tokens_per_sec'].mean()) / old_data['tokens_per_sec'].mean() * 100
            
            print(f"\n📈 TRENDS (All logged steps):")
            print(f"   Loss Change:         {loss_trend:+.1f}%")
            print(f"   Speed Change:        {speed_trend:+.1f}%")
        
        # Generate plots with better naming
        print(f"\nGenerating training plots...")
        
        # Generate plots with custom names
        plot_recent = f"logs/{model_name}_{run_id}_steps_{max(1, last_step-100)}_to_{last_step}.png"
        plot_all = f"logs/{model_name}_{run_id}_steps_all.png"
        
        logger.generate_plots(save_plots=True)  # This saves to default location
        
        # Rename plots to better names
        import shutil
        default_plot = f"logs/{model_name}_training_plots.png"
        if os.path.exists(default_plot):
            try:
                shutil.copy(default_plot, plot_all)
                print(f"📈 Plots saved: {plot_all}")
                print(f"📈 Plots saved: {plot_recent} (copy of full plot)")
            except:
                print(f"📈 Plots saved: {default_plot}")
        
        # ENHANCED HEALTH CHECK with explanations
        print(f"\n🏥 TRAINING HEALTH CHECK:")
        
        # Loss stability analysis
        if len(df) >= 10:
            recent_losses = df['loss'].tail(10)
            loss_std = recent_losses.std()
            loss_mean = recent_losses.mean()
            loss_cv = loss_std / loss_mean if loss_mean > 0 else float('inf')
            
            if loss_cv < 0.1:
                stability_status = "✅ Loss stability: Good"
                stability_reason = f"(CV={loss_cv:.3f} < 0.1, low variance in recent losses)"
            elif loss_cv < 0.3:
                stability_status = "⚠️  Loss stability: Moderate"
                stability_reason = f"(CV={loss_cv:.3f}, some variance in recent losses)"
            else:
                stability_status = "❌ Loss stability: Poor"
                stability_reason = f"(CV={loss_cv:.3f} > 0.3, high variance indicates instability)"
            
            print(f"   {stability_status}")
            print(f"       {stability_reason}")
        
        # Speed consistency analysis
        if len(df) >= 10:
            recent_speeds = df['tokens_per_sec'].tail(10)
            speed_std = recent_speeds.std()
            speed_mean = recent_speeds.mean()
            speed_cv = speed_std / speed_mean if speed_mean > 0 else float('inf')
            
            if speed_cv < 0.1:
                speed_status = "✅ Speed consistency: Good"
                speed_reason = f"(CV={speed_cv:.3f} < 0.1, consistent performance)"
            elif speed_cv < 0.2:
                speed_status = "⚠️  Speed consistency: Moderate"
                speed_reason = f"(CV={speed_cv:.3f}, some performance variation)"
            else:
                speed_status = "❌ Speed consistency: Poor"
                speed_reason = f"(CV={speed_cv:.3f} > 0.2, high performance variation)"
            
            print(f"   {speed_status}")
            print(f"       {speed_reason}")
        
        print("="*75)

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
        # No arguments - enhanced analysis
        enhanced_analysis()
    
    elif sys.argv[1] == "export":
        # Export to CSV
        log_file = sys.argv[2] if len(sys.argv) > 2 else None
        output_file = sys.argv[3] if len(sys.argv) > 3 else None
        export_csv(log_file, output_file)
    
    else:
        # Analyze specific file
        enhanced_analysis(sys.argv[1])
