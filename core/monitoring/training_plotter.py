"""
Professional Training Plots Generator

Creates beautiful 4-panel training plots from JSON logs.
Compatible with existing log format.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List, Optional
from datetime import datetime


class TrainingPlotter:
    """Professional training plots generator."""
    
    def __init__(self):
        self.style_config = {
            'figure_size': (16, 12),
            'dpi': 300,
            'grid_alpha': 0.3,
            'line_width': 2,
            'colors': {
                'loss': '#e74c3c',
                'lr': '#3498db', 
                'tokens_per_sec': '#2ecc71',
                'gpu_memory': '#f39c12'
            }
        }
    
    def create_training_plots(self, json_log_path: str, output_path: Optional[str] = None) -> str:
        """Create 4-panel training plots from JSON log."""
        
        # Load training data
        with open(json_log_path, 'r') as f:
            log_data = json.load(f)
        
        metadata = log_data['metadata']
        training_data = log_data['training_data']
        
        if not training_data:
            raise ValueError("No training data found in log file")
        
        # Extract data
        steps = [d['step'] for d in training_data]
        losses = [d['loss'] for d in training_data]
        lrs = [d['lr'] for d in training_data]
        tokens_per_sec = [d['tokens_per_sec'] for d in training_data]
        gpu_memory = [d['gpu_memory_gb'] for d in training_data]
        
        # Create figure
        plt.style.use('default')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.style_config['figure_size'])
        fig.suptitle(f"Training Progress: {metadata['model_name']}", fontsize=16, fontweight='bold')
        
        # Plot 1: Loss
        ax1.plot(steps, losses, color=self.style_config['colors']['loss'], 
                linewidth=self.style_config['line_width'], label='Training Loss')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True, alpha=self.style_config['grid_alpha'])
        ax1.legend()
        
        # Add loss trend
        if len(steps) > 10:
            z = np.polyfit(steps, losses, 1)
            p = np.poly1d(z)
            ax1.plot(steps, p(steps), "--", alpha=0.7, color='darkred', label='Trend')
            ax1.legend()
        
        # Plot 2: Learning Rate
        ax2.plot(steps, lrs, color=self.style_config['colors']['lr'], 
                linewidth=self.style_config['line_width'], label='Learning Rate')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True, alpha=self.style_config['grid_alpha'])
        ax2.legend()
        ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # Plot 3: Tokens per Second
        ax3.plot(steps, tokens_per_sec, color=self.style_config['colors']['tokens_per_sec'], 
                linewidth=self.style_config['line_width'], label='Tokens/sec')
        ax3.set_xlabel('Steps')
        ax3.set_ylabel('Tokens per Second')
        ax3.set_title('Training Throughput')
        ax3.grid(True, alpha=self.style_config['grid_alpha'])
        ax3.legend()
        
        # Add throughput stats
        avg_throughput = np.mean(tokens_per_sec)
        ax3.axhline(y=avg_throughput, color='darkgreen', linestyle='--', alpha=0.7, 
                   label=f'Avg: {avg_throughput:,.0f} tok/s')
        ax3.legend()
        
        # Plot 4: GPU Memory
        ax4.plot(steps, gpu_memory, color=self.style_config['colors']['gpu_memory'], 
                linewidth=self.style_config['line_width'], label='GPU Memory')
        ax4.set_xlabel('Steps')
        ax4.set_ylabel('GPU Memory (GB)')
        ax4.set_title('GPU Memory Usage')
        ax4.grid(True, alpha=self.style_config['grid_alpha'])
        ax4.legend()
        
        # Add memory stats
        max_memory = max(gpu_memory)
        avg_memory = np.mean(gpu_memory)
        ax4.axhline(y=avg_memory, color='darkorange', linestyle='--', alpha=0.7, 
                   label=f'Avg: {avg_memory:.1f} GB')
        ax4.axhline(y=max_memory, color='red', linestyle=':', alpha=0.7, 
                   label=f'Peak: {max_memory:.1f} GB')
        ax4.legend()
        
        # Add training info
        total_steps = metadata.get('total_steps', 'Unknown')
        start_time = metadata.get('start_time', 'Unknown')
        if start_time != 'Unknown':
            start_time = datetime.fromisoformat(start_time).strftime('%Y-%m-%d %H:%M')
        
        info_text = f"""Training Info:
Model: {metadata['model_name']}
Started: {start_time}
Target Steps: {total_steps:,}
Current Step: {max(steps):,}
Progress: {(max(steps)/total_steps*100):.1f}%"""
        
        fig.text(0.02, 0.02, info_text, fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.93, bottom=0.15)
        
        # Save plot
        if output_path is None:
            output_path = json_log_path.replace('.json', '_training_plots.png')
        
        plt.savefig(output_path, dpi=self.style_config['dpi'], bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def create_comparison_plots(self, log_paths: List[str], output_path: str) -> str:
        """Create comparison plots for multiple training runs."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.style_config['figure_size'])
        fig.suptitle("Training Runs Comparison", fontsize=16, fontweight='bold')
        
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
        
        for i, log_path in enumerate(log_paths):
            with open(log_path, 'r') as f:
                log_data = json.load(f)
            
            metadata = log_data['metadata']
            training_data = log_data['training_data']
            
            if not training_data:
                continue
            
            model_name = metadata['model_name']
            color = colors[i % len(colors)]
            
            steps = [d['step'] for d in training_data]
            losses = [d['loss'] for d in training_data]
            lrs = [d['lr'] for d in training_data]
            tokens_per_sec = [d['tokens_per_sec'] for d in training_data]
            gpu_memory = [d['gpu_memory_gb'] for d in training_data]
            
            # Plot comparisons
            ax1.plot(steps, losses, color=color, linewidth=2, label=model_name, alpha=0.8)
            ax2.plot(steps, lrs, color=color, linewidth=2, label=model_name, alpha=0.8)
            ax3.plot(steps, tokens_per_sec, color=color, linewidth=2, label=model_name, alpha=0.8)
            ax4.plot(steps, gpu_memory, color=color, linewidth=2, label=model_name, alpha=0.8)
        
        # Configure axes
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Comparison')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Comparison')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        ax3.set_xlabel('Steps')
        ax3.set_ylabel('Tokens per Second')
        ax3.set_title('Throughput Comparison')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        ax4.set_xlabel('Steps')
        ax4.set_ylabel('GPU Memory (GB)')
        ax4.set_title('Memory Usage Comparison')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path


def create_plots_for_latest_run(model_name: str) -> Optional[str]:
    """Create plots for the latest training run of a model."""
    
    log_dir = "training_logs"
    if not os.path.exists(log_dir):
        return None
    
    # Find latest log file for model (exclude summary files)
    log_files = [f for f in os.listdir(log_dir)
                 if f.startswith(f"{model_name}_run_")
                 and f.endswith('.json')
                 and '_summary' not in f]
    if not log_files:
        return None

    # Get latest run
    log_files.sort(key=lambda x: int(x.split('_run_')[1].split('.')[0]), reverse=True)
    latest_log = os.path.join(log_dir, log_files[0])
    
    # Create plots
    plotter = TrainingPlotter()
    plot_path = plotter.create_training_plots(latest_log)
    
    return plot_path


def create_plots_from_log(log_path: str) -> str:
    """Create plots from specific log file."""
    plotter = TrainingPlotter()
    return plotter.create_training_plots(log_path)
