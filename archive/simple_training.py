#!/usr/bin/env python3
"""
🚀 SIMPLE TRAINING SCRIPT
Einfaches Training mit token-basierter Konfiguration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_training_config import set_training_target, update_model_architecture, show_current_config

# Import training function directly
import subprocess

def main():
    """Main training function with simple configuration"""
    
    print("🚀 SIMPLE LLM TRAINING")
    print("=" * 60)
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("❌ Usage: python simple_training.py <tokens_in_billions>")
        print()
        print("Examples:")
        print("  python simple_training.py 0.1    # 100M tokens (quick test)")
        print("  python simple_training.py 1.0    # 1B tokens (development)")
        print("  python simple_training.py 5.0    # 5B tokens (medium)")
        print("  python simple_training.py 18.5   # 18.5B tokens (minimum viable)")
        print("  python simple_training.py 46.0   # 46B tokens (optimal)")
        print()
        
        # Show recommendations
        from simple_training_config import get_recommended_targets
        get_recommended_targets()
        return
    
    try:
        target_tokens_b = float(sys.argv[1])
    except ValueError:
        print("❌ Error: Please provide a valid number for tokens in billions")
        return
    
    # Validate range
    if target_tokens_b < 0.01:
        print("❌ Error: Minimum 0.01B (10M) tokens required")
        return
    elif target_tokens_b > 100:
        print("❌ Error: Maximum 100B tokens supported")
        return
    
    print(f"🎯 Training Target: {target_tokens_b:.1f}B tokens")
    print()
    
    # Optional: Update model architecture for better performance
    if target_tokens_b >= 5.0:  # For serious training runs
        print("🏗️  Updating model architecture for better performance...")
        update_model_architecture(hidden_size=1536, num_layers=24)
        print()
    
    # Set training target
    config = set_training_target(target_tokens_b)
    print()
    
    # Show final configuration
    show_current_config()
    print()
    
    # Confirm training
    if target_tokens_b >= 5.0:
        estimated_days = config['estimated_hours'] / 24
        print(f"⚠️  This will take approximately {estimated_days:.1f} days to complete.")
        response = input("Continue? (y/N): ").strip().lower()
        if response != 'y':
            print("Training cancelled.")
            return
    
    print("🚀 Starting training...")
    print("=" * 60)
    
    # Start training with automatic dataset sizing
    try:
        # Run the training script directly
        print("🚀 Launching training-windows.py...")
        result = subprocess.run([
            "python", "training-windows.py"
        ], capture_output=False, text=True)

        if result.returncode == 0:
            print("✅ Training completed successfully!")
        else:
            print(f"❌ Training failed with return code: {result.returncode}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

def quick_test():
    """Quick test with 100M tokens"""
    print("🧪 QUICK TEST MODE")
    print("Running 100M token training for testing...")
    
    set_training_target(0.1)  # 100M tokens
    
    try:
        subprocess.run(["python", "training-windows.py"], capture_output=False)
    except Exception as e:
        print(f"❌ Quick test failed: {e}")

def development_run():
    """Development run with 1B tokens"""
    print("🔧 DEVELOPMENT MODE")
    print("Running 1B token training for development...")

    set_training_target(1.0)  # 1B tokens

    try:
        subprocess.run(["python", "training-windows.py"], capture_output=False)
    except Exception as e:
        print(f"❌ Development run failed: {e}")

def production_run():
    """Production run with 18.5B tokens (minimum viable)"""
    print("🏭 PRODUCTION MODE")
    print("Running 18.5B token training for production model...")

    # Update architecture for production
    update_model_architecture(hidden_size=1536, num_layers=24)
    set_training_target(18.5)  # 18.5B tokens

    print("⚠️  This will take approximately 2 weeks to complete.")
    response = input("Continue? (y/N): ").strip().lower()
    if response != 'y':
        print("Training cancelled.")
        return

    try:
        subprocess.run(["python", "training-windows.py"], capture_output=False)
    except Exception as e:
        print(f"❌ Production run failed: {e}")

if __name__ == "__main__":
    main()
