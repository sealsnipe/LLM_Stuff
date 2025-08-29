#!/usr/bin/env python3
"""
🔧 Progress Bar Fix for Windows Systems
Diagnoses and fixes progress bar issues on different terminals
"""

import sys
import os

def diagnose_terminal():
    """Diagnose terminal capabilities"""
    print("🔍 TERMINAL DIAGNOSIS")
    print("=" * 50)
    
    # Check Python version
    print(f"Python Version: {sys.version}")
    
    # Check terminal type
    term = os.environ.get('TERM', 'unknown')
    print(f"TERM: {term}")
    
    # Check if running in various environments
    environments = {
        'VS Code': 'VSCODE_PID' in os.environ,
        'PyCharm': 'PYCHARM_HOSTED' in os.environ,
        'Jupyter': 'JPY_PARENT_PID' in os.environ,
        'Windows Terminal': 'WT_SESSION' in os.environ,
        'PowerShell': 'PSModulePath' in os.environ,
    }
    
    print("\nEnvironment Detection:")
    for env, detected in environments.items():
        status = "✅" if detected else "❌"
        print(f"  {status} {env}: {detected}")
    
    # Check ANSI support
    print(f"\nANSI Support Test:")
    try:
        # Test ANSI escape codes
        print("\033[32mGreen text test\033[0m")
        print("\033[1mBold text test\033[0m")
        print("\033[2K\rCarriage return test", end="")
        print(" - Success!")
        ansi_works = True
    except:
        ansi_works = False
    
    print(f"ANSI Codes: {'✅ Working' if ansi_works else '❌ Not working'}")
    
    return ansi_works

def check_dependencies():
    """Check required dependencies"""
    print("\n📦 DEPENDENCY CHECK")
    print("=" * 50)
    
    required_packages = {
        'tqdm': 'Progress bars',
        'colorama': 'Windows ANSI support',
        'rich': 'Advanced terminal formatting (optional)'
    }
    
    missing = []
    
    for package, description in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {package}: Installed - {description}")
        except ImportError:
            print(f"❌ {package}: Missing - {description}")
            missing.append(package)
    
    return missing

def install_fixes():
    """Install missing dependencies"""
    missing = check_dependencies()
    
    if missing:
        print(f"\n🔧 INSTALLING FIXES")
        print("=" * 50)
        
        for package in missing:
            print(f"Installing {package}...")
            try:
                import subprocess
                result = subprocess.run([
                    sys.executable, '-m', 'pip', 'install', package
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"✅ {package} installed successfully")
                else:
                    print(f"❌ Failed to install {package}: {result.stderr}")
            except Exception as e:
                print(f"❌ Error installing {package}: {e}")
    else:
        print("\n✅ All dependencies already installed!")

def test_progress_bar():
    """Test progress bar functionality"""
    print(f"\n🧪 PROGRESS BAR TEST")
    print("=" * 50)
    
    try:
        from tqdm import tqdm
        import time
        
        print("Testing tqdm progress bar...")
        
        # Test 1: Basic progress bar
        print("\nTest 1: Basic Progress Bar")
        for i in tqdm(range(10), desc="Basic Test"):
            time.sleep(0.1)
        
        # Test 2: Manual update
        print("\nTest 2: Manual Update")
        pbar = tqdm(total=10, desc="Manual Test")
        for i in range(10):
            time.sleep(0.1)
            pbar.update(1)
        pbar.close()
        
        # Test 3: Nested progress bars
        print("\nTest 3: Nested Progress Bars")
        for i in tqdm(range(3), desc="Outer"):
            for j in tqdm(range(5), desc="Inner", leave=False):
                time.sleep(0.05)
        
        print("✅ Progress bar tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Progress bar test failed: {e}")
        return False

def apply_terminal_fixes():
    """Apply terminal-specific fixes"""
    print(f"\n⚙️ APPLYING TERMINAL FIXES")
    print("=" * 50)
    
    # Fix 1: Initialize colorama for Windows
    try:
        import colorama
        colorama.init(autoreset=True)
        print("✅ Colorama initialized for Windows ANSI support")
    except ImportError:
        print("❌ Colorama not available")
    
    # Fix 2: Set environment variables
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['TERM'] = 'xterm-256color'
    print("✅ Environment variables set")
    
    # Fix 3: Force UTF-8 encoding
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
            print("✅ UTF-8 encoding configured")
        except:
            print("⚠️ Could not reconfigure encoding")

def create_training_fix():
    """Create a fixed version of the training script"""
    print(f"\n🔧 CREATING TRAINING FIX")
    print("=" * 50)
    
    fix_code = '''
# Add this to the top of training-windows.py
import sys
import os

# Windows Terminal Fixes
def fix_windows_terminal():
    """Fix Windows terminal issues"""
    try:
        # Initialize colorama for ANSI support
        import colorama
        colorama.init(autoreset=True)
    except ImportError:
        pass
    
    # Set encoding
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    # Reconfigure stdout/stderr if possible
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except:
            pass

# Call at the beginning of main()
fix_windows_terminal()

# Alternative: Use simple progress instead of tqdm
def simple_progress(current, total, desc="Progress"):
    """Simple progress without ANSI codes"""
    percent = (current / total) * 100
    bar_length = 50
    filled = int(bar_length * current / total)
    bar = '█' * filled + '░' * (bar_length - filled)
    print(f"\\r{desc}: [{bar}] {percent:.1f}% ({current}/{total})", end="", flush=True)
    if current == total:
        print()  # New line when complete
'''
    
    with open('terminal_fix.py', 'w', encoding='utf-8') as f:
        f.write(fix_code)
    
    print("✅ Created terminal_fix.py with fixes")
    print("📝 Add the fix_windows_terminal() call to your training script")

def main():
    """Main diagnosis and fix function"""
    print("🚀 PROGRESS BAR DIAGNOSTIC & FIX TOOL")
    print("=" * 60)
    
    # Step 1: Diagnose terminal
    ansi_works = diagnose_terminal()
    
    # Step 2: Check dependencies
    missing = check_dependencies()
    
    # Step 3: Install fixes if needed
    if missing or not ansi_works:
        install_fixes()
        apply_terminal_fixes()
    
    # Step 4: Test progress bar
    test_progress_bar()
    
    # Step 5: Create training fix
    create_training_fix()
    
    print(f"\n🎉 DIAGNOSIS COMPLETE!")
    print("=" * 60)
    
    if ansi_works and not missing:
        print("✅ Your system should work fine with progress bars")
    else:
        print("⚠️ Your system may have progress bar issues")
        print("💡 Recommendations:")
        print("   1. Use Windows Terminal instead of cmd.exe")
        print("   2. Install colorama: pip install colorama")
        print("   3. Use the terminal_fix.py code in your training script")

if __name__ == "__main__":
    main()
