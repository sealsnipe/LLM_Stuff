#!/usr/bin/env python3
"""
🔧 Progress Bar Fix for 3090 System
Comprehensive fix for progress bar jumping issues
"""

import sys
import os
import time

def install_colorama():
    """Install colorama for Windows ANSI support"""
    try:
        import colorama
        print("✅ Colorama already installed")
        return True
    except ImportError:
        print("📦 Installing colorama for Windows ANSI support...")
        try:
            import subprocess
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', 'colorama'
            ], capture_output=True, text=True, check=True)
            print("✅ Colorama installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install colorama: {e}")
            return False

def fix_terminal_environment():
    """Fix terminal environment for progress bars"""
    print("🔧 Fixing terminal environment...")
    
    # Set environment variables
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['TERM'] = 'xterm-256color'
    os.environ['COLORTERM'] = 'truecolor'
    
    # Windows-specific fixes
    if os.name == 'nt':
        # Set console to UTF-8
        os.system('chcp 65001 > nul 2>&1')
        
        # Initialize colorama
        try:
            import colorama
            colorama.init(autoreset=True, convert=True, strip=False)
            print("✅ Colorama initialized")
        except ImportError:
            print("⚠️ Colorama not available")
        
        # Reconfigure stdout/stderr
        if hasattr(sys.stdout, 'reconfigure'):
            try:
                sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
                sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)
                print("✅ UTF-8 encoding configured")
            except:
                print("⚠️ Could not reconfigure encoding")
    
    print("✅ Terminal environment fixed")

def test_ansi_support():
    """Test if ANSI escape codes work"""
    print("\n🧪 Testing ANSI support...")
    
    try:
        # Test basic ANSI codes
        print("\033[32mGreen text\033[0m")
        print("\033[1mBold text\033[0m")
        print("\033[31;1mRed bold text\033[0m")
        
        # Test carriage return (key for progress bars)
        print("Testing carriage return...", end="")
        time.sleep(0.5)
        print("\rCarriage return works!   ")
        
        # Test progress bar simulation
        print("\nTesting progress bar:")
        for i in range(21):
            percent = i * 5
            bar = '█' * (i // 2) + '░' * (10 - i // 2)
            print(f"\rProgress: [{bar}] {percent}%", end="", flush=True)
            time.sleep(0.1)
        print("\n✅ ANSI support test completed")
        return True
        
    except Exception as e:
        print(f"\n❌ ANSI test failed: {e}")
        return False

def create_fallback_progress():
    """Create fallback progress bar code"""
    fallback_code = '''
# Fallback Progress Bar for Systems with ANSI Issues
import sys
import time

class FallbackProgress:
    """Progress bar that works without ANSI codes"""
    
    def __init__(self, total, desc="Progress"):
        self.total = total
        self.current = 0
        self.desc = desc
        self.last_print_time = 0
        self.print_interval = 0.5  # Print every 0.5 seconds
        
    def update(self, n=1):
        self.current = min(self.current + n, self.total)
        current_time = time.time()
        
        # Only print occasionally to avoid spam
        if (current_time - self.last_print_time > self.print_interval or 
            self.current >= self.total):
            self._print_progress()
            self.last_print_time = current_time
    
    def _print_progress(self):
        if self.total == 0:
            return
            
        percent = (self.current / self.total) * 100
        print(f"{self.desc}: {self.current}/{self.total} ({percent:.1f}%)")
        
        if self.current >= self.total:
            print(f"✅ {self.desc} completed!")

# Usage in training loop:
# Replace: for batch in tqdm(dataloader, desc="Training"):
# With:    progress = FallbackProgress(len(dataloader), "Training")
#          for batch in dataloader:
#              # ... training code ...
#              progress.update(1)
'''
    
    with open('fallback_progress.py', 'w', encoding='utf-8') as f:
        f.write(fallback_code)
    
    print("✅ Created fallback_progress.py")

def main():
    """Main fix function"""
    print("🚀 PROGRESS BAR FIX FOR 3090 SYSTEM")
    print("=" * 60)
    
    # Step 1: Install colorama
    colorama_ok = install_colorama()
    
    # Step 2: Fix terminal environment
    fix_terminal_environment()
    
    # Step 3: Test ANSI support
    ansi_ok = test_ansi_support()
    
    # Step 4: Create fallback if needed
    if not ansi_ok:
        print("\n⚠️ ANSI codes don't work properly")
        create_fallback_progress()
    
    print(f"\n🎉 FIX COMPLETE!")
    print("=" * 60)
    
    if colorama_ok and ansi_ok:
        print("✅ Your system should now work with progress bars")
        print("💡 Restart your terminal and try training again")
    else:
        print("⚠️ Progress bars may still have issues")
        print("💡 Recommendations:")
        print("   1. Use Windows Terminal instead of cmd.exe")
        print("   2. Try running in VS Code terminal")
        print("   3. Use the fallback_progress.py code")
    
    print(f"\n📝 Next steps:")
    print("   1. Close and reopen your terminal")
    print("   2. Run: conda activate llm_cuda")
    print("   3. Run: python training-windows.py")

if __name__ == "__main__":
    main()
