@echo off
echo ========================================
echo 🧪 Quick Test Training (100M tokens)
echo ========================================
echo.
echo This will run a quick 100M token training
echo to verify everything works (~2 hours)
echo.

call conda activate llm_cuda
python simple_training.py 0.1

pause
