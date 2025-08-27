@echo off
echo ========================================
echo 🔧 Development Training (1B tokens)
echo ========================================
echo.
echo This will run a 1B token training
echo for development and testing (~20 hours)
echo.

call conda activate llm_cuda
python simple_training.py 1.0

pause
