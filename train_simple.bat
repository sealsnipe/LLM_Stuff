@echo off
echo ========================================
echo 🚀 Simple LLM Training
echo ========================================
echo.

REM Check if argument provided
if "%1"=="" (
    echo ❌ Usage: train_simple.bat ^<tokens_in_billions^>
    echo.
    echo Examples:
    echo   train_simple.bat 0.1    # 100M tokens ^(quick test^)
    echo   train_simple.bat 1.0    # 1B tokens ^(development^)
    echo   train_simple.bat 5.0    # 5B tokens ^(medium^)
    echo   train_simple.bat 18.5   # 18.5B tokens ^(minimum viable^)
    echo   train_simple.bat 46.0   # 46B tokens ^(optimal^)
    echo.
    pause
    exit /b 1
)

REM Activate conda environment
echo 🔧 Activating conda environment...
call conda activate llm_cuda
if errorlevel 1 (
    echo ❌ Failed to activate conda environment 'llm_cuda'
    echo Please make sure the environment exists and try again.
    pause
    exit /b 1
)

echo ✅ Environment activated!
echo.

REM Run training
echo 🚀 Starting training with %1B tokens...
echo.
python simple_training.py %1

echo.
echo 🏁 Training completed!
echo.
pause
