@echo off
echo ========================================
echo 🤖 LLM Conversation Test
echo ========================================
echo.

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

REM Change to project directory
cd /d "%~dp0\.."

REM Run the conversation test
echo 🚀 Starting conversation test...
echo.
python tests/test_conversation.py

echo.
echo 🏁 Test completed!
echo.
pause
