@echo off
echo 📈 Starting Competitive Portfolio Dashboard...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

REM Check if pip is available
pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ pip is not available
    echo Please ensure pip is installed
    pause
    exit /b 1
)

REM Install dependencies if requirements.txt exists
if exist requirements.txt (
    echo 📦 Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Failed to install dependencies
        pause
        exit /b 1
    )
) else (
    echo ⚠️  requirements.txt not found, skipping dependency installation
)

REM Check if Streamlit is installed
streamlit --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Streamlit is not installed
    echo Installing Streamlit...
    pip install streamlit
)

echo.
echo 🚀 Launching dashboard...
echo Dashboard will open in your default browser
echo Press Ctrl+C to stop the server
echo.

REM Launch the Streamlit app
streamlit run app.py

pause