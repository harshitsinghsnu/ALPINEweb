@echo off
REM ALPINE Web Application - Quick Start Script (Windows)

echo 🧬 ALPINE - Protein Binding Predictor
echo ======================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo 📦 Creating virtual environment...
    python -m venv venv
    echo ✅ Virtual environment created
) else (
    echo ✅ Virtual environment found
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo 📥 Installing dependencies...
python -m pip install -q --upgrade pip
pip install -q -r requirements.txt
echo ✅ Dependencies installed

REM Check for GPU
echo.
echo 🔍 Checking for GPU...
python -c "import torch; print('GPU Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo.
echo 🚀 Starting ALPINE Web Application...
echo 📍 Application will be available at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo.

REM Run Streamlit
streamlit run alpine_web_app.py