@echo off
echo =====================================
echo Testing Data Augmentation Web App
echo =====================================
echo.

echo Checking Python...
python --version
if errorlevel 1 (
    echo ERROR: Python not found!
    pause
    exit /b 1
)

echo.
echo Installing dependencies...
cd webapp
pip install -r requirements.txt

echo.
echo Starting application...
echo.
echo =====================================
echo Application will start on port 222
echo Access: http://localhost:222
echo Press Ctrl+C to stop
echo =====================================
echo.

python app.py

pause
