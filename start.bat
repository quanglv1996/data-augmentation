@echo off
echo =====================================
echo Data Augmentation Web App
echo =====================================
echo.

echo [1/4] Checking Docker...
docker --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker not found! Please install Docker Desktop.
    pause
    exit /b 1
)

echo [2/4] Building Docker image...
docker-compose build

if errorlevel 1 (
    echo ERROR: Build failed!
    pause
    exit /b 1
)

echo.
echo [3/4] Starting containers...
docker-compose up -d

if errorlevel 1 (
    echo ERROR: Failed to start containers!
    pause
    exit /b 1
)

echo.
echo [4/4] Checking status...
docker-compose ps

echo.
echo =====================================
echo Application started successfully!
echo Access the app at: http://localhost:2222
echo =====================================
echo.
echo Commands:
echo - Stop: docker-compose down
echo - Logs: docker-compose logs -f
echo - Restart: docker-compose restart
echo.
pause
