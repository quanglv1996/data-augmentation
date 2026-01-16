@echo off
echo =====================================
echo Data Augmentation Web App
echo =====================================
echo.

echo [1/3] Building Docker image...
docker-compose build

echo.
echo [2/3] Starting containers...
docker-compose up -d

echo.
echo [3/3] Checking status...
docker-compose ps

echo.
echo =====================================
echo Application started successfully!
echo Access the app at: http://localhost:222
echo =====================================
echo.
echo Commands:
echo - Stop: docker-compose down
echo - Logs: docker-compose logs -f
echo - Restart: docker-compose restart
echo.
pause
