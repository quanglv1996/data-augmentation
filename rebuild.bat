@echo off
echo =====================================
echo Rebuilding Docker Container
echo =====================================
echo.

echo [1/3] Stopping and removing old containers...
docker-compose down

echo.
echo [2/3] Rebuilding image...
docker-compose build --no-cache

echo.
echo [3/3] Starting new containers...
docker-compose up -d

echo.
echo =====================================
echo Done! Check status:
docker-compose ps
echo.
echo View logs:
echo docker-compose logs -f
echo.
echo Access: http://localhost:2222
echo =====================================
pause
