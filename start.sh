#!/bin/bash
# Render startup script for StockBot VF

echo "[STARTUP] Starting StockBot VF..."

# Run database migrations
echo "[STARTUP] Running database migrations..."
alembic upgrade head

# Start the server
echo "[STARTUP] Starting Uvicorn server..."
exec uvicorn stockbot.api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2
