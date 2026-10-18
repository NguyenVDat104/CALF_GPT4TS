@echo off
title CALF Forecasting Server & Cloudflare HTTPS Tunnel
color 0A

echo ======================================================================
echo          STARTING CALF FORECASTING SERVER & CLOUDFLARE HTTPS
echo ======================================================================

cd /d "%~dp0"

echo [1/2] Starting Python FastAPI Backend on port 8443...
start "CALF-Backend" /min python server.py

timeout /t 3 /nobreak > nul

echo [2/2] Starting Cloudflare Public HTTPS Tunnel...
echo.
echo Tunnel will generate a secure public HTTPS domain below:
echo ----------------------------------------------------------------------
.\cloudflared.exe tunnel --url https://localhost:8443 --no-tls-verify

pause
