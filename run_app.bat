@echo off
title AI Facial Health Screening Dashboard
cd /d "%~dp0"
echo ================================================================
echo       AI Facial Health Screening Dashboard (Triage Wajah)
echo ================================================================
echo.
echo [1] Alamat untuk dibuka di LAPTOP ini : http://localhost:8501
echo [2] Alamat untuk dibuka di HP (Wi-Fi) : http://192.168.1.4:8501
echo.
echo Membuka browser laptop otomatis...
echo ================================================================
echo.
start http://localhost:8501
python -m streamlit run app.py --server.headless=true --server.address=0.0.0.0
pause
