@echo off
title AI Facial Health Screening - Mode Online & HP
cd /d "%~dp0"
cls
echo ====================================================================
echo      AI Facial Health Screening Dashboard (Mode Online dan HP)
echo ====================================================================
echo.
echo Sedang menyalakan server aplikasi dan membuat link publik HTTPS...
echo Mohon tunggu sekitar 5 detik...
echo.

:: Jalankan Streamlit di latar belakang
start /B python -m streamlit run app.py --server.headless=true --server.address=127.0.0.1 >nul 2>&1

:: Tunggu 3 detik agar streamlit siap
timeout /t 3 /nobreak >nul

echo.
echo ====================================================================
echo   LINK PUBLIK ONLINE ANDA AKAN MUNCUL DI BAWAH INI:
echo   (Bisa dibuka dari HP mana saja dengan kuota/paket data 4G/5G)
echo ====================================================================
echo.

:: Jalankan Cloudflare Tunnel untuk menghasilkan link publik HTTPS
cloudflared.exe tunnel --url http://127.0.0.1:8501
pause
