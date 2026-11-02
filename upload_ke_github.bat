@echo off
title Upload Proyek ke GitHub
cd /d "%~dp0"
echo ========================================================
echo        Upload Proyek ke GitHub (Untuk Online 24/7)
echo ========================================================
echo.
echo Pastikan Anda sudah membuat repository baru di github.com
echo Contoh link repo: https://github.com/username/nama-repo.git
echo.
set /p REPO_URL="Masukkan Link Repository GitHub Anda: "

if "%REPO_URL%"=="" (
    echo Error: Link tidak boleh kosong!
    pause
    exit /b
)

echo.
echo Mengatur remote GitHub...
git remote remove origin 2>nul
git remote add origin %REPO_URL%
git branch -M main

echo Mengunggah file ke GitHub...
git push -u origin main

echo.
echo ========================================================
echo Selesai! Jika berhasil, buka share.streamlit.io untuk deploy online.
echo ========================================================
pause
