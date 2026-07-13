@echo off
title AI Fire Sentinel - Desktop GUI
echo ====================================================================
echo   🚨 KICH HOAT HE THONG CANH BAO CHAY THONG MINH (DESKTOP GUI) 🚨
echo ====================================================================
echo.
echo [*] Dang khoi dong bang moi truong ao (.venv)...
.venv\Scripts\python.exe app.py
if %errorlevel% neq 0 (
    echo.
    echo [X] Co loi xay ra khi chay chuong trinh.
    pause
)
