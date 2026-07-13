@echo off
title AI Fire Sentinel - Web Dashboard
echo ====================================================================
echo   🚨 KICH HOAT HE THONG CANH BAO CHAY THONG MINH (WEB DASHBOARD) 🚨
echo ====================================================================
echo.
echo [*] Dang khoi dong server Streamlit bang moi truong ao (.venv)...
.venv\Scripts\python.exe -m streamlit run web_app.py
if %errorlevel% neq 0 (
    echo.
    echo [X] Co loi xay ra khi chay chuong trinh.
    pause
)
