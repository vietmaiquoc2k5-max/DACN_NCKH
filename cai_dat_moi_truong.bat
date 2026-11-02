@echo off
title Cai Dat Moi Truong - Phat Hien Chay
echo ====================================================================
echo   CAI DAT MOI TRUONG VA THU VIEN CHO DU AN CANH BAO CHAY PCCC
echo ====================================================================
echo.

REM Kiem tra Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [X] May tinh chua cai dat Python hoac chua them Python vao PATH!
    echo [!] Vui long tai va cai dat Python 3.9 - 3.11 tu: https://www.python.org/downloads/
    echo [!] Khi cai dat hay tich chon "Add python.exe to PATH".
    pause
    exit /b 1
)

echo [*] Phat hien Python hop le:
python --version
echo.

REM Tao moi truong ao neu chua co
if not exist ".venv" (
    echo [*] Dang tao moi truong ao .venv...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [X] Loi khi tao moi truong ao .venv!
        pause
        exit /b 1
    )
    echo [OK] Tao .venv thanh cong.
) else (
    echo [*] Da co san thu muc .venv.
)
echo.

REM Nang cap pip va cai dat thu vien
echo [*] Dang nang cap pip...
.venv\Scripts\python.exe -m pip install --upgrade pip

echo.
echo [*] Dang cai dat cac thu vien can thiet tu requirements.txt...
.venv\Scripts\pip.exe install -r requirements.txt
if %errorlevel% neq 0 (
    echo [X] Co loi xay ra trong qua trinh cai dat thu vien!
    pause
    exit /b 1
)

echo.
echo ====================================================================
echo   [OK] CAI DAT HOAN TAT 100%!
echo   Ban co the chay ung dung bang 1 trong 2 cach:
echo     1. chay_giao_dien_web.bat     (Giao dien Web 9 Camera)
echo     2. chay_giao_dien_desktop.bat (Giao dien Desktop)
echo ====================================================================
pause

