@echo off
setlocal

echo ======================================================================
echo HUAN LUYEN BO PHAT HIEN MT2 TREN GPU NVIDIA RTX 3050
echo ======================================================================

set PYTHON_GPU=C:\Users\vietm\AppData\Local\Programs\Python\Python39\python.exe

if not exist "%PYTHON_GPU%" (
    echo [ERROR] Khong tim thay Python 3.9 tai: %PYTHON_GPU%
    pause
    exit /b 1
)

echo Dang su dung moi truong: %PYTHON_GPU%
echo Khoi chay pipeline huan luyen scripts\train_benchmarks.py...

"%PYTHON_GPU%" scripts\train_benchmarks.py --models yolov8n yolo11n --data data_dfire.yaml --epochs 50 --batch 8 --seeds 42 123 456 --device 0 %*

pause
