@echo off
echo ============================================================
echo DenseNet Resumed Training with Monitoring
echo ============================================================

cd /d "%~dp0"
echo Current directory: %CD%

echo.
echo Step 1: Testing imports and environment...
python test_resume_imports.py
if %ERRORLEVEL% neq 0 (
    echo ERROR: Import test failed!
    pause
    exit /b 1
)

echo.
echo Step 2: Starting resumed training from epoch 51...
echo Command: python resume_densenet_training.py --checkpoint results_densenet_20250906_230232\best_densenet.pth --max_epochs 100 --patience 10 --learning_rate 0.0005

python resume_densenet_training.py --checkpoint results_densenet_20250906_230232\best_densenet.pth --max_epochs 100 --patience 10 --learning_rate 0.0005

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Training failed with error code %ERRORLEVEL%
    echo Check the error messages above.
    pause
    exit /b 1
) else (
    echo.
    echo SUCCESS: Training completed successfully!
    echo Check the results directory for outputs.
)

echo.
echo Training process completed.
pause
