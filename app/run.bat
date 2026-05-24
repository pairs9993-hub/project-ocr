@echo off
REM ============================================================
REM OCR Validation App launcher (Windows)
REM
REM First run (one-time):
REM   1) install Python 3.10 or 3.11 (64-bit) from python.org
REM   2) double-click this file. It will create a local venv and
REM      install dependencies on first launch, then open the app
REM      in your default browser.
REM ============================================================

setlocal
cd /d "%~dp0"

set VENV_DIR=.venv_app
set PY_LAUNCHER=py
set MODEL_DIR=models
set EXPORT_MODEL_DIR=..\artifacts\models\real_ui_company_pseudo_rec

REM ---- sync latest exported project models into the app defaults ----
if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%" >nul 2>&1
set HAS_LATEST_MODELS=0
if exist "%EXPORT_MODEL_DIR%\rec.onnx" if exist "%EXPORT_MODEL_DIR%\det.onnx" if exist "%EXPORT_MODEL_DIR%\ppocr_keys.txt" set HAS_LATEST_MODELS=1
if "%HAS_LATEST_MODELS%"=="1" (
    echo [model] Syncing latest real_ui_company_pseudo_rec models into app\models...
    copy /Y "%EXPORT_MODEL_DIR%\rec.onnx" "%MODEL_DIR%\rec_v0.onnx" >nul || goto model_copy_failed
    copy /Y "%EXPORT_MODEL_DIR%\det.onnx" "%MODEL_DIR%\det_v0.onnx" >nul || goto model_copy_failed
    copy /Y "%EXPORT_MODEL_DIR%\ppocr_keys.txt" "%MODEL_DIR%\ppocr_keys.txt" >nul || goto model_copy_failed
) else (
    echo [model] Latest exported models not found; using bundled app\models files.
)

REM ---- silence Streamlit's first-run email prompt + telemetry ----
if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit" >nul 2>&1
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
    > "%USERPROFILE%\.streamlit\credentials.toml" echo [general]
    >> "%USERPROFILE%\.streamlit\credentials.toml" echo email = ""
)
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [setup] Creating virtual environment...
    %PY_LAUNCHER% -3 -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [setup] 'py -3' failed; trying 'python'...
        python -m venv "%VENV_DIR%"
    )
    if errorlevel 1 (
        echo [error] Could not create virtual environment.
        echo         Install Python 3.10 or 3.11 from python.org and retry.
        pause
        exit /b 1
    )
    "%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip
    "%VENV_DIR%\Scripts\python.exe" -m pip install -r requirements.txt
)

if /I "%~1"=="--prepare-only" (
    echo [check] prepare-only complete.
    exit /b 0
)

echo.
echo [run] Starting OCR Validation App...
echo [run] If the browser does not open, visit http://localhost:8501
echo.
"%VENV_DIR%\Scripts\python.exe" -m streamlit run app.py --server.headless=false --browser.gatherUsageStats=false

pause
endlocal
exit /b 0

:model_copy_failed
echo [error] Failed to copy exported model files into app\models.
pause
exit /b 1
