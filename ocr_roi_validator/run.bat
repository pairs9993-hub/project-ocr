@echo off
setlocal
cd /d %~dp0

REM --- Verify the app has the exact-OCR-input capture support -----------------
REM A stale .pyc or an out-of-date checkout silently produces failure
REM diagnostics without the recorded recognizer input, which cannot be used for
REM candidate evaluation. Check before launching rather than after capturing.
findstr /C:"exact_recorded_ocr_input" ocr_roi_validator\gui.py >nul 2>&1
if errorlevel 1 (
  echo.
  echo [ERROR] This checkout does not support exact OCR input capture.
  echo         Failure diagnostics would be saved WITHOUT roi_ocr_input.png.
  echo.
  echo         Update the checkout first:
  echo             git checkout fr-specialist-router-review
  echo             git pull
  echo.
  pause
  exit /b 1
)

REM Drop stale bytecode so an old gui.pyc cannot shadow the updated source.
if exist ocr_roi_validator\__pycache__ rd /s /q ocr_roi_validator\__pycache__
if exist __pycache__ rd /s /q __pycache__

set INSTALL_DEPS=0
if not exist .venv\Scripts\python.exe (
  py -3 -m venv .venv
  set INSTALL_DEPS=1
)

call .venv\Scripts\activate
if "%INSTALL_DEPS%"=="1" (
  python -m pip install --upgrade pip
  pip install -r requirements.txt
)

set RAPID_PACKAGE=artifacts\my_rapid_package

if not exist "%RAPID_PACKAGE%\manifest.json" (
  echo [INFO] Building trained Rapid model package...
  python scripts\build_model_package.py ^
    --output "%RAPID_PACKAGE%" ^
    --detector "..\artifacts\models\real_ui_company_pseudo_rec\det.onnx" ^
    --rec-en-es "..\artifacts\models\real_ui_company_pseudo_rec\rec.onnx" ^
    --rec-fr "..\artifacts\models\real_ui_fr_rec_v2_hard\rec.onnx" ^
    --rec-zh "..\artifacts\models\real_ui_1m\rec.onnx" ^
    --dict "..\artifacts\models\real_ui_company_pseudo_rec\ppocr_keys.txt"
)

if exist "%RAPID_PACKAGE%\manifest.json" (
  echo [INFO] Launching app with trained Rapid package: %RAPID_PACKAGE%
  python main.py --backend rapid --model-package "%RAPID_PACKAGE%"
) else (
  echo [WARN] Trained package unavailable. Falling back to Rapid default models.
  python main.py --backend rapid --rapid-default
)
