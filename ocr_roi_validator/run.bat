@echo off
setlocal
cd /d %~dp0

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
