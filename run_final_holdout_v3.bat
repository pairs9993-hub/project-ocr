@echo off
REM Single-writer generation of final_holdout_v3.
REM Detached from any Claude session so it survives teardown. Resumes from the
REM existing checkpoint; recipe and seed are never changed.
cd /d %~dp0\ocr_roi_validator
.venv\Scripts\python.exe scripts\build_final_holdout_v3.py ^
  --package artifacts\my_rapid_package ^
  --out-dir ..\artifacts\accent_ds\final_holdout_v3 ^
  --recipe ..\artifacts\accent_ds\final_holdout_v3_recipe.json ^
  --max-renderings 60000 ^
  --progress-every 500 ^
  --other-split ..\artifacts\accent_ds\train_v3 ^
  --other-split ..\artifacts\accent_ds\validation_v3 ^
  --other-split ..\artifacts\accent_ds\diagnostic_v1 ^
  --other-split ..\artifacts\accent_ds\diagnostic_v2
