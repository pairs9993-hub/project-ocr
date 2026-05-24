@echo off
setlocal
cd /d "%~dp0"

if not exist "app\run.bat" (
    echo [error] app\run.bat was not found.
    echo         Make sure you are running this file from the project root.
    pause
    exit /b 1
)

call "app\run.bat" %*
exit /b %ERRORLEVEL%