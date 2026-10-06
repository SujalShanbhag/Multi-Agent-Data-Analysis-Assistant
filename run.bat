@echo off
title Multi-Agent Data Analysis Assistant
echo ========================================================
echo   Starting Multi-Agent Data Analysis Assistant
echo ========================================================
echo.

IF EXIST .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

python -m backend.main
pause
