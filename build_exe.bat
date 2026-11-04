@echo off
setlocal
cd /d "%~dp0"
title PyAgent - Build EXE
echo Building PyAgent.exe. Please keep this window open.
echo Python 3.10+ (64-bit) is required. Missing build packages will be installed automatically.
echo.
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\build_exe.ps1" -InstallDependencies %*
set "buildResult=%ERRORLEVEL%"
echo.
if "%buildResult%"=="0" (
    echo SUCCESS: %~dp0dist\PyAgent.exe
) else (
    echo FAILED: Read the error above. Fix the problem and double-click this script again.
)
echo.
pause
exit /b %buildResult%
