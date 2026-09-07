@echo off
setlocal
cd /d "%~dp0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Install-MiniMax-H3-Telegram-Autostart.ps1"
if errorlevel 1 (
  echo.
  echo Autostart installation failed. Check the message above.
  pause
  exit /b 1
)

echo.
echo Autostart installation completed.
pause
endlocal
