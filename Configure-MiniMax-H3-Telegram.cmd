@echo off
setlocal
cd /d "%~dp0"

echo This will securely configure the Telegram Bot.
echo First revoke the old token in @BotFather and create a new one.
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Configure-MiniMax-H3-Telegram.ps1"
if errorlevel 1 (
  echo.
  echo Configuration failed. Check the message above.
  pause
  exit /b 1
)

echo.
echo Configuration completed. You can now run Start-MiniMax-H3-Telegram.cmd.
pause
endlocal
