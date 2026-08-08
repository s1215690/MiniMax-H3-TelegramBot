@echo off
setlocal
if "%MINIMAX_TELEGRAM_BOT_TOKEN%"=="" (
  echo Missing MINIMAX_TELEGRAM_BOT_TOKEN. Run Configure-MiniMax-H3-Telegram.ps1 first.
  pause
  exit /b 2
)
if "%MINIMAX_TELEGRAM_CHAT_ID%"=="" (
  echo Missing MINIMAX_TELEGRAM_CHAT_ID. Run Configure-MiniMax-H3-Telegram.ps1 first.
  pause
  exit /b 2
)
wscript.exe //nologo "%~dp0Start-MiniMax-H3-Telegram.vbs"
endlocal
