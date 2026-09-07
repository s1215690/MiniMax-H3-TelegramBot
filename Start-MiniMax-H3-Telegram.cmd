@echo off
setlocal
rem Clear environment pollution (Hermes/Git Bash may set PYTHONPATH to its own venv).
set "PYTHONPATH="
set "PYTHONHOME="
rem [TEST] Disable two-stage latent upscaling to diagnose line artifacts
set "MINIMAX_H3_LATENT_UPSCALE=0"
rem [TEST] Disable Motion Context (latent format incompatible after official-node migration)
set "MINIMAX_H3_LONG_CONTINUITY=off"
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
