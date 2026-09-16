@echo off
setlocal
rem Restart the MiniMax H3 Telegram Bot: stop the running bot process, then
rem relaunch it hidden via the same VBS launcher used by Start-MiniMax-H3-Telegram.cmd.

rem Clear environment pollution (Hermes/Git Bash may set PYTHONPATH to its own venv).
set "PYTHONPATH="
set "PYTHONHOME="
rem [TEST] Disable two-stage latent upscaling to diagnose line artifacts
set "MINIMAX_H3_LATENT_UPSCALE=0"
rem [TEST] Disable Motion Context (latent format incompatible after official-node migration)
set "MINIMAX_H3_LONG_CONTINUITY=off"

echo === Restarting MiniMax H3 Telegram Bot ===

rem Refuse to restart while a generation job is running: killing the bot mid-job
rem loses a finished clip (never delivered to Telegram) and breaks any auto-chain
rem waiting on it. Read the last heartbeat line from the bot log first; a stale
rem heartbeat (older than 5 minutes) means the bot is gone, so restart anyway.
set "BOTDIR="
if defined MINIMAX_TELEGRAM_STATE for %%I in ("%MINIMAX_TELEGRAM_STATE%") do set "BOTDIR=%%~dpI"
if not defined BOTDIR set "BOTDIR=E:\MiniMax-H3-Telegram\runtime\bot\"
if "%BOTDIR:~-1%"=="\" set "BOTDIR=%BOTDIR:~0,-1%"
set "BOTLOG=%BOTDIR%\bot.log"

powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$line = Get-Content -LiteralPath '%BOTLOG%' -Tail 60 -ErrorAction SilentlyContinue | Where-Object { $_ -match 'heartbeat' } | Select-Object -Last 1; if ($line -and $line -notmatch 'job=idle') { if ($line -match '^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})') { $ts = [datetime]::ParseExact($matches[1], 'yyyy-MM-dd HH:mm:ss', $null); if (((Get-Date) - $ts).TotalMinutes -lt 5) { Write-Host ('  BUSY: ' + $line.Trim()); exit 7 } } }"
if errorlevel 7 (
  echo   [!] A generation job is running - see the last heartbeat above. Not restarting; run this again once the job is done.
  pause
  exit /b 1
)

echo Stopping running Bot process...

rem Kill only the python process whose command line references the Bot script.
rem $self excludes this PowerShell process itself, otherwise the pattern would match its own command line.
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$self=$PID; $targets=@(Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne $self -and $_.CommandLine -like '*MiniMax-H3-Telegram-Bot.py*' }); if ($targets.Count -eq 0) { Write-Host '  No running Bot found.' } else { foreach ($p in $targets) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue; Write-Host ('  Stopped PID ' + $p.ProcessId) } }"

timeout /t 2 /nobreak >nul

echo Starting Bot (hidden)...
wscript.exe //nologo "%~dp0Start-MiniMax-H3-Telegram.vbs"
timeout /t 1 /nobreak >nul

echo === Done. Send /start or /menu in Telegram to confirm. ===
endlocal
