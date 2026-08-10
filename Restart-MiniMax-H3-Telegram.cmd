@echo off
setlocal
rem Restart the MiniMax H3 Telegram Bot: stop the running bot process, then
rem relaunch it hidden via the same VBS launcher used by Start-MiniMax-H3-Telegram.cmd.

rem Clear environment pollution (Hermes/Git Bash may set PYTHONPATH to its own venv).
set "PYTHONPATH="
set "PYTHONHOME="

echo === Restarting MiniMax H3 Telegram Bot ===
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
