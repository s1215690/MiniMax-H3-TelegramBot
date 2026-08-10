@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Stop-MiniMax-H3-Turbo.ps1"
if errorlevel 1 pause
