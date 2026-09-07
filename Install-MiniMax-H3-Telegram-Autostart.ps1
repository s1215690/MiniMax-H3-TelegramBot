$ErrorActionPreference = 'Stop'

$taskName = 'MiniMax H3 Telegram Bot'
$pythonPath = 'E:\Comfy\ComfyUI\ComfyUI\.venv\Scripts\python.exe'
$botPath = Join-Path $PSScriptRoot 'MiniMax-H3-Telegram-Bot.py'
$launcherPath = Join-Path $PSScriptRoot 'Start-MiniMax-H3-Telegram.vbs'
$wscriptPath = Join-Path $env:SystemRoot 'System32\wscript.exe'

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "找不到 Python：$pythonPath"
}
if (-not (Test-Path -LiteralPath $botPath -PathType Leaf)) {
    throw "找不到 Bot 程式：$botPath"
}

$action = New-ScheduledTaskAction `
    -Execute $wscriptPath `
    -Argument ('//nologo "{0}"' -f $launcherPath) `
    -WorkingDirectory $PSScriptRoot
$trigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
$principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive `
    -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -Hidden `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings `
    -Force | Out-Null

Write-Host "Registered Windows logon autostart: $taskName"
Write-Host 'The Bot will start after the next Windows logon. ComfyUI starts only when requested.'
Write-Host 'To test now, configure the replacement token with Configure-MiniMax-H3-Telegram.cmd.'
