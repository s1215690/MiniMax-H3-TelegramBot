param(
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'

# Clear environment pollution (Hermes/Git Bash may set PYTHONPATH to its own venv).
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue

$comfyRoot = 'E:\Comfy\ComfyUI\ComfyUI-Turbo'
$baseDirectory = 'E:\Comfy\ComfyUI\ComfyUI'
$python = Join-Path $baseDirectory '.venv\Scripts\python.exe'
$inputDir = if ($env:MINIMAX_COMFY_INPUT) {
    $env:MINIMAX_COMFY_INPUT
} else {
    'E:\MiniMax-H3-Telegram\input'
}
$outputDir = if ($env:MINIMAX_COMFY_OUTPUT) {
    $env:MINIMAX_COMFY_OUTPUT
} else {
    'E:\MiniMax-H3-Telegram\output'
}
$stateDir = if ($env:MINIMAX_COMFY_STATE_DIR) {
    $env:MINIMAX_COMFY_STATE_DIR
} else {
    'E:\MiniMax-H3-Telegram\runtime\comfyui'
}
$userDir = Join-Path $stateDir 'user'
$workflowDir = Join-Path $userDir 'default\workflows'
$port = 8191
$url = "http://127.0.0.1:$port"

if (-not (Test-Path -LiteralPath $python)) {
    throw "ComfyUI Python was not found: $python"
}
if (-not (Test-Path -LiteralPath $baseDirectory)) {
    throw "ComfyUI model/custom-node base directory was not found: $baseDirectory"
}

$legacy = Get-NetTCPConnection -LocalPort 8190 -State Listen -ErrorAction SilentlyContinue
if ($legacy) {
    throw 'The Q3 ComfyUI server is still running on port 8190. Stop it before starting Turbo to avoid exhausting VRAM and RAM.'
}

$listener = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($listener) {
    $existing = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener.OwningProcess)"
    if ($existing.CommandLine -like '*main.py*--port 8191*') {
        Write-Host "MiniMax H3 Turbo ComfyUI is already running at $url"
        if (-not $NoBrowser) { Start-Process $url }
        exit 0
    }
    throw "Port $port is occupied by another process (PID $($listener.OwningProcess))."
}

New-Item -ItemType Directory -Path $workflowDir -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'MiniMax-H3-Turbo-Heretic-10GB.json') `
    -Destination (Join-Path $workflowDir 'MiniMax-H3-Turbo-Heretic-10GB.json') -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'MiniMax-H3-Turbo-Experimental-MultiRate-10GB.json') `
    -Destination (Join-Path $workflowDir 'MiniMax-H3-Turbo-Experimental-MultiRate-10GB.json') -Force

$databasePath = Join-Path $stateDir 'comfyui.db'
$databaseUrl = "sqlite:///$($databasePath.Replace('\', '/'))"
$stdout = Join-Path $stateDir 'comfy.stdout.log'
$stderr = Join-Path $stateDir 'comfy.stderr.log'
$memoryArguments = @('--lowvram')
if ($env:MINIMAX_SAGE_ATTENTION -notin @('0', 'false', 'no', 'off')) {
    $memoryArguments += '--use-sage-attention'
}

$arguments = @(
    '-s',
    'main.py',
    '--base-directory', $baseDirectory
) + $memoryArguments + @(
    '--disable-auto-launch',
    '--listen', '127.0.0.1',
    '--port', "$port",
    '--user-directory', $userDir,
    '--database-url', $databaseUrl,
    '--input-directory', $inputDir,
    '--output-directory', $outputDir
)

$process = Start-Process -FilePath $python `
    -ArgumentList $arguments `
    -WorkingDirectory $comfyRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -PassThru

$ready = $false
for ($i = 0; $i -lt 120; $i++) {
    try {
        Invoke-RestMethod -Uri "$url/system_stats" -TimeoutSec 2 | Out-Null
        $ready = $true
        break
    } catch {
        Start-Sleep -Seconds 1
    }
}

if (-not $ready) {
    $tail = Get-Content -LiteralPath $stderr -Tail 60 -ErrorAction SilentlyContinue
    throw "MiniMax H3 Turbo ComfyUI did not become ready.`n$($tail -join [Environment]::NewLine)"
}

Write-Host "MiniMax H3 Turbo ComfyUI is ready at $url"
Write-Host 'Memory mode: Turbo (--lowvram)'
Write-Host 'Attention: SageAttention'
Write-Host "Launcher PID: $($process.Id)"
Write-Host "Logs: $stateDir"

if (-not $NoBrowser) { Start-Process $url }
