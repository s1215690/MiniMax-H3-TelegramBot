$ErrorActionPreference = 'Stop'

$port = 8191
$listener = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if (-not $listener) {
    Write-Host "MiniMax H3 Turbo ComfyUI is not running on port $port."
    exit 0
}

$child = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener.OwningProcess)"
if ($child.CommandLine -notlike '*main.py*--port 8191*') {
    throw "Refusing to stop unexpected process on port $port (PID $($child.ProcessId))."
}

$parentId = [int]$child.ParentProcessId
$parent = Get-CimInstance Win32_Process -Filter "ProcessId=$parentId" -ErrorAction SilentlyContinue

Stop-Process -Id $child.ProcessId -Force
Start-Sleep -Milliseconds 500

if ($parent -and $parent.CommandLine -like '*main.py*--port 8191*') {
    Stop-Process -Id $parentId -Force -ErrorAction SilentlyContinue
}

Write-Host 'MiniMax H3 Turbo ComfyUI stopped.'
