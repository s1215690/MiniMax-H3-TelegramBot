$ErrorActionPreference = 'Stop'

Write-Host 'Revoke the old token in @BotFather first, then enter the new token. Input is hidden.'
$secureToken = Read-Host 'New Telegram Bot Token' -AsSecureString
$tokenPtr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureToken)
try {
    $token = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($tokenPtr)
}
finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($tokenPtr)
}

$chatId = (Read-Host 'Your Telegram Chat ID').Trim()
if ([string]::IsNullOrWhiteSpace($token) -or [string]::IsNullOrWhiteSpace($chatId)) {
    throw 'Token and Chat ID are required.'
}

[Environment]::SetEnvironmentVariable('MINIMAX_TELEGRAM_BOT_TOKEN', $token, 'User')
[Environment]::SetEnvironmentVariable('MINIMAX_TELEGRAM_CHAT_ID', $chatId, 'User')
Write-Host 'Saved to Windows user environment variables. Start a new terminal before launching the Bot.'
