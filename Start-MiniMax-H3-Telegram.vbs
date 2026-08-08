Option Explicit

Dim fso, shell, scriptDir, pythonPath, botPath, command
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
botPath = fso.BuildPath(scriptDir, "MiniMax-H3-Telegram-Bot.py")

shell.CurrentDirectory = scriptDir
pythonPath = shell.ExpandEnvironmentStrings("%MINIMAX_COMFY_PYTHON%")
If InStr(1, pythonPath, "%MINIMAX_COMFY_PYTHON%", vbTextCompare) > 0 Then
    command = "py -3 " & Chr(34) & botPath & Chr(34)
Else
    command = Chr(34) & pythonPath & Chr(34) & " " & Chr(34) & botPath & Chr(34)
End If

' Window style 0 keeps both the Bot and its console hidden.
shell.Run command, 0, False
