Option Explicit

Dim fso, shell, env, scriptDir, pythonPath, botPath, command
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")
Set env = shell.Environment("Process")

' Do not let another Python/Conda environment break the hidden boot launch.
env("PYTHONPATH") = ""
env("PYTHONHOME") = ""

scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
pythonPath = "E:\Comfy\ComfyUI\ComfyUI\.venv\Scripts\python.exe"
botPath = fso.BuildPath(scriptDir, "MiniMax-H3-Telegram-Bot.py")

shell.CurrentDirectory = scriptDir
command = Chr(34) & pythonPath & Chr(34) & " -u " & Chr(34) & botPath & Chr(34)

' Window style 0 keeps both the Bot and its console hidden.
Dim exitCode
exitCode = shell.Run(command, 0, True)
WScript.Quit exitCode
