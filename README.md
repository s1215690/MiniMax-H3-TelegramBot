# MiniMax H3 Telegram Bot

這是一個在 Windows 上控制本機 ComfyUI MiniMax H3 Turbo 工作流的 Telegram Bot。
它可以從 Telegram 設定文字生視頻、圖片生視頻、片長、解析度、steps 和提示詞，生成完成後把 MP4 傳回 Telegram。

本專案只包含 Bot 和啟動腳本，不包含模型、VAE、ComfyUI、個人輸入圖片、輸出影片或 Telegram Token。

## 功能

- 文字生視頻（T2VA）和首幀圖片生視頻（I2VA）
- 5、10、12、15 秒短片，以及自訂總片長
- 長片自動分段生成並用 FFmpeg 合併
- 解析度、steps 和提示詞按鈕
- ComfyUI 啟動、停止、重啟和生成進度查詢
- 快速 Turbo（`--lowvram`）和極限顯存（`--novram --disable-smart-memory`）模式
- Windows 登入後自動啟動 Bot
- Token 只從 Windows 使用者環境變數讀取

## 先決條件

1. Windows 10/11、Python 3.10 或更新版本，以及可執行的 FFmpeg。
2. 已安裝 ComfyUI 和 MiniMax H3 T8 自訂節點：
   `MiniMaxH3AudioConditioningT8`、`MiniMaxH3DualClockSamplerT8` 等。
3. 已準備相容的 H3 模型、Qwen3-VL CLIP、Video VAE、Audio VAE 和 Turbo LoRA。
4. BotFather 建立的 Telegram Bot，以及自己的 Chat ID。

模型檔案名稱可以透過環境變數覆寫；請不要把模型檔案提交到這個 Repository。

## 安裝

```powershell
git clone https://github.com/s1215690/MiniMax-H3-TelegramBot.git
cd MiniMax-H3-TelegramBot
py -3 -m pip install -r requirements.txt
```

先設定 ComfyUI 路徑。最少需要設定以下變數；路徑請換成自己的安裝位置：

```powershell
[Environment]::SetEnvironmentVariable('MINIMAX_COMFY_DIR', 'D:\ComfyUI\ComfyUI-Turbo', 'User')
[Environment]::SetEnvironmentVariable('MINIMAX_COMFY_BASE_DIR', 'D:\ComfyUI\ComfyUI', 'User')
[Environment]::SetEnvironmentVariable('MINIMAX_COMFY_PYTHON', 'D:\ComfyUI\ComfyUI\.venv\Scripts\python.exe', 'User')
```

然後雙擊 `Configure-MiniMax-H3-Telegram.cmd`，輸入新的 Bot Token 和 Chat ID。
Token 輸入框會隱藏內容，Token 會保存到 Windows 使用者環境變數，不會寫入 Repository。

設定後請開一個新的終端機，再雙擊 `Start-MiniMax-H3-Telegram.cmd`。

## 環境變數

| 變數 | 用途 |
|---|---|
| `MINIMAX_TELEGRAM_BOT_TOKEN` | Telegram Bot Token，必須只放在本機環境變數 |
| `MINIMAX_TELEGRAM_CHAT_ID` | 允許使用 Bot 的 Chat ID |
| `MINIMAX_COMFY_DIR` | ComfyUI Turbo 執行目錄，內含 `main.py` |
| `MINIMAX_COMFY_BASE_DIR` | ComfyUI 模型和 custom nodes 的 base directory |
| `MINIMAX_COMFY_PYTHON` | 執行 ComfyUI 的 Python |
| `MINIMAX_T8_API_TEMPLATE` | T8 API 工作流 JSON；預設使用 `workflow/dual_clock_4step_api.json` |
| `MINIMAX_COMFY_OUTPUT` | ComfyUI output 目錄 |
| `MINIMAX_COMFY_INPUT` | ComfyUI input 目錄 |
| `MINIMAX_COMFY_VRAM_MODE` | `lowvram` 或 `novram` |
| `MINIMAX_COMFY_PORT` | ComfyUI API Port，預設 `8191` |
| `MINIMAX_FFMPEG` | FFmpeg 可執行檔路徑；不設定時使用 PATH 中的 `ffmpeg` |

模型檔名也可以覆寫：

```text
MINIMAX_VIDEO_VAE
MINIMAX_AUDIO_VAE
MINIMAX_CLIP
MINIMAX_UNET
MINIMAX_LORA
```

## Telegram 指令

```text
/start 或 /menu       開啟按鈕控制面板
/prompt                輸入提示詞
/image                 切換到圖片生視頻
/text                  切換到文字生視頻
/progress              查看生成進度
/status                查看 Bot 狀態
/comfy_status          查看 ComfyUI 狀態
/comfy_start           啟動 ComfyUI
/comfy_restart         重啟 ComfyUI
/comfy_stop            關閉 ComfyUI
/cancel                取消目前生成
```

也可以使用指令直接設定一段短片：

```text
/gen 736 416 8 10
cinematic bright daylight scene with smooth camera movement and clear synchronized sound
```

總片長超過 15 秒時，Bot 會使用上一段最後畫面作為下一段首幀，然後合併成一個 MP4。

## 顯存模式

- `lowvram`：速度較快，適合一般使用。
- `novram`：使用 `--novram --disable-smart-memory`，把更多模型資料卸載到系統 RAM；可能更慢，但適合顯存不足時測試。

Telegram 面板可以切換模式。切換會重啟 ComfyUI，並取消正在執行的生成工作。

## 安全提醒

- 不要把 Bot Token、Chat ID、`.env`、日誌、資料庫、輸入圖片或輸出影片提交到 Git。
- 如果 Token 曾經貼到聊天、截圖或公開網站，請立即在 `@BotFather` 撤銷並重新建立。
- Bot 只接受設定的 Chat ID，ComfyUI 預設只監聽 `127.0.0.1`。

## 授權

本專案採用 MIT License。MiniMax H3 模型、ComfyUI 和自訂節點各自遵循其原作者授權條款。
